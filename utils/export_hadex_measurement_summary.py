"""Join audited acquisition metadata to each HADEX analysis row, without database access."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def export_summary(rows, qness, sem, xrd, output):
    """Preserve row order and repetitions; resolve acquisitions by record ID."""
    rows=rows.copy().reset_index(drop=True)
    required=['sample_id','formula','HV','qness_record_id','xrd_peak_count','core_phase_fraction_mean']
    missing=set(required)-set(rows)
    if missing:raise ValueError(f'Missing analysis columns: {sorted(missing)}')
    if rows.qness_record_id.isna().any():raise ValueError('Every hardness target needs its acquisition record ID')
    for name,frame in [('hardness',qness),('SEM',sem),('XRD',xrd)]:
        if 'record_id' not in frame or not frame.record_id.is_unique:raise ValueError(f'{name} metadata must contain unique record_id values')
    q=qness.set_index('record_id').reindex(rows.qness_record_id).reset_index(drop=True)
    if q.mean_hv.isna().any():raise ValueError('A hardness acquisition is missing from the audit')
    if not np.allclose(rows.HV,q.mean_hv):raise ValueError('Hardness targets do not match their acquisition records')
    if not np.array_equal(rows.sample_id.astype(str),q.sample_id.astype(str)):raise ValueError('Hardness record belongs to a different specimen')
    out=pd.DataFrame({'analysis_row_id':rows.analysis_row_id if 'analysis_row_id' in rows else np.arange(len(rows)), 'sample_id':rows.sample_id,'formula':rows.formula,'hardness_hv':rows.HV,'hardness_record_id':rows.qness_record_id})
    definitions={
        'analysis_row_id':'Analysis-row identifier; repeated acquisitions remain separate analysis rows.',
        'sample_id':'Stored specimen identifier.',
        'formula':'Composition used for this model datapoint.',
        'hardness_hv':'Target hardness: stored mean over retained indents (HV).',
        'hardness_record_id':'Hardness acquisition identifier; repeated rows may share this ID.',
    }
    fields={'n_raw_indents':('recorded_indent_count','Number of recorded hardness readings before retained-subset inference.'),'n_used_indents':('retained_indent_count','Number of indents in the subset matching the stored mean and standard error.'),'raw_sd_hv':('recorded_indent_sd_hv','Population SD across all recorded hardness readings (HV).'),'used_sd_hv':('retained_indent_sd_hv','Population SD across inferred retained readings (HV).'),'diagonal_um':('median_indent_diagonal_um','Median estimated diagonal, converted from exported millimetres to micrometres.'),'spacing_um':('median_nearest_indent_spacing_um','Median nearest-indent spacing within acquisition paths (micrometres).'),'peak_load_g':('median_peak_load_g','Median observed peak load across indents (grams).'),'dwell_s':('median_peak_dwell_s','Median time in the longest contiguous >=99% peak-load segment (seconds).')}
    for source,(column,description) in fields.items():out[column]=q[source];definitions[column]=description
    out['excluded_indent_count']=out.recorded_indent_count-out.retained_indent_count
    out['retained_relative_sd_pct']=100*out.retained_indent_sd_hv/out.hardness_hv
    out['screening_pass']=(out.retained_indent_count>=3)&(out.retained_relative_sd_pct>=0)&(out.retained_relative_sd_pct<25)&(out.hardness_hv>0)
    out['retained_sem_hv']=out.retained_indent_sd_hv/np.sqrt(out.retained_indent_count)
    definitions.update(excluded_indent_count='Recorded minus retained indent count; individual rejection reasons are unavailable.',retained_relative_sd_pct='100 times retained population SD divided by target mean hardness.',screening_pass='True when at least three retained indents, positive mean and relative population SD below 25%.',retained_sem_hv='Retained population SD divided by square root of retained count (HV).')
    if not out.screening_pass.all():raise ValueError('Input contains a hardness datapoint that fails the screen')
    def characterization(frame, prefix, id_column, columns):
        ids=rows[id_column].copy() if id_column in rows else pd.Series(np.nan,index=rows.index,dtype=object)
        needed=ids.isna()
        if frame.sample_id.duplicated().any() and needed.any():raise ValueError(f'{prefix}: ambiguous specimen fallback; supply explicit record IDs')
        fallback=frame.set_index('sample_id').record_id
        ids.loc[needed]=rows.loc[needed,'sample_id'].map(fallback)
        matched=frame.set_index('record_id').reindex(ids).reset_index(drop=True)
        present=matched.sample_id.notna()
        if not (matched.loc[present,'sample_id'].astype(str).to_numpy()==rows.loc[present,'sample_id'].astype(str).to_numpy()).all():raise ValueError(f'{prefix} acquisition belongs to a different specimen')
        out[id_column]=ids
        out[prefix+'_metadata_status']=np.where(present,'matched','unavailable')
        definitions[id_column]=prefix.upper()+' acquisition ID, explicit when supplied or resolved from the uniquely selected specimen record.'
        definitions[prefix+'_metadata_status']='Matched means audited acquisition metadata exist; unavailable remains blank in its numeric fields.'
        for source,(column,description) in columns.items():out[column]=matched[source];definitions[column]=description
    characterization(sem,'sem','sem_record_id',{'n_locations':('sem_point_capture_count','Number of stored SEM point-capture entries; not independently verified unique physical regions.'),'n_maps':('eds_map_count','Number of stored EDS map entries in the selected SEM record.'),'n_images':('sem_image_count','Number of stored images in the selected SEM record.'),'n_elements':('eds_element_count','Number of elements in the selected global EDS composition.'),'formula':('eds_global_formula','Global EDS composition from the audited selected map.'),'working_distance_mm':('sem_working_distance_mm','Median stored SEM working distance (mm).'),'voltage_kv':('sem_voltage_kv','Median stored SEM accelerating voltage (kV).'),'dwell_ns':('sem_pixel_dwell_ns','Median stored SEM pixel dwell (ns).'),'integrations':('sem_frame_integrations','Median stored SEM frame integrations.'),'pixel_width_nm':('sem_pixel_width_nm','Median stored pixel width (nm).')})
    characterization(xrd,'xrd','xrd_record_id',{'n_peaks':('matched_xrd_peak_count','Peak count of first matched scan; saved model input is reported separately.'),'duration_min':('xrd_scan_duration_min','Matched first-scan duration (minutes).'),'start_deg':('xrd_start_2theta_deg','Matched scan start angle (degrees 2theta).'),'stop_deg':('xrd_stop_2theta_deg','Matched scan stop angle (degrees 2theta).'),'step_deg':('xrd_step_deg','Matched scan step (degrees).'),'speed_deg_min':('xrd_scan_speed_deg_min','Matched scan speed (degrees/minute).')})
    if not out.sem_metadata_status.eq('matched').all():raise ValueError('A selected SEM acquisition is missing from the audit')
    out['model_xrd_peak_count']=rows.xrd_peak_count;out['core_phase_fraction']=rows.core_phase_fraction_mean
    definitions.update(model_xrd_peak_count='Saved XRD peak count actually used by the property model; retained even if scan metadata are unavailable.',core_phase_fraction='Stored core phase fraction used for the microstructure descriptors.')
    out['measurement_abstract']=[f'Hardness {r.hardness_hv:.2f} HV; {int(r.retained_indent_count)} retained of {int(r.recorded_indent_count)} recorded indents; retained SD {r.retained_indent_sd_hv:.2f} HV ({r.retained_relative_sd_pct:.2f}%); {int(r.eds_map_count)} EDS map entries, {int(r.sem_point_capture_count)} SEM point captures and {int(r.sem_image_count)} images; model XRD peak count {int(r.model_xrd_peak_count)}; XRD acquisition metadata {r.xrd_metadata_status}.' for r in out.itertuples()]
    definitions['measurement_abstract']='Human-readable acquisition summary for this analysis row; repeated acquisitions reuse metadata.'
    for column in ['recorded_indent_count','retained_indent_count','excluded_indent_count','sem_point_capture_count','eds_map_count','sem_image_count','eds_element_count','matched_xrd_peak_count','model_xrd_peak_count']:out[column]=out[column].astype('Int64')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(output,index=False)
    dictionary=output.with_name(output.stem+'.columns.csv');pd.DataFrame([{'column':c,'definition':definitions[c]} for c in out.columns]).to_csv(dictionary,index=False)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['rows','qness','sem','xrd','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=export_summary(*(pd.read_csv(getattr(args,n)) for n in ['rows','qness','sem','xrd']),args.output)
    print(f'Wrote {len(out)} analysis-row summaries to {args.output}; all hardness screens passed.')

if __name__=='__main__':main()
