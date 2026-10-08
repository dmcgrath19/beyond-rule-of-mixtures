"""Check acquisition identity, missing-data handling and leakage boundaries."""
from pathlib import Path
import tempfile,unittest
from dataclasses import replace
import numpy as np
import pandas as pd
import gpytorch,torch
from models import get_model_config
from models.hardness_gp import HardnessGP
from models.sigma_models import SigmaModelSpec
from utils.export_hadex_measurement_summary import export_summary
from utils.rerun_screened_hadex_kernels import folds_for,configuration,job_list


class AcquisitionExportTests(unittest.TestCase):
    def setUp(self):
        self.rows=pd.DataFrame({'analysis_row_id':[10,11],'sample_id':['RAI-1']*2,'formula':['NbTi']*2,'HV':[100.]*2,'qness_record_id':['q1']*2,'xrd_peak_count':[7]*2,'core_phase_fraction_mean':[.4]*2,'sem_record_id':['s1']*2,'xrd_record_id':['x1']*2})
        self.q=pd.DataFrame({'sample_id':['RAI-1'],'record_id':['q1'],'mean_hv':[100.],'n_raw_indents':[5],'n_used_indents':[3],'raw_sd_hv':[20.],'used_sd_hv':[10.],'diagonal_um':[50.],'spacing_um':[1500.],'peak_load_g':[1000.],'dwell_s':[10.]})
        self.sem=pd.DataFrame({'sample_id':['RAI-1'],'record_id':['s1'],'formula':['NbTi'],'n_locations':[4],'n_maps':[2],'n_images':[8],'n_elements':[2],'working_distance_mm':[9.],'voltage_kv':[20.],'dwell_ns':[200.],'integrations':[32.],'pixel_width_nm':[156.]})
        self.xrd=pd.DataFrame(columns=['sample_id','record_id','n_peaks','duration_min','start_deg','stop_deg','step_deg','speed_deg_min'])

    def export(self):
        with tempfile.TemporaryDirectory() as d:return export_summary(self.rows,self.q,self.sem,self.xrd,Path(d)/'summary.csv')

    def test_repetitions_preserve_identity_and_missing_scan_is_not_zero(self):
        out=self.export()
        self.assertEqual(out.analysis_row_id.tolist(),[10,11])
        self.assertEqual(out.retained_indent_count.tolist(),[3,3])
        self.assertEqual(out.eds_map_count.tolist(),[2,2])
        self.assertTrue(out.xrd_scan_duration_min.isna().all())
        self.assertTrue(out.xrd_metadata_status.eq('unavailable').all())
        self.assertEqual(out.model_xrd_peak_count.tolist(),[7,7])
        self.assertTrue(out.retained_relative_sd_pct.eq(10.).all())

    def test_wrong_hardness_acquisition_is_rejected(self):
        self.q['mean_hv']=101.
        with self.assertRaisesRegex(ValueError,'targets do not match'):self.export()

    def test_wrong_specimen_characterization_is_rejected(self):
        self.sem['sample_id']='RAI-2'
        with self.assertRaisesRegex(ValueError,'different specimen'):self.export()

    def test_screen_boundary_is_exclusive(self):
        self.q['used_sd_hv']=25.
        with self.assertRaisesRegex(ValueError,'fails the screen'):self.export()

    def test_duplicate_metadata_identity_is_rejected(self):
        self.q=pd.concat([self.q,self.q],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'unique record_id'):self.export()


class KernelAndFoldTests(unittest.TestCase):
    def test_kernels_keep_identical_arm_settings(self):
        for arm in ['composition','analytic','effective','composition_full','analytic_full','effective_full','shared_sigma','analytic_xrd','analytic_micro']:
            r=configuration(arm,'rbf');m=configuration(arm,'matern52')
            self.assertEqual(r,replace(m,kernel='rbf'))

    def test_parent_families_are_never_split(self):
        frame=pd.DataFrame({'conservative_family_group':np.repeat(['RAI-1','RAI-2','RAI-3','RAI-4','RAI-5'],2),'system':np.repeat(['Nb-Ti','Nb-Ta','Ta-Ti','Mo-Nb','Mo-Ta'],2)})
        for group,column in [('sample','conservative_family_group'),('system','system')]:
            folds=folds_for(frame,group,0)
            for tr,te in folds:self.assertFalse(set(frame.iloc[tr][column])&set(frame.iloc[te][column]))
            self.assertEqual(sorted(np.concatenate([te for _,te in folds]).tolist()),list(range(len(frame))))

    def test_every_gp_job_has_both_kernels(self):
        jobs=job_list();gp=[j for j in jobs if j['kernel']!='baseline']
        for j in gp:
            counterpart=dict(j,kernel='matern52' if j['kernel']=='rbf' else 'rbf')
            self.assertIn(counterpart,gp)

    def test_covariance_selection_and_invalid_name(self):
        x=torch.zeros((3,25),dtype=torch.float64);y=torch.zeros(3,dtype=torch.float64)
        def make(kernel):
            return HardnessGP(x,y,gpytorch.likelihoods.GaussianLikelihood().double(),n_features=3,physics_start=3,composition_start=8,n_composition_dims=17,y_mean=torch.tensor(0.),y_std=torch.tensor(1.),sigma_model_spec=SigmaModelSpec(),mean_type='constant',kernel=kernel)
        self.assertIsInstance(make('rbf').covar_module.base_kernel,gpytorch.kernels.RBFKernel)
        self.assertIsInstance(make('matern52').covar_module.base_kernel,gpytorch.kernels.MaternKernel)
        with self.assertRaisesRegex(ValueError,'Unknown kernel'):make('typo')

if __name__=='__main__':unittest.main()
