"""Generate audited tables and plots for the revised two-kernel HADEX analyses."""
from pathlib import Path
import hashlib, html, json
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.evaluate_uncertainty import cluster_bootstrap_diff
from utils.rerun_screened_hadex_kernels import OUT, tag, atomic_csv, KERNELS, MAIN_ARMS, BASELINES

LABEL = {'composition':'Composition GP','analytic':'Analytic PI-GP','effective':'Effective volume','composition_full':'Composition + descriptors','analytic_full':'Analytic + descriptors','effective_full':'Effective + descriptors','analytic_xrd':'Analytic + XRD','analytic_micro':'Analytic + microstructure','shared_sigma':'Shared sigma','curtin':'Curtin prior','vela':'Vela','gao_rf':'Gao RF','gao_svr':'Gao SVR'}
KEY = ['kind','cohort','group','arm','kernel']
# Manuscript palette; observation whiskers remain neutral.
PARITY_COLORS = {'curtin':'#1f77b4', 'gao_rf':'#8c564b', 'gao_svr':'#e377c2',
                 'vela':'#7f7f7f', 'effective':'#1f77b4', 'effective_full':'#9467bd'}


def metrics(g):
    e = np.abs(g.actual.to_numpy()-g.prediction.to_numpy())
    sd = g.predictive_sd.to_numpy()
    return dict(n_rows=len(g),mae_hv=float(e.mean()),rmse_hv=float(np.sqrt(np.mean(e**2))),median_ae_hv=float(np.median(e)),coverage68=float(np.mean(e<=sd)),coverage95=float(np.mean(e<=norm.ppf(.975)*sd)),mean_sd_hv=float(sd.mean()),crps_hv=float(g.crps.mean()),nlpd=float(g.nlpd.mean()) if (sd>0).all() else np.nan,zero_sd_points=int((sd==0).sum()),sd_error_spearman=float(spearmanr(sd,e).statistic) if len(np.unique(sd))>1 else np.nan)


def quantile(scores, level=.9):
    scores = np.asarray(scores)
    rank = int(np.ceil((len(scores)+1)*level))
    return float(np.sort(scores)[min(rank,len(scores))-1])


def calibrate(g):
    g = g.copy()
    e = np.abs(g.actual-g.prediction).to_numpy()
    sd = g.predictive_sd.to_numpy()
    distance = g.nearest_train_distance.to_numpy()
    folds = g.fold.to_numpy()
    score = e/sd
    widths = {'raw':norm.ppf(.95)*sd,'global_scale':np.zeros(len(g)),'global_conformal':np.zeros(len(g)),'stratified_conformal':np.zeros(len(g))}
    far = np.zeros(len(g),dtype=bool)
    for fold in np.unique(folds):
        te,tr = folds==fold,folds!=fold
        threshold = np.median(distance[tr])
        far[te] = distance[te]>threshold
        widths['global_scale'][te] = norm.ppf(.95)*np.sqrt(np.mean(score[tr]**2))*sd[te]
        widths['global_conformal'][te] = quantile(score[tr])*sd[te]
        for stratum in [False,True]:
            calibration = tr & ((distance>threshold)==stratum)
            target = te & ((distance>threshold)==stratum)
            widths['stratified_conformal'][target] = quantile(score[calibration])*sd[target]
    records=[]
    for name, half in widths.items():
        for regime,mask in [('all',np.ones(len(g),bool)),('near',~far),('far',far)]:
            records.append(dict(method=name,regime=regime,n=int(mask.sum()),coverage90=float(np.mean(e[mask]<=half[mask])),mean_width_hv=float(2*half[mask].mean())))
        g[name+'_halfwidth_hv'] = half
    g['far_stratum'] = far
    return records,g


def savefig(fig,name):
    for ext in ['png','pdf']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def parity(ax,g,title,color="#1f77b4"):
    ax.errorbar(g.actual,g.prediction,yerr=g.predictive_sd,fmt='o',ms=3,elinewidth=.6,alpha=.6,color=color,ecolor='#2f2f2f')
    low=min(g.actual.min(),g.prediction.min())-15;high=max(g.actual.max(),g.prediction.max())+15
    ax.plot([low,high],[low,high],'k--',lw=1)
    ax.set(title=title,xlabel='Measured hardness (HV)',ylabel='Predicted hardness (HV)')
    ax.text(.04,.95,f'MAE {np.mean(np.abs(g.actual-g.prediction)):.1f} HV; n={len(g)}',transform=ax.transAxes,va='top',fontsize=8)
    ax.grid(alpha=.2)


def main():
    configuration=json.loads((OUT/'configuration.json').read_text())
    jobs=configuration['jobs'];blocks={};runs=[];folds=[];cal=[];intervals=[]
    for job in jobs:
        name=tag(job);path=OUT/'predictions'/f'{name}.csv'
        if not path.exists():raise FileNotFoundError(name)
        g=pd.read_csv(path).sort_values('row_id')
        expected=configuration['cohorts']['heldout' if job['kind']=='heldout' else job['cohort']]['n_rows']
        assert np.array_equal(g.row_id,np.arange(expected)),name
        assert np.isfinite(g[['actual','prediction','predictive_sd']].to_numpy()).all(),name
        context={key:job[key] for key in KEY+['seed']}
        context['beta']=job.get('beta',.05 if job['arm'].startswith('effective') else np.nan)
        runs.append(context|metrics(g));blocks[name]=g
        for fold,b in g.groupby('fold'):
            folds.append(context|dict(fold=fold)|metrics(b))
        if job['kind']=='cv' and job['cohort']=='specimens' and job['kernel'] in KERNELS:
            rec,intv=calibrate(g)
            cal.extend(context|r for r in rec)
            intv=intv.assign(**context)
            intervals.append(intv)
    run=pd.DataFrame(runs);atomic_csv(run,OUT/'run_metrics.csv');atomic_csv(pd.DataFrame(folds),OUT/'fold_metrics.csv')
    summary=run.groupby(KEY+['beta'],dropna=False).agg(n_rows=('n_rows','first'),repeats=('seed','size'),mae_hv=('mae_hv','mean'),mae_sd=('mae_hv','std'),rmse_hv=('rmse_hv','mean'),coverage68=('coverage68','mean'),coverage95=('coverage95','mean'),mean_sd_hv=('mean_sd_hv','mean'),crps_hv=('crps_hv','mean'),nlpd=('nlpd','mean'),zero_sd_points=('zero_sd_points','sum')).reset_index()
    atomic_csv(summary,OUT/'summary.csv')
    calibration=pd.DataFrame(cal);atomic_csv(calibration,OUT/'recalibration_run_metrics.csv')
    cs=calibration.groupby(KEY+['method','regime']).agg(coverage90=('coverage90','mean'),coverage_sd=('coverage90','std'),mean_width_hv=('mean_width_hv','mean')).reset_index();atomic_csv(cs,OUT/'recalibration_summary.csv')
    atomic_csv(pd.concat(intervals,ignore_index=True),OUT/'recalibration_point_intervals.csv')
    # Paired comparisons use seed-averaged errors on exactly the same rows.
    errors={};first={}
    for key,r in run.groupby(KEY):
        values=[]
        for _,row in r.iterrows():
            job=next(j for j in jobs if all(j[k]==row[k] for k in KEY+['seed']) and (row['kind']!='beta' or j.get('beta')==row.beta))
            g=blocks[tag(job)]
            if key in first:
                assert np.array_equal(g[['row_id','actual','formula']].to_numpy(),first[key][['row_id','actual','formula']].to_numpy())
            first[key]=g;values.append(g.abs_error.to_numpy())
        errors[key]=np.mean(values,axis=0)
    pairs=[]
    for cohort,group in [('specimens','sample'),('specimens','system'),('rows','random'),('rows','sample')]:
        for arm in MAIN_ARMS+['analytic_xrd','analytic_micro','shared_sigma']:
            rk=('cv',cohort,group,arm,'rbf');mk=('cv',cohort,group,arm,'matern52')
            delta,lo,hi,p=cluster_bootstrap_diff(errors[mk]-errors[rk],first[rk].system.to_numpy())
            pairs.append(dict(cohort=cohort,group=group,arm=arm,rbf_advantage_hv=delta,ci_lo=lo,ci_hi=hi,p_boot=p))
    atomic_csv(pd.DataFrame(pairs),OUT/'paired_kernel_differences.csv')
    pairs=[]
    for cohort,group in [('specimens','sample'),('specimens','system'),('rows','random'),('rows','sample')]:
        for kernel in KERNELS:
            for effective,controls in [('effective',['composition','analytic']+BASELINES),('effective_full',['composition_full','analytic_full','effective']+BASELINES)]:
                ek=('cv',cohort,group,effective,kernel)
                for control in controls:
                    ck=('cv',cohort,group,control,'baseline' if control in BASELINES else kernel)
                    delta,lo,hi,p=cluster_bootstrap_diff(errors[ck]-errors[ek],first[ek].system.to_numpy())
                    pairs.append(dict(cohort=cohort,group=group,kernel=kernel,effective=effective,control=control,effective_advantage_hv=delta,ci_lo=lo,ci_hi=hi,p_boot=p))
    atomic_csv(pd.DataFrame(pairs),OUT/'paired_model_differences.csv')
    # External-alloy predictions averaged over independent initialization seeds.
    held=[]
    for arm in MAIN_ARMS+BASELINES:
        for kernel in (['baseline'] if arm in BASELINES else KERNELS):
            matching=[blocks[tag(j)] for j in jobs if j['kind']=='heldout' and j['arm']==arm and j['kernel']==kernel]
            g=pd.concat(matching).groupby(['row_id','sample_id','formula','system'],as_index=False).agg(actual=('actual','first'),prediction=('prediction','mean'),predictive_sd=('predictive_sd','mean'),latent_sd=('latent_sd','mean'))
            g=g.assign(arm=arm,kernel=kernel,abs_error=np.abs(g.actual-g.prediction))
            held.append(g)
    held=pd.concat(held,ignore_index=True);atomic_csv(held,OUT/'heldout_mean_predictions.csv')
    hp=[]
    for kernel in KERNELS:
        g=held[held.arm.eq('effective_full')&held.kernel.eq(kernel)].sort_values('row_id')
        for control in ['vela','gao_rf','gao_svr','analytic_full','composition_full']:
            cg=held[held.arm.eq(control)&held.kernel.eq('baseline' if control in BASELINES else kernel)].sort_values('row_id')
            delta,lo,hi,p=cluster_bootstrap_diff(cg.abs_error.to_numpy()-g.abs_error.to_numpy(),g.system.to_numpy())
            hp.append(dict(kernel=kernel,control=control,effective_full_advantage_hv=delta,ci_lo=lo,ci_hi=hi,p_boot=p))
    atomic_csv(pd.DataFrame(hp),OUT/'heldout_paired_differences.csv')
    kernel_heldout=[]
    for arm in MAIN_ARMS:
        r=held[held.arm.eq(arm)&held.kernel.eq('rbf')].sort_values('row_id')
        m=held[held.arm.eq(arm)&held.kernel.eq('matern52')].sort_values('row_id')
        delta,lo,hi,p=cluster_bootstrap_diff(m.abs_error.to_numpy()-r.abs_error.to_numpy(),r.system.to_numpy())
        kernel_heldout.append(dict(arm=arm,rbf_advantage_hv=delta,ci_lo=lo,ci_hi=hi,p_boot=p))
    atomic_csv(pd.DataFrame(kernel_heldout),OUT/'heldout_paired_kernel_differences.csv')
    reconstruction=[]
    for group in ['sample','system']:
        for kernel in KERNELS:
            k=('cv','specimens',group,'effective_full',kernel)
            b=('bulk_reconstruction','specimens',group,'effective_full',kernel)
            delta,lo,hi,p=cluster_bootstrap_diff(errors[b]-errors[k],first[k].system.to_numpy())
            reconstruction.append(dict(group=group,kernel=kernel,microstructure_advantage_hv=delta,ci_lo=lo,ci_hi=hi,p_boot=p))
    atomic_csv(pd.DataFrame(reconstruction),OUT/'bulk_reconstruction_paired_differences.csv')
    # Distance bins are pooled separately by arm/kernel, with repeat variation saved.
    distances=[]
    for arm in ['effective','effective_full']:
        for kernel in KERNELS:
            js=[j for j in jobs if j['kind']=='cv' and j['cohort']=='specimens' and j['group']=='sample' and j['arm']==arm and j['kernel']==kernel]
            pool=pd.concat([blocks[tag(j)].assign(seed=j['seed']) for j in js],ignore_index=True)
            pool['distance_bin']=pd.qcut(pool.nearest_train_distance.rank(method='first'),6,labels=False)
            for (seed,bin_id),g in pool.groupby(['seed','distance_bin']):
                distances.append(dict(arm=arm,kernel=kernel,seed=seed,distance_bin=bin_id+1,mean_distance=g.nearest_train_distance.mean(),n=len(g),**metrics(g)))
    distance=pd.DataFrame(distances);atomic_csv(distance,OUT/'distance_bin_metrics.csv')
    prefactors=[]
    for job in jobs:
        path=OUT/'prefactors'/f'{tag(job)}.csv'
        if path.exists():prefactors.append(pd.read_csv(path).assign(**job))
    if prefactors:
        pref=pd.concat(prefactors,ignore_index=True);atomic_csv(pref,OUT/'prefactors.csv')
        means=pref[pref.kind.eq('cv')&pref.cohort.eq('specimens')].groupby(['group','arm','kernel']).agg({k:['mean','std','min','max'] for k in ['alpha','taylor','hv_scale','thermal_inv_c','thermal_exp','line_tension','observation_noise_sd_hv']})
        means.to_csv(OUT/'prefactor_summary.csv')
    # New replicate statistics count each hardness record exactly once.
    ri=pd.read_csv(OUT/'rows_inputs.csv').drop_duplicates('qness_record_id')
    rows=[]
    for col in ['n_retained_indents','retained_sd_hv','retained_rsd_pct']:
        x=ri[col];rows.append(dict(quantity=col,n=len(x),median=x.median(),q25=x.quantile(.25),q75=x.quantile(.75),minimum=x.min(),maximum=x.max()))
    atomic_csv(pd.DataFrame(rows),OUT/'replicate_summary.csv')
    # Publication-ready table fragments for each kernel.
    main=summary[summary.kind.eq('cv')&summary.cohort.eq('specimens')]
    for kernel in KERNELS:
        table=main[main.kernel.isin([kernel,'baseline'])].pivot(index='arm',columns='group',values='mae_hv')
        atomic_csv(table.reset_index(),OUT/'tables'/f'grouped_mae_{kernel}.csv')
        lines=[r'\begin{tabular}{lrr}',r'\hline',r'Model & Specimen MAE (HV) & System MAE (HV) \\',r'\hline']
        for arm in MAIN_ARMS+['analytic_xrd','analytic_micro','shared_sigma']+BASELINES:
            vals=[]
            for group in ['sample','system']:
                r=main[main.arm.eq(arm)&main.group.eq(group)&main.kernel.eq('baseline' if arm in BASELINES else kernel)].iloc[0]
                vals.append(f"${r.mae_hv:.1f} \\pm {r.mae_sd:.1f}$")
            lines.append(LABEL[arm]+' & '+' & '.join(vals)+r' \\')
        lines.extend([r'\hline',r'\end{tabular}'])
        (OUT/'tables'/f'grouped_mae_{kernel}.tex').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    order=MAIN_ARMS+BASELINES
    for ax,kernel in zip(axes,KERNELS):
        for offset,group,color in [(-.18,'sample','#2469a0'),(.18,'system','#c16b29')]:
            selected=main[main.arm.isin(order)&main.group.eq(group)&main.kernel.isin([kernel,'baseline'])].set_index('arm').loc[order]
            ax.barh(np.arange(len(order))+offset,selected.mae_hv,xerr=selected.mae_sd.fillna(0),height=.34,label=group,color=color)
        ax.set(yticks=np.arange(len(order)),yticklabels=[LABEL[a] for a in order],xlabel='MAE (HV)',title=kernel.upper());ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend()
    savefig(fig,'grouped_mae')
    for kernel in KERNELS:
        fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
        for ax,arm in zip(axes.flat,['curtin','gao_rf','gao_svr','vela','effective','effective_full']):
            j=next(j for j in jobs if j['kind']=='cv' and j['cohort']=='rows' and j['group']=='random' and j['arm']==arm and j['kernel']==('baseline' if arm in BASELINES else kernel))
            parity(ax,blocks[tag(j)],LABEL[arm],PARITY_COLORS[arm])
        fig.suptitle(f'{kernel.upper()}: 653 rows, row-level folds; repeated inputs can cross folds')
        savefig(fig,f'row_parity_{kernel}')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,kernel in zip(axes,KERNELS):
        g=held[held.arm.eq('effective_full')&held.kernel.eq(kernel)];parity(ax,g,kernel.upper())
    savefig(fig,'heldout14_parity')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    levels=np.linspace(.1,.99,40)
    for ax,kernel in zip(axes,KERNELS):
        for arm in ['composition_full','analytic_full','effective_full']:
            matching=[blocks[tag(j)] for j in jobs if j['kind']=='cv' and j['cohort']=='specimens' and j['group']=='sample' and j['arm']==arm and j['kernel']==kernel]
            g=pd.concat(matching);e=np.abs(g.actual-g.prediction);sd=g.predictive_sd
            ax.plot(levels,[np.mean(e<=norm.ppf((1+v)/2)*sd) for v in levels],label=LABEL[arm])
        ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Nominal coverage',ylabel='Empirical coverage',title=kernel.upper());ax.legend(fontsize=8);ax.grid(alpha=.2)
    savefig(fig,'calibration_reliability')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,kernel in zip(axes,KERNELS):
        for arm in ['effective','effective_full']:
            g=distance[distance.arm.eq(arm)&distance.kernel.eq(kernel)].groupby('distance_bin').agg(coverage=('coverage95','mean'),sd=('coverage95','std'))
            ax.errorbar(g.index,g.coverage,yerr=g.sd,marker='o',capsize=3,label=LABEL[arm])
        ax.axhline(.95,c='k',ls='--');ax.set(xlabel='Nearest-training-distance bin',ylabel='95% coverage',title=kernel.upper(),ylim=(0,1.02));ax.legend(fontsize=8);ax.grid(alpha=.2)
    savefig(fig,'coverage_vs_distance')
    beta=run[run.kind.eq('beta')].copy()
    base=run[run.kind.eq('cv')&run.cohort.eq('specimens')&run.group.eq('sample')&run.seed.eq(0)&run.arm.eq('effective_full')]
    beta=pd.concat([beta,base],ignore_index=True);atomic_csv(beta,OUT/'beta_sweep.csv')
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    for kernel,g in beta.groupby('kernel'):
        g=g.sort_values('beta');ax.plot(g.beta,g.mae_hv,'o-',label=kernel)
    ax.set(xlabel='Correction bound beta',ylabel='Specimen-grouped MAE (HV)');ax.legend();ax.grid(alpha=.2);savefig(fig,'beta_sweep')
    # Self-contained review report with explicit cohort and calibration limits.
    body='<h1>Screened HADEX: RBF and Matérn 5/2</h1><p>Screen: ≥3 retained indents and relative population SD &lt;25%. The analysis retains its row-level view and a separately grouped specimen view. Main evaluation uses one stored passing hardness record per conservative specimen family; the row-level plots retain repeated inputs.</p>'
    body+='<p>Both kernels use identical cohorts, folds, initialization seeds and training settings within each arm. Results below average five fold assignments. Baselines are fitted on the same cohorts and folds.</p>'
    body+='<h2>Specimen and system validation</h2>'+main[['group','arm','kernel','n_rows','mae_hv','mae_sd','rmse_hv','coverage95','crps_hv','nlpd']].round(3).to_html(index=False)
    body+='<img src="figures/grouped_mae.png" width="1100"><h2>Paired kernel differences</h2>'+pd.read_csv(OUT/'paired_kernel_differences.csv').query('cohort == "specimens"').round(3).to_html(index=False)
    body+='<h2>External 14-alloy validation</h2>'+summary[summary.kind.eq('heldout')][['arm','kernel','mae_hv','mae_sd','coverage95','crps_hv','nlpd']].round(3).to_html(index=False)+'<img src="figures/heldout14_parity.png" width="900">'
    body+='<h2>Uncertainty and screening limits</h2><p>Gaussian predictive SD includes observation noise. Recalibration uses the other outer folds’ out-of-fold residuals and is evaluated empirically; these dependent cross-fitted intervals do not carry an exact split-conformal coverage guarantee. Random-forest tree-spread SD can be zero: point-mass CRPS is used there, and Gaussian NLPD for a run with any zero-SD point is withheld. Curtin intervals use training residual RMS as a simple uncertainty reference.</p><p>Retained-indent subsets reproduce stored mean and SEM, but exclusion reasons are absent from the database. Multiphase additions without verified phase descriptors were withheld. Suffixes are grouped by parent specimen family conservatively. External validation excludes all 14 experimental parent specimen IDs before fitting; overlapping training families are removed. Composition/system overlap still requires interpreting this as transfer within represented systems.</p>'
    body+='<img src="figures/calibration_reliability.png" width="900"><img src="figures/coverage_vs_distance.png" width="900"><h2>Replicate statistics</h2>'+pd.DataFrame(rows).round(3).to_html(index=False)
    body+='<h2>Learned-volume surfaces</h2><p>Volume heads are bounded latent corrections learned from hardness, rather than independent measurements of atomic volume. The figures use the most nonlinear element in each ternary and a plane residual to quantify nonlinearity.</p><img src="figures/learned_volume_surfaces_rbf.png" width="1100"><img src="figures/learned_volume_surfaces_matern52.png" width="1100"><h2>Result files</h2><ul>'+''.join(f'<li><a href="{html.escape(x)}">{html.escape(x)}</a></li>' for x in ['summary.csv','paired_model_differences.csv','recalibration_summary.csv','heldout_mean_predictions.csv','heldout_paired_differences.csv','heldout_paired_kernel_differences.csv','bulk_reconstruction_paired_differences.csv','volume_surface_plane_metrics.csv','heldout_overlap_audit.json','prefactor_summary.csv','beta_sweep.csv','configuration.json'])+'</ul>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px Arial;margin:30px;max-width:1300px}table{border-collapse:collapse;font-size:12px}td,th{padding:5px;border:1px solid #ddd}img{max-width:100%}</style>'+body)
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'predictions').glob('*.csv')}
    (OUT/'output_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(main[main.arm.isin(['effective','effective_full','vela'])][['group','arm','kernel','mae_hv','mae_sd','coverage95']].to_string(index=False),flush=True)
    print(f'Validated {len(jobs)} prediction files; saved summaries, paired comparisons, 8 figures, and report.html.',flush=True)

if __name__=='__main__':main()
