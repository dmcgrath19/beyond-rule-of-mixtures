"""Resumable, matched RBF/Matérn analyses of the screened HADEX revision."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
from dataclasses import replace, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import argparse, hashlib, json, multiprocessing, time, traceback
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from run import add_priors, add_physics_intermediates, add_composition_intermediates, COMPOSITION_COLS, PHYS_COLS, STAGE1_ITERS, STAGE2_ITERS, _feature_cols_for, _train_gp_fold
from models import get_model_config
from utils.run_grouped_validation import balanced_group_folds, chemical_system, model_config, bulk_microstructure
from utils.run_grouped_baselines import predict_fold
from utils.evaluate_uncertainty import gaussian_crps, gaussian_nlpd

ROOT = Path(__file__).resolve().parents[1]
# Input files are supplied locally; no proprietary data or database credentials
# are shipped with this repository. Environment values also reach spawned workers.
OUT = Path(os.environ.get('HADEX_OUTPUT_DIR', str(ROOT/'results/hadex_kernel_rerun')))
INPUT_FILES = {}

MAIN_ARMS = ['composition', 'analytic', 'effective', 'composition_full', 'analytic_full', 'effective_full']
ABLATIONS = ['analytic_xrd', 'analytic_micro', 'shared_sigma']
BASELINES = ['curtin', 'vela', 'gao_rf', 'gao_svr']
KERNELS = ['rbf', 'matern52']
CACHE = {}


def configure_paths(rows_csv, specimens_csv, heldout_csv, output_dir, smoke=False):
    global OUT, INPUT_FILES, STAGE1_ITERS, STAGE2_ITERS
    INPUT_FILES = dict(rows=Path(rows_csv), specimens=Path(specimens_csv), heldout=Path(heldout_csv))
    OUT = Path(output_dir)
    for cohort, path in INPUT_FILES.items():
        os.environ['HADEX_'+cohort.upper()+'_CSV'] = str(path.resolve())
    os.environ['HADEX_OUTPUT_DIR'] = str(OUT.resolve())
    os.environ['HADEX_SMOKE'] = '1' if smoke else '0'
    CACHE.clear()
    if smoke:
        import run as engine
        engine.STAGE1_ITERS = STAGE1_ITERS = 2
        engine.STAGE2_ITERS = STAGE2_ITERS = 2


def atomic_csv(df, path):
    temp = path.with_suffix('.tmp')
    df.to_csv(temp, index=False)
    temp.replace(path)


def prepare(frame):
    frame = frame.copy().reset_index(drop=True)
    frame = add_composition_intermediates(add_physics_intermediates(add_priors(frame)))
    assert np.isfinite(frame[COMPOSITION_COLS+PHYS_COLS+['HV','HV_prior']].to_numpy(float)).all()
    assert np.allclose(frame[COMPOSITION_COLS].sum(axis=1), 1)
    return frame


def inputs(cohort):
    if cohort not in CACHE:
        path = INPUT_FILES.get(cohort) or os.environ.get('HADEX_'+cohort.upper()+'_CSV')
        if not path:
            raise ValueError(f'Supply the {cohort} input CSV explicitly')
        frame = pd.read_csv(path).rename(columns={'HV_ground_truth':'HV'})
        if cohort != 'heldout':
            required = ['sample_id','qness_record_id','n_retained_indents','retained_sd_hv','HV']
            missing = set(required)-set(frame)
            if missing:raise ValueError(f'Missing screening columns: {sorted(missing)}')
            screen = frame[['n_retained_indents','retained_sd_hv','HV']].to_numpy(float)
            rsd = 100*frame.retained_sd_hv/frame.HV
            if not np.isfinite(screen).all() or not ((frame.n_retained_indents>=3)&(frame.n_retained_indents%1==0)&(frame.HV>0)&(rsd>=0)&(rsd<25)).all():
                raise ValueError('Input must be screened: >=3 retained indents and relative population SD <25%')
            if frame.sample_id.isna().any() or frame.qness_record_id.isna().any():raise ValueError('Specimen and hardness acquisition IDs are required')
            family = frame.sample_id.astype(str).str.extract(r'^(RAI-\d+)')[0]
            if family.isna().any():raise ValueError('HADEX IDs must identify an RAI parent specimen')
            frame['conservative_family_group'] = family
            frame['retained_rsd_pct'] = rsd
            if cohort == 'specimens' and family.duplicated().any():raise ValueError('Primary specimen CSV must contain one selected passing record per parent family')
        elif len(frame)!=14:
            raise ValueError('This paper workflow requires the 14-alloy validation CSV')
        generated = COMPOSITION_COLS+PHYS_COLS+['HV_prior']
        frame = frame.drop(columns=generated, errors='ignore')
        frame = prepare(frame)
        frame['row_id'] = np.arange(len(frame))
        frame['system'] = chemical_system(frame)
        if cohort != 'heldout':
            missing = set(_feature_cols_for(configuration('effective_full','rbf')))-set(frame)
            if missing:raise ValueError(f'Missing full-model inputs: {sorted(missing)}')
            if not np.isfinite(frame[_feature_cols_for(configuration('effective_full','rbf'))].to_numpy(float)).all():raise ValueError('Every arm must use the same complete-input cohort')
        CACHE[cohort] = frame
    return CACHE[cohort].copy()


def configuration(arm, kernel, beta=None):
    if arm in MAIN_ARMS:
        config = model_config(arm)
    elif arm == 'analytic_xrd':
        config = get_model_config('xrd_peak_count')
    elif arm == 'analytic_micro':
        config = get_model_config('microstructure')
    elif arm == 'shared_sigma':
        config = replace(get_model_config('gpytorch_nonlinear_sigma'), sigma_variant='shared')
    else:
        raise ValueError(arm)
    return replace(config, kernel=kernel, **({'sigma_log_bound':beta} if beta is not None else {}))


def tag(job):
    return '_'.join(str(job[k]) for k in ('kind','cohort','group','seed','arm','kernel')) + (f"_beta{job['beta']}" if 'beta' in job else '')


def folds_for(frame, group, seed):
    if group == 'random':
        return list(KFold(5, shuffle=True, random_state=42+seed).split(frame))
    values = frame.system if group == 'system' else frame.conservative_family_group
    folds = balanced_group_folds(values, seed=seed)
    for tr, te in folds:
        assert not set(values.iloc[tr]) & set(values.iloc[te])
    return folds


def predictions(job):
    if os.environ.get('HADEX_SMOKE') == '1':
        import run as engine
        engine.STAGE1_ITERS = engine.STAGE2_ITERS = 2
    torch.set_num_threads(1)
    started = time.monotonic()
    name = tag(job)
    path = OUT/'predictions'/f'{name}.csv'
    if path.exists():
        return name, 0, 'cached'
    with (OUT/'logs'/f'{name}.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        frame = inputs(job['cohort'])
        if job['kind'] == 'bulk_reconstruction':
            frame = bulk_microstructure(frame)
        test = inputs('heldout') if job['kind'] == 'heldout' else None
        if test is not None:
            # Worksheet IDs omit RAI- and database records may have piece suffixes.
            heldout_families = set('RAI-'+test.sample_id.astype(str).str.extract(r'(\d+)')[0])
            overlap = frame.conservative_family_group.isin(heldout_families)
            print('Excluded held-out parent families:', sorted(frame.loc[overlap,'conservative_family_group'].unique()), flush=True)
            frame = frame.loc[~overlap].reset_index(drop=True)
            assert not set(frame.conservative_family_group) & heldout_families
        splits = [(np.arange(len(frame)), np.arange(len(test)))] if test is not None else folds_for(frame, job['group'], job['seed'])
        parts, prefactors = [], []
        arm, kernel = job['arm'], job['kernel']
        for fold, (tr, te) in enumerate(splits):
            train = frame.iloc[tr].reset_index(drop=True)
            target = (test.iloc[te] if test is not None else frame.iloc[te]).reset_index(drop=True)
            torch.manual_seed(job['seed'])
            np.random.seed(job['seed'])
            diag = None
            if arm in BASELINES:
                if arm == 'curtin':
                    mu = target.HV_prior.to_numpy(float)
                    latent = sd = np.full(len(target), np.sqrt(np.mean((train.HV-train.HV_prior)**2)))
                else:
                    mu, sd = predict_fold(arm, 'hadex', train, target, seed=job['seed']*10+fold)
                    latent = sd.copy()
                prior = target.HV_prior.to_numpy(float)
                columns = _feature_cols_for(model_config('analytic'))+['phys_T']
            else:
                config = configuration(arm, kernel, job.get('beta'))
                columns = _feature_cols_for(config)+['phys_T']
                def capture(model, likelihood, scaler, ymean, ysd):
                    rec = dict(fold=fold, observation_noise_sd_hv=float(likelihood.noise.sqrt().item()*ysd))
                    for key in ('alpha','taylor','hv_scale','thermal_inv_c','thermal_exp','line_tension'):
                        parameter = getattr(model.mean_module, 'log_'+key, None)
                        if parameter is not None:
                            rec[key] = parameter.exp().item()
                    prefactors.append(rec)
                mu, latent, prior, diag = _train_gp_fold(train[columns].to_numpy(float), target[columns].to_numpy(float), train[PHYS_COLS].to_numpy(float), target[PHYS_COLS].to_numpy(float), train[COMPOSITION_COLS].to_numpy(float), target[COMPOSITION_COLS].to_numpy(float), train.HV.to_numpy(float), f'{name} fold{fold}', config, capture_fit=capture)
                sd = diag.predictive_std
            sd = np.asarray(sd, dtype=float)
            assert np.isfinite(mu).all() and np.isfinite(sd).all()
            assert (sd >= 0).all() if arm in BASELINES else (sd > 0).all()
            actual = target.HV.to_numpy(float)
            positive = sd > 0
            crps = np.abs(actual-mu)
            nlpd = np.full(len(actual), np.nan)
            crps[positive] = gaussian_crps(actual[positive],mu[positive],sd[positive])
            nlpd[positive] = gaussian_nlpd(actual[positive],mu[positive],sd[positive])
            scaler = StandardScaler().fit(train[columns])
            distance = cdist(scaler.transform(target[columns]), scaler.transform(train[columns])).min(axis=1)
            block = target[['row_id','sample_id','formula','system']].copy()
            for col in ['qness_record_id','conservative_family_group','retained_rsd_pct','n_retained_indents']:
                if col in target:
                    block[col] = target[col].to_numpy()
            block = block.assign(fold=fold, actual=target.HV.to_numpy(float), prediction=mu, latent_sd=latent, predictive_sd=sd, prior=prior, nearest_train_distance=distance, abs_error=np.abs(target.HV.to_numpy(float)-mu), crps=crps, nlpd=nlpd)
            if diag is not None:
                block['sigma'] = diag.sigma
                for prefix, array in [('Veff_',diag.effective_volumes), ('dV_',diag.delta_volumes), ('sigma_contribution_',diag.contributions)]:
                    if array is not None:
                        assert array.shape[1] == len(COMPOSITION_COLS)
                        for i, c in enumerate(COMPOSITION_COLS):
                            block[prefix+c[5:]] = array[:,i]
            parts.append(block)
        result = pd.concat(parts).sort_values('row_id')
        expected = len(test) if test is not None else len(frame)
        assert np.array_equal(result.row_id.to_numpy(), np.arange(expected))
        if prefactors:
            atomic_csv(pd.DataFrame(prefactors), OUT/'prefactors'/f'{name}.csv')
        atomic_csv(result, path)
    return name, time.monotonic()-started, 'computed'


def job_list():
    jobs = []
    def add(kind,cohort,group,seed,arm,kernel,**kw):
        jobs.append(dict(kind=kind,cohort=cohort,group=group,seed=seed,arm=arm,kernel=kernel,**kw))
    # Main specimen-grouped and chemical-system validation plus descriptor ablations.
    for group in ['sample','system']:
        for seed in range(5):
            for arm in MAIN_ARMS+ABLATIONS:
                for kernel in KERNELS:
                    add('cv','specimens',group,seed,arm,kernel)
            for arm in BASELINES:
                add('cv','specimens',group,seed,arm,'baseline')
    # Row-level evaluation and a grouped check of its weighting.
    for group in ['random','sample']:
        for arm in MAIN_ARMS+ABLATIONS:
            for kernel in KERNELS:
                add('cv','rows',group,0,arm,kernel)
        for arm in BASELINES:
            add('cv','rows',group,0,arm,'baseline')
    for seed in range(5):
        for arm in MAIN_ARMS:
            for kernel in KERNELS:
                add('heldout','specimens','external14',seed,arm,kernel)
    for arm in BASELINES:
        add('heldout','specimens','external14',0,arm,'baseline')
    # Bound sensitivity and core=shell=bulk reconstruction checks.
    for kernel in KERNELS:
        for beta in [.0,.01,.02,.10]: # .05 comes from main CV
            add('beta','specimens','sample',0,'effective_full',kernel,beta=beta)
        for group in ['sample','system']:
            for seed in range(5):
                add('bulk_reconstruction','specimens',group,seed,'effective_full',kernel)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    for name in ['rows-csv','specimens-csv','heldout-csv']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=OUT)
    ap.add_argument('--smoke',action='store_true',help='Two steps per stage, two kernel jobs only; not paper results')
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--only', help='Optional substring selecting resumable jobs')
    ap.add_argument('--summarize-only',action='store_true')
    args = ap.parse_args()
    configure_paths(args.rows_csv,args.specimens_csv,args.heldout_csv,args.output_dir,args.smoke)
    for name in ['predictions','logs','prefactors','figures','tables']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    jobs = job_list()
    sources = ['utils/rerun_screened_hadex_kernels.py','run.py','models/model_configs.py','models/hardness_gp.py','models/sigma_models.py','lib/physics.py','utils/run_grouped_baselines.py','utils/run_grouped_validation.py']
    source_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    metadata = dict(kernels=KERNELS, stages=[STAGE1_ITERS,STAGE2_ITERS], arms={arm:asdict(configuration(arm,'rbf')) for arm in MAIN_ARMS+ABLATIONS}, screening=dict(minimum_retained_indents=3,rsd_percent_exclusive_maximum=25), heldout_exclusion='Exclude all 14 worksheet parent RAI IDs, including suffixed specimens, before each external-validation fit.', source_sha256=source_hashes, jobs=jobs, cohorts={})
    for cohort in ['rows','specimens','heldout']:
        frame = inputs(cohort)
        metadata['cohorts'][cohort] = dict(n_rows=len(frame),sha256=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest())
        atomic_csv(frame,OUT/f'{cohort}_inputs.csv')
    manifest = OUT/'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError('Existing output configuration differs; archive the output or use a separate output directory')
    manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    if args.summarize_only:
        from utils.summarize_screened_hadex_kernels import main as summarize
        summarize()
        return
    selected = [job for job in jobs if args.only is None or args.only in tag(job)]
    if args.smoke:selected=[j for j in selected if j['kind']=='cv' and j['cohort']=='specimens' and j['group']=='sample' and j['arm']=='effective_full' and j['seed']==0]
    fits = sum(1 if j['kind']=='heldout' else 5 for j in selected)
    print(f'{len(selected)} jobs, {fits} fits, workers={args.workers}',flush=True)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(predictions,job):job for job in selected}
        for count, future in enumerate(as_completed(futures),1):
            try:
                name,elapsed,status = future.result()
                print(f'COMPLETE {count}/{len(selected)} {name} {status} {elapsed:.1f}s',flush=True)
            except Exception as error:
                job = futures[future]
                failure = dict(job=job,error=str(error),traceback=traceback.format_exc())
                failures.append(failure)
                print('FAILED '+tag(job)+' '+str(error),flush=True)
                (OUT/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed; inspect failures.json')
    missing = [tag(j) for j in jobs if not (OUT/'predictions'/f'{tag(j)}.csv').exists()]
    if not missing and not args.smoke:
        from utils.summarize_screened_hadex_kernels import main as summarize
        summarize()
        (OUT/'completion.json').write_text(json.dumps(dict(status='complete',jobs=len(jobs),fits=sum(1 if j['kind']=='heldout' else 5 for j in jobs)),indent=2)+'\n')
    else:
        print(f'{len(missing)} planned jobs remain',flush=True)

if __name__ == '__main__':
    main()
