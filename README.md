# Beyond Rule-of-Mixtures

Physics-informed Gaussian-process hardness prediction with learned effective atomic volumes.

HADEX data are not included. The interatomic-potential checkpoint used to compute the
volume surfaces is not distributed; the cached surfaces in `data/simulation/` are
sufficient for every analysis in the paper, and the checkpoint is available from the
authors on request.

## Install

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root.

## Layout

| Path | Contents |
|---|---|
| `lib/`, `models/`, `run.py` | the model: descriptors, physics prior, GP arms, training pipeline |
| `utils/` | grouped validation and baseline runners, uncertainty evaluation, diagnostics, figures |
| `analysis/` | learning curves, calibration, partial molar volumes, volume-anchor comparison, DFT check |
| `effective_volume_gp.py`, `hardness_predictor.py` | standalone training and prediction from composition |
| `data/borg/` | Borg spreadsheet, the 93-row room-temperature cohort, fold assignments for all four protocols and five seeds |
| `data/simulation/`, `data/dft/` | cached MLIP volume surfaces; DFT equation-of-state outputs |
| `configs/training_settings.json` | hyperparameters of every reported analysis |

## Prediction

```bash
python hardness_predictor.py --composition NbTaTiZr MoTiTa
```

The predictor uses the grouped-analysis training engine: RBF covariance, a
width-four correction head, a 0.05 correction bound, and 200 kernel-only plus
800 joint training steps. Use `--kernel matern52` for the matched covariance
comparison. Reduced step counts are useful for smoke checks and are not paper results.
`effective_volume_gp.py` retains the separate exploratory implementation;
`run.py` and `utils/paper_training.py` define the reported training configuration.
The learning-curve and volume-anchor analysis entry points use that same engine.
`configs/training_settings.json` records the current RBF revision and matched
HADEX Matérn sensitivity analyses.

## Screened HADEX analyses

The following workflow accepts local CSV files; it does not connect to MongoDB.
Every hardness record must have at least three retained indents and relative
population SD below 25%. The row CSV preserves analysis-row repetitions, while
the specimen CSV selects one passing record per parent specimen family. Both
must contain the same complete composition, XRD and microstructure inputs for
all arms. Required identity and screening columns are `sample_id`,
`qness_record_id`, `n_retained_indents`, `retained_sd_hv` and `HV`.
The validation CSV contains the 14 experimental alloys and their measured targets.

```bash
python -m utils.rerun_screened_hadex_kernels \
  --rows-csv /private/revised_hadex_analysis_rows.csv \
  --specimens-csv /private/revised_hadex_specimen_cohort.csv \
  --heldout-csv /private/heldout14_inputs.csv \
  --output-dir /private/hadex_results --workers 4
```

This runs both kernels for the mean-function controls, descriptor ablations,
grouped and row-level validation, external validation, correction-bound
sensitivity and bulk-reconstruction control. Baselines share the same cohorts
and folds. External fits exclude all validation parent IDs, including suffixed
sections. Predictive intervals include observation noise. Summaries report
paired comparisons and empirical cross-fitted recalibration; this recalibration
does not have an exact split-conformal coverage guarantee. Output manifests
check inputs and configurations before reusing cached fits. Use a separate
output directory for `--smoke` or a changed configuration. No Borg jobs are run
by this command.

## Per-datapoint acquisition summaries

```bash
python -m utils.export_hadex_measurement_summary \
  --rows /private/revised_hadex_analysis_rows.csv \
  --qness /private/selected_qness.csv \
  --sem /private/selected_sem.csv --xrd /private/selected_xrd.csv \
  --output /private/hadex_datapoint_measurements.csv
```

The exporter preserves one output row per analysis datapoint, including
repetitions. It joins hardness records by acquisition ID, and SEM/XRD records
by explicit acquisition ID or a unique selected specimen record. Output includes
recorded and retained indent counts, population SD and relative SD, load and
spacing, EDS map entries, SEM point captures and images, acquisition settings,
saved XRD inputs and a readable `measurement_abstract`. Missing scan metadata
remain blank with an explicit status. A companion `.columns.csv` defines units
and counting conventions. Indent exclusions are inferred from the stored mean
and standard error; the export does not invent individual rejection reasons.
Keep proprietary inputs and exported acquisition summaries outside this public
checkout; they are not included with the code.
