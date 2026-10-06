# Beyond Rule-of-Mixtures

Physics-informed Gaussian-process hardness prediction with learned effective atomic volumes.
The repository contains the public Borg benchmark, the fold assignments behind every Borg
result in the paper, the cached MLIP volume surfaces and the DFT outputs used to validate
them, and the code to regenerate every reported result.

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

## Regenerate the Borg results

```bash
for g in random formula reference system; do for s in 0 1 2 3 4; do
  for m in analytic effective composition; do
    python -m utils.run_grouped_validation --dataset borg --model $m --group $g --seed $s --fold-file data/borg/folds_${g}_seed$s.csv
  done
  python -m utils.run_grouped_baselines --dataset borg --group $g --seed $s
done; done
python -m utils.evaluate_uncertainty
python -m utils.borg_diagnostics
python -m utils.plot_public_results
```

Outputs are written to `results/`, which is not tracked. With the archived fold files the
effective-volume MAEs come out at 39.3 HV (shuffled), 43.9 HV (formula), 72.5 HV
(publication) and 68.4 HV (chemical system), to within seed variation.

## Refit Borg models

```bash
python -m utils.run_grouped_validation --dataset borg --model effective --group system --seed 0 --out-dir results/refits
python -m utils.run_grouped_baselines --dataset borg --group system --seed 0 --out-dir results/refits
```

Models: `analytic`, `effective`, `composition`. Groups: `random`, `formula`, `reference`,
`system`. `--log-bound` overrides the correction bound β for a sweep. The grouped cohort
is the 93 room-temperature rows in `data/borg/cohort.csv`; the single-split, calibration
and learning-curve analyses use all 99 rows. Initialisation states of the archived runs
were not saved, so fresh refits reproduce the tables to within seed variation rather than
bitwise.

## Prediction and learning curve

```bash
python effective_volume_gp.py --variants analytic effective_volume constant_mean
python hardness_predictor.py --composition NbTaTiZr MoTiTa
python analysis/learning_curve.py --repeats 150 --sizes 25 35 50 70
```

The predictor uses the same descriptor function for training and prediction. The
learning-curve runner saves its splits to `results/learning_curve/splits.json`; pass
`--split-in` to reuse them.

## Volume calculations

```bash
python analysis/predict_misfit.py --comp "Nb49.1 Ti16.1 W34.6" -o results/misfit.csv
python analysis/experimental_14_pmv.py
python analysis/experimental_14_anchors.py --seeds 3 -o results/experimental_14_anchors_refit.csv
python analysis/analyze_dft.py
```

These run from the cached surfaces in `data/simulation/` and the DFT outputs in
`data/dft/`. The analysis scripts use the published compositions of the 14 validation
alloys, not HADEX records.

## Training settings

All analyses use the same model: composition on the element basis through one hidden
tanh layer with a zero-initialised output, a bounded-exponential correction, and two-stage
Adam (kernel-only, then joint) with the physical prefactors re-optimised in every training
fold. The bound β is selected by out-of-fold MAE under formula-grouped cross-validation.
Hyperparameters by analysis:

| Analysis | Kernel (ARD) | Width | β | Steps (kernel-only / joint) | Learning rates |
|---|---|---|---|---|---|
| Grouped validation, repeated-protocol calibration and conformal, held-out alloys, stability, mass balance | RBF | 4 | 0.05 | 200 / 800 | 0.05 / 0.005 (head 0.001, weight decay 0.01, frozen for the first 600 joint steps) |
| Single-split controls, calibration and conformal tables, learning curves, prefactors, volume-anchor comparison | Matérn-5/2 | 32 | 0.02 | 150 / 500 | 0.05 / 0.01 |

The full settings are in `configs/training_settings.json`. Raw GP intervals are not
conformally calibrated.
