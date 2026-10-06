# Beyond Rule-of-Mixtures

Physics-informed GP hardness prediction with learned effective volumes.
Includes public Borg data, saved validation results, MLIP volume surfaces and DFT outputs. HADEX data are not included.

## Install

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root.

## Reproduce saved results

```bash
python -m utils.evaluate_uncertainty
python -m utils.plot_public_results
python -m utils.borg_diagnostics
python simulation/analyze_dft.py
```

Outputs go to `results/`. The saved Borg effective-volume MAEs are 39.324 HV (shuffled), 43.938 HV (formula), 72.547 HV (publication) and 68.399 HV (chemical system).

## Refit Borg models

```bash
python -m utils.run_grouped_validation --dataset borg --model effective --group system --seed 0 --fold-file data/borg/folds_system_seed0.csv --out-dir results/refits
python -m utils.run_grouped_baselines --dataset borg --group system --seed 0 --out-dir results/refits
```

Models: `analytic`, `effective`, `composition`. Groups: `random`, `formula`, `reference`, `system`. Add `--smoke` to the GP command for a short training check.

The grouped cohort contains 93 room-temperature rows; the single-split, calibration and learning-curve analyses use all 99 rows. Archived outputs reproduce the reported tables. Historical model-initialization states were not saved, so fresh refits can differ.

## Prediction and learning curve

```bash
python effective_volume_gp.py --variants analytic effective_volume constant_mean
python hardness_predictor.py --composition NbTaTiZr MoTiTa
python analysis/learning_curve.py --repeats 150 --sizes 25 35 50 70
```

The predictor uses the same descriptor function for training and prediction. The learning-curve runner saves splits to `results/learning_curve/splits.json`; use `--split-in` to reuse them. The published 150-repeat outputs are included separately; their original split indices were not saved.

## Volume calculations

```bash
python analysis/predict_misfit.py --comp "Nb49.1 Ti16.1 W34.6" -o results/misfit.csv
python analysis/experimental_14_pmv.py
python analysis/experimental_14_anchors.py --seeds 3 -o results/experimental_14_anchors_refit.csv
```

`data/simulation/` contains four cached surfaces. Analysis scripts use the published 14-alloy validation values, not HADEX training records.

## Simulation inputs

```bash
python simulation/generate_eos.py --composition "Nb50 Ti25 W25" --structures-only
python simulation/generate_eos.py --composition "Nb50 Ti25 W25" --device cuda
python simulation/write_dft_inputs.py --composition "Mo40 Ti40 Ta20" --mlip-volume-per-atom 16.5253525646
```

Optional dependencies are listed in `requirements-simulation.txt`. Fresh EGIP calculations require the compatible FairChem fork used by the original environment; that workflow has not been validated with a public replacement. DFT inputs require a separate VASP installation and PBE_64 POTCARs.

The EGIP checkpoint is not distributed with this repository; it is available from the authors on request, and its checksum is recorded in `configs/checkpoint.json` so a copy can be verified. All volume analyses in the paper run from the cached surfaces in `data/simulation/`, so the checkpoint is needed only to generate new surfaces.

## Training settings

All analyses use the same model: composition on the element basis through one hidden tanh layer with a zero-initialised output, a bounded-exponential correction, and two-stage Adam (kernel-only, then joint) with the physical prefactors re-optimised in every training fold. The bound β is selected by out-of-fold MAE under formula-grouped cross-validation. Hyperparameters by analysis:

| Analysis | Kernel (ARD) | Width | β | Steps (kernel-only / joint) | Learning rates |
|---|---|---|---|---|---|
| Grouped validation, repeated-protocol calibration and conformal, held-out alloys, stability, mass balance | RBF | 4 | 0.05 | 200 / 800 | 0.05 / 0.005 (head 0.001, weight decay 0.01, frozen for the first 600 joint steps) |
| Single-split controls, calibration and conformal tables, learning curves, prefactors, volume-anchor comparison | Matérn-5/2 | 32 | 0.02 | 150 / 500 | 0.05 / 0.01 |

The full settings and artifact provenance are in `configs/training_settings.json` and `configs/artifact_sources.json`. Raw GP intervals are not conformally calibrated. The ignored `rebuttal/` directory is local working material, outside the release package.
