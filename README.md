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
