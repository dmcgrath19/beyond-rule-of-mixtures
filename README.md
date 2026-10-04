# Beyond Rule-of-Mixtures: learned atomic volumes for MPEA hardness prediction

Code for the physics-informed Gaussian-process hardness model with learnable
effective-volume corrections to the Maresca--Curtin solid-solution-strengthening
prior, evaluated on the public Borg benchmark.

## Layout

```
effective_volume_gp.py      the model: analytic, shared-sigma and effective-volume arms,
                            a composition-only control, staged training, grouped CV
curtin_ys_prior.py          Maresca--Curtin prior, elemental lattice and elastic constants
hardness_predictor.py       analytic-mean predictor used in the internal screening workflow
Borg_Datase_PUB.xlsx        public Borg benchmark (n = 99 after filtering)
```

## Running

```bash
python effective_volume_gp.py                        # three arms, five-fold CV grouped by formula
python effective_volume_gp.py --protocol shuffled --seeds 5
python effective_volume_gp.py --variants effective_volume constant_mean
python effective_volume_gp.py --dump-volumes learned_volumes.csv
```

Defaults are the settings used in the paper: Matern-5/2 ARD kernel, one hidden
layer of width 32 with tanh activation in the correction heads, bound beta = 0.02,
Adam with 150 warm-up steps at 0.05 followed by 500 joint steps at 0.01, single
threaded so that runs are bitwise reproducible. `--help` lists every flag.

`effective_volume_gp.py` needs only numpy, pandas, scipy, scikit-learn, torch,
gpytorch, pymatgen and pydantic. `hardness_predictor.py` additionally imports
`atlas.workflows.hea.composition_to_features` from an internal package and is not
required by the model script.
