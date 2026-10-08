"""Borg grouped-validation engine, extracted from the result-generating pipeline.

No database access or proprietary datasets are included. See configs/training_settings.json.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

from models import (
    ModelConfig,
    HardnessGP,
    compute_yield_strength,
    curtin_intermediates,
    composition_supported,
)
from models.sigma_models import SigmaModelSpec
from utils.load_data import (
    FEATURE_COLS,
    load_borg,
)
from lib.features import MICROSTRUCTURE_COLS
from lib.physics import SUPPORTED_ELEMENTS, composition_fraction_array

SEED = 42
N_FOLDS = 5
RESULTS_DIR = Path("results")
SUMMARY_FILE = RESULTS_DIR / "summary.csv"
YS_SUMMARY_FILE = RESULTS_DIR / "ys_summary.csv"
PHYS_COLS = ["phys_G", "phys_nu", "phys_b", "phys_sigma", "phys_T"]
COMPOSITION_COLS = [f"comp_{element}" for element in SUPPORTED_ELEMENTS]
MODEL_CHOICES = [
    "curtin",
    "gao_rf",
    "gao_svr",
    "gpytorch",
    "gpytorch_nonlinear_sigma",
    "gpytorch_dual",
    "gpytorch_nonlinear_sigma_dual",
    "xrd_peak_count",
    "microstructure",
    "xrd_and_microstructure",
]
EXPERIMENTAL_MODEL_CHOICES = [
    "gpytorch_nonlinear_sigma_dual_diagnostic",
    "gpytorch_nonlinear_sigma_dual_gated",
]
YS_MODEL_CHOICES = ["vela_ys"]
STAGE1_ITERS = 200
STAGE2_ITERS = 800
VELA_REPEATS_DEFAULT = 200



YS_COLS = ['YS_0K_MPa', 'YS_at_T_MPa', 'activation_energy_eV', 'C11_rom_GPa', 'C12_rom_GPa', 'C44_rom_GPa', 'bulk_modulus_rom_GPa', 'shear_modulus_rom_GPa', 'poisson']
_NAN_YS_ROW = {col: np.nan for col in YS_COLS}
SVR_UNCERTAINTY_ENSEMBLE_SIZE = 32

def _load_dataset(dataset, config, exclude_single_k=False):
    if dataset != "borg":
        raise ValueError("Only the public Borg dataset is available")
    return load_borg()

@dataclass(frozen=True)
class GpRunResult:
    results: pd.DataFrame
    prior_mae_hv: float
    prior_rmse_hv: float
    gpr_mae_hv: float
    gpr_rmse_hv: float

@dataclass(frozen=True)
class FoldSigmaDiagnostics:
    sigma: np.ndarray
    effective_volumes: np.ndarray | None = None
    delta_volumes: np.ndarray | None = None
    contributions: np.ndarray | None = None
    edge_yield_strength_mpa: np.ndarray | None = None
    screw_yield_strength_mpa: np.ndarray | None = None
    edge_probability: np.ndarray | None = None
    edge_blend_weight: np.ndarray | None = None
    mechanism_gate_strength: np.ndarray | None = None
    mechanism_delta: np.ndarray | None = None
    mechanism_delta_threshold: np.ndarray | None = None
    mechanism_delta_sharpness: np.ndarray | None = None
    controlling_is_screw: np.ndarray | None = None
    predictive_std: np.ndarray | None = None

def _append_summary(row: pd.DataFrame) -> None:
    if SUMMARY_FILE.exists():
        existing = pd.read_csv(SUMMARY_FILE)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(SUMMARY_FILE, index=False)
    print(f"Updated {SUMMARY_FILE}")

def _dataset_label(dataset: str, save_suffix: str) -> str:
    if save_suffix == "":
        return dataset
    return f"{dataset}_{save_suffix}"

def _sigma_model_spec(config: ModelConfig) -> SigmaModelSpec:
    return SigmaModelSpec(
        variant=config.sigma_variant,
        hidden_dim=config.sigma_hidden_dim,
        log_bound=config.sigma_log_bound,
    )

def add_priors(df: pd.DataFrame) -> pd.DataFrame:
    hv_prior = np.full(len(df), np.nan)
    ys_rows: list[dict[str, float]] = []
    for i, (formula, temp) in enumerate(zip(df["formula"], df["test_temperature_K"])):
        comp = Composition(formula)
        if not composition_supported(comp):
            ys_rows.append(_NAN_YS_ROW)
            continue
        ys_res = compute_yield_strength(comp, temp)
        hv_prior[i] = ys_res.yield_strength_at_temperature * 3.0 / 9.81
        ys_rows.append({
            "YS_0K_MPa": ys_res.yield_strength_at_zero_kelvin,
            "YS_at_T_MPa": ys_res.yield_strength_at_temperature,
            "activation_energy_eV": ys_res.activation_energy,
            "C11_rom_GPa": ys_res.C11_rom,
            "C12_rom_GPa": ys_res.C12_rom,
            "C44_rom_GPa": ys_res.C44_rom,
            "bulk_modulus_rom_GPa": ys_res.bulk_modulus_rom,
            "shear_modulus_rom_GPa": ys_res.shear_modulus_rom,
            "poisson": ys_res.poisson,
        })
    df = df.copy()
    df["HV_prior"] = hv_prior
    ys_df = pd.DataFrame(ys_rows, index=df.index)
    return pd.concat([df, ys_df], axis=1)

def add_physics_intermediates(df: pd.DataFrame) -> pd.DataFrame:
    nan_row = (np.nan,) * len(PHYS_COLS)
    phys_rows = []
    for formula, temp in zip(df["formula"], df["test_temperature_K"]):
        if composition_supported(Composition(formula)):
            phys_rows.append(curtin_intermediates(formula, temp))
        else:
            phys_rows.append(nan_row)
    df = df.copy()
    df[PHYS_COLS] = pd.DataFrame(phys_rows, index=df.index)
    return df

def add_composition_intermediates(df: pd.DataFrame) -> pd.DataFrame:
    composition_rows = []
    for formula in df["formula"]:
        composition = Composition(formula)
        if composition_supported(composition):
            composition_rows.append(composition_fraction_array(composition))
        else:
            composition_rows.append(np.full(len(COMPOSITION_COLS), np.nan))
    df = df.copy()
    df[COMPOSITION_COLS] = pd.DataFrame(composition_rows, columns=COMPOSITION_COLS, index=df.index)
    return df

def _predict_residual_uncertainty(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    pred_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    n_neighbors: int = 10,
) -> np.ndarray:
    residual_sq = np.square(y_tr - pred_tr)
    if len(X_tr) == 0:
        return np.zeros(len(X_te), dtype=float)
    scaler = preprocessing.StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    neighbor_count = max(1, min(n_neighbors, len(X_tr_scaled)))
    model = KNeighborsRegressor(n_neighbors=neighbor_count, weights="distance")
    model.fit(X_tr_scaled, residual_sq)
    pred_sq = np.clip(model.predict(X_te_scaled), a_min=1e-6, a_max=None)
    return np.sqrt(pred_sq)

def _predict_rf_uncertainty(model, X_te: np.ndarray) -> np.ndarray:
    tree_predictions = np.stack([estimator.predict(X_te) for estimator in model.estimators_], axis=0)
    return np.std(tree_predictions, axis=0, ddof=0)

def _predict_svr_bootstrap_uncertainty(
    model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
) -> np.ndarray:
    ensemble_predictions = []
    for bootstrap_id in range(SVR_UNCERTAINTY_ENSEMBLE_SIZE):
        sample_ix = np.random.default_rng(SEED + bootstrap_id).choice(len(X_tr), size=len(X_tr), replace=True)
        bootstrap_model = clone(model)
        bootstrap_model.fit(X_tr[sample_ix], y_tr[sample_ix])
        ensemble_predictions.append(bootstrap_model.predict(X_te))
    return np.std(np.stack(ensemble_predictions, axis=0), axis=0, ddof=0)

def _predict_sklearn_uncertainty(
    model,
    sklearn_model: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
) -> np.ndarray:
    if sklearn_model == "gao_rf":
        return _predict_rf_uncertainty(model, X_te)
    if sklearn_model == "gao_svr":
        return _predict_svr_bootstrap_uncertainty(model, X_tr, y_tr, X_te)
    raise ValueError(f"Unsupported sklearn model for uncertainty: {sklearn_model}")

def _feature_cols_for(config: ModelConfig) -> list[str]:
    cols = list(FEATURE_COLS)
    if config.include_xrd_peak_count:
        cols.append("xrd_peak_count")
    if config.include_microstructure:
        cols.extend(MICROSTRUCTURE_COLS)
    return cols

def _train_gp_fold(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    phys_tr: np.ndarray,
    phys_te: np.ndarray,
    composition_tr: np.ndarray,
    composition_te: np.ndarray,
    y_tr: np.ndarray,
    fold_label: str,
    config: ModelConfig,
    *,
    capture_fit=None,
    volume_anchors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FoldSigmaDiagnostics]:
    """Train GP on one fold, return predictions, uncertainties, prior, and sigma diagnostics."""
    import time
    import torch
    import gpytorch as _gpytorch

    def _unique_parameters(*parameter_groups: torch.nn.ParameterList | list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
        seen: set[int] = set()
        unique: list[torch.nn.Parameter] = []
        for group in parameter_groups:
            for parameter in group:
                parameter_id = id(parameter)
                if parameter_id not in seen:
                    seen.add(parameter_id)
                    unique.append(parameter)
        return unique

    sx = preprocessing.StandardScaler()
    X_tr_scaled = sx.fit_transform(X_tr)
    X_te_scaled = sx.transform(X_te)
    n_feat = X_tr.shape[1]
    n_phys = phys_tr.shape[1]
    n_composition = composition_tr.shape[1]

    X_tr_aug = np.hstack([X_tr_scaled, phys_tr, composition_tr])
    X_te_aug = np.hstack([X_te_scaled, phys_te, composition_te])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_mean_val, y_std_val = float(np.mean(y_tr)), float(np.std(y_tr))

    train_x = torch.tensor(X_tr_aug, dtype=torch.float64, device=device)
    train_y = torch.tensor((y_tr - y_mean_val) / y_std_val, dtype=torch.float64, device=device)
    test_x = torch.tensor(X_te_aug, dtype=torch.float64, device=device)
    y_mean_t = torch.tensor(y_mean_val, dtype=torch.float64, device=device)
    y_std_t = torch.tensor(y_std_val, dtype=torch.float64, device=device)

    likelihood = _gpytorch.likelihoods.GaussianLikelihood()
    model = HardnessGP(
        train_x, train_y, likelihood,
        n_features=n_feat, physics_start=n_feat,
        composition_start=n_feat + n_phys,
        n_composition_dims=n_composition,
        y_mean=y_mean_t, y_std=y_std_t,
        sigma_model_spec=_sigma_model_spec(config),
        use_dual_mechanism=config.use_dual_mechanism,
        mechanism_mode=config.mechanism_mode,
        mechanism_delta_threshold=config.mechanism_delta_threshold,
        mechanism_probability_mode=config.mechanism_probability_mode,
        mean_type=config.mean_type,
        kernel=config.kernel,
    ).double().to(device)
    likelihood = likelihood.double().to(device)

    if volume_anchors is not None:
        sigma_model = model.mean_module.sigma_model
        if not hasattr(sigma_model, "base_volumes"):
            raise ValueError("Volume anchors require a volume correction model")
        anchors = torch.as_tensor(volume_anchors, dtype=torch.float64, device=device)
        if anchors.shape != sigma_model.base_volumes.shape or not torch.isfinite(anchors).all() or not torch.all(anchors > 0):
            raise ValueError("Volume anchors must be finite, positive and match the element basis")
        sigma_model.base_volumes.copy_(anchors)

    mll = _gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    t0 = time.perf_counter()
    sigma_parameters = list(model.mean_module.sigma_model.parameters())
    sigma_parameter_ids = {id(parameter) for parameter in sigma_parameters}
    non_sigma_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in sigma_parameter_ids
    ]

    model.mean_module.requires_grad_(False)
    model.train()
    likelihood.train()
    opt1 = torch.optim.Adam(
        _unique_parameters(
            [p for p in model.parameters() if p.requires_grad],
            list(likelihood.parameters()),
        ),
        lr=config.stage1_lr,
    )
    for step in range(STAGE1_ITERS):
        opt1.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        opt1.step()
        if step == 0 or (step + 1) % 50 == 0:
            print(f"  [{fold_label}] S1 {step + 1:3d}/{STAGE1_ITERS}  loss {loss.item():.3f}")

    model.mean_module.requires_grad_(True)
    sigma_warmup_iters = min(config.sigma_warmup_iters, STAGE2_ITERS)
    if sigma_warmup_iters > 0 and len(sigma_parameters) > 0:
        model.mean_module.sigma_model.requires_grad_(False)
        opt2_warmup = torch.optim.Adam(
            _unique_parameters(non_sigma_parameters, list(likelihood.parameters())),
            lr=config.stage2_lr,
        )
        for step in range(sigma_warmup_iters):
            opt2_warmup.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            opt2_warmup.step()
            if step == 0 or (step + 1) % 50 == 0:
                print(f"  [{fold_label}] S2a {step + 1:3d}/{sigma_warmup_iters}  loss {loss.item():.3f}")
        model.mean_module.sigma_model.requires_grad_(True)

    sigma_finetune_iters = STAGE2_ITERS - sigma_warmup_iters
    if sigma_finetune_iters > 0:
        parameter_groups = [
            {
                "params": _unique_parameters(non_sigma_parameters, list(likelihood.parameters())),
                "lr": config.stage2_lr,
            }
        ]
        if len(sigma_parameters) > 0:
            parameter_groups.append({
                "params": _unique_parameters(sigma_parameters),
                "lr": config.sigma_stage2_lr,
                "weight_decay": config.sigma_weight_decay,
            })
        opt2 = torch.optim.Adam(parameter_groups)
        for step in range(sigma_finetune_iters):
            opt2.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            opt2.step()
            if step == 0 or (step + 1) % 50 == 0:
                print(f"  [{fold_label}] S2b {step + 1:3d}/{sigma_finetune_iters}  loss {loss.item():.3f}")

    elapsed = time.perf_counter() - t0
    print(f"  [{fold_label}] done in {elapsed:.1f}s")

    model.eval()
    likelihood.eval()
    if capture_fit is not None:
        capture_fit(model, likelihood, sx, y_mean_val, y_std_val)
    with torch.no_grad():
        post = model(test_x)
        pred = post.mean.cpu().numpy() * y_std_val + y_mean_val
        std = post.stddev.cpu().numpy() * y_std_val
        predictive_std = likelihood(post).stddev.cpu().numpy() * y_std_val
        prior = model.mean_module(test_x).cpu().numpy()
        sigma_diagnostics = model.mean_module.sigma_diagnostics(test_x)
        mechanism_diagnostics = model.mean_module.mechanism_diagnostics(test_x)
        effective_volumes = None
        if sigma_diagnostics.effective_volumes is not None:
            effective_volumes = sigma_diagnostics.effective_volumes.cpu().numpy()
        delta_volumes = None
        if sigma_diagnostics.delta_volumes is not None:
            delta_volumes = sigma_diagnostics.delta_volumes.cpu().numpy()
        contributions = None
        if sigma_diagnostics.contributions is not None:
            contributions = sigma_diagnostics.contributions.cpu().numpy()
        screw_yield_strength_mpa = None
        if mechanism_diagnostics.yield_strength_screw is not None:
            screw_yield_strength_mpa = mechanism_diagnostics.yield_strength_screw.cpu().numpy()
        edge_probability = None
        if mechanism_diagnostics.edge_probability is not None:
            edge_probability = mechanism_diagnostics.edge_probability.cpu().numpy()
        edge_blend_weight = None
        if mechanism_diagnostics.edge_blend_weight is not None:
            edge_blend_weight = mechanism_diagnostics.edge_blend_weight.cpu().numpy()
        mechanism_gate_strength = None
        if mechanism_diagnostics.mechanism_gate_strength is not None:
            mechanism_gate_strength = mechanism_diagnostics.mechanism_gate_strength.cpu().numpy()
        mechanism_delta = None
        if mechanism_diagnostics.mechanism_delta is not None:
            mechanism_delta = mechanism_diagnostics.mechanism_delta.cpu().numpy()
        mechanism_delta_threshold = None
        if mechanism_diagnostics.mechanism_delta_threshold is not None:
            mechanism_delta_threshold = mechanism_diagnostics.mechanism_delta_threshold.cpu().numpy()
        mechanism_delta_sharpness = None
        if mechanism_diagnostics.mechanism_delta_sharpness is not None:
            mechanism_delta_sharpness = mechanism_diagnostics.mechanism_delta_sharpness.cpu().numpy()
        controlling_is_screw = None
        if mechanism_diagnostics.controlling_is_screw is not None:
            controlling_is_screw = mechanism_diagnostics.controlling_is_screw.cpu().numpy()

    return pred, std, prior, FoldSigmaDiagnostics(
        sigma=sigma_diagnostics.sigma.cpu().numpy(),
        effective_volumes=effective_volumes,
        delta_volumes=delta_volumes,
        contributions=contributions,
        edge_yield_strength_mpa=mechanism_diagnostics.yield_strength_edge.cpu().numpy(),
        screw_yield_strength_mpa=screw_yield_strength_mpa,
        edge_probability=edge_probability,
        edge_blend_weight=edge_blend_weight,
        mechanism_gate_strength=mechanism_gate_strength,
        mechanism_delta=mechanism_delta,
        mechanism_delta_threshold=mechanism_delta_threshold,
        mechanism_delta_sharpness=mechanism_delta_sharpness,
        controlling_is_screw=controlling_is_screw,
        predictive_std=predictive_std,
    )

def run_gpytorch_gpr(
    config: ModelConfig,
    dataset: str,
    dataset_label: str,
    exclude_single_k: bool = False,
    preloaded: pd.DataFrame | None = None,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    write_outputs: bool = True,
    write_summary: bool = True,
) -> GpRunResult:
    if preloaded is not None:
        df = preloaded
    else:
        df = _load_dataset(dataset, config, exclude_single_k=exclude_single_k)
        print("Computing priors + physics intermediates...")
        df = add_priors(df)
        df = add_physics_intermediates(df)
    if not set(COMPOSITION_COLS).issubset(df.columns):
        df = add_composition_intermediates(df)

    feature_cols = _feature_cols_for(config)
    required = feature_cols + PHYS_COLS + COMPOSITION_COLS + ["HV", "HV_prior"]
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"Valid samples: {len(df)}")

    if folds is None:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = list(kf.split(np.arange(len(df))))

    kernel_cols = feature_cols + ["phys_T"]
    n = len(df)
    oof_pred = np.full(n, np.nan)
    oof_std = np.full(n, np.nan)
    oof_pred_std = np.full(n, np.nan)
    oof_prior = np.full(n, np.nan)
    oof_sigma = np.full(n, np.nan)
    oof_edge_yield_strength_mpa = np.full(n, np.nan)
    oof_screw_yield_strength_mpa = np.full(n, np.nan)
    oof_edge_probability = np.full(n, np.nan)
    oof_edge_blend_weight = np.full(n, np.nan)
    oof_mechanism_gate_strength = np.full(n, np.nan)
    oof_mechanism_delta = np.full(n, np.nan)
    oof_mechanism_delta_threshold = np.full(n, np.nan)
    oof_mechanism_delta_sharpness = np.full(n, np.nan)
    oof_controlling_is_screw = np.full(n, None, dtype=object)
    oof_effective_volumes = np.full((n, len(SUPPORTED_ELEMENTS)), np.nan)
    oof_delta_volumes = np.full((n, len(SUPPORTED_ELEMENTS)), np.nan)
    oof_sigma_contributions = np.full((n, len(SUPPORTED_ELEMENTS)), np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    fold_maes: list[float] = []
    fold_rmses: list[float] = []

    print(f"Running {len(folds)}-fold cross-validation\n")

    for fold_i, (ix_tr, ix_te) in enumerate(folds):
        label = f"fold {fold_i + 1}/{len(folds)}"
        print(f"--- {label}: train={len(ix_tr)}, test={len(ix_te)} ---")

        pred_te, std_te, prior_te, sigma_diagnostics_te = _train_gp_fold(
            X_tr=df.loc[ix_tr, kernel_cols].values,
            X_te=df.loc[ix_te, kernel_cols].values,
            phys_tr=df.loc[ix_tr, PHYS_COLS].values,
            phys_te=df.loc[ix_te, PHYS_COLS].values,
            composition_tr=df.loc[ix_tr, COMPOSITION_COLS].values,
            composition_te=df.loc[ix_te, COMPOSITION_COLS].values,
            y_tr=df.loc[ix_tr, "HV"].values,
            fold_label=label,
            config=config,
        )

        oof_pred[ix_te] = pred_te
        oof_std[ix_te] = std_te
        oof_pred_std[ix_te] = sigma_diagnostics_te.predictive_std
        oof_prior[ix_te] = prior_te
        oof_sigma[ix_te] = sigma_diagnostics_te.sigma
        if sigma_diagnostics_te.edge_yield_strength_mpa is not None:
            oof_edge_yield_strength_mpa[ix_te] = sigma_diagnostics_te.edge_yield_strength_mpa
        if sigma_diagnostics_te.screw_yield_strength_mpa is not None:
            oof_screw_yield_strength_mpa[ix_te] = sigma_diagnostics_te.screw_yield_strength_mpa
        if sigma_diagnostics_te.edge_probability is not None:
            oof_edge_probability[ix_te] = sigma_diagnostics_te.edge_probability
        if sigma_diagnostics_te.edge_blend_weight is not None:
            oof_edge_blend_weight[ix_te] = sigma_diagnostics_te.edge_blend_weight
        if sigma_diagnostics_te.mechanism_gate_strength is not None:
            oof_mechanism_gate_strength[ix_te] = sigma_diagnostics_te.mechanism_gate_strength
        if sigma_diagnostics_te.mechanism_delta is not None:
            oof_mechanism_delta[ix_te] = sigma_diagnostics_te.mechanism_delta
        if sigma_diagnostics_te.mechanism_delta_threshold is not None:
            oof_mechanism_delta_threshold[ix_te] = sigma_diagnostics_te.mechanism_delta_threshold
        if sigma_diagnostics_te.mechanism_delta_sharpness is not None:
            oof_mechanism_delta_sharpness[ix_te] = sigma_diagnostics_te.mechanism_delta_sharpness
        if sigma_diagnostics_te.controlling_is_screw is not None:
            oof_controlling_is_screw[ix_te] = sigma_diagnostics_te.controlling_is_screw
        if sigma_diagnostics_te.effective_volumes is not None:
            oof_effective_volumes[ix_te, :] = sigma_diagnostics_te.effective_volumes
        if sigma_diagnostics_te.delta_volumes is not None:
            oof_delta_volumes[ix_te, :] = sigma_diagnostics_te.delta_volumes
        if sigma_diagnostics_te.contributions is not None:
            oof_sigma_contributions[ix_te, :] = sigma_diagnostics_te.contributions
        fold_ids[ix_te] = fold_i

        y_te = df.loc[ix_te, "HV"].values
        fold_mae = float(np.mean(np.abs(y_te - pred_te)))
        fold_rmse = float(np.sqrt(np.mean((y_te - pred_te) ** 2)))
        fold_maes.append(fold_mae)
        fold_rmses.append(fold_rmse)
        print(f"  [{label}] MAE={fold_mae:.1f}  RMSE={fold_rmse:.1f}\n")

    y_all = df["HV"].values
    pr_all = df["HV_prior"].values
    if config.sigma_variant != "analytic":
        pr_all = oof_prior
    mae_oof = float(np.mean(np.abs(y_all - oof_pred)))
    rmse_oof = float(np.sqrt(np.mean((y_all - oof_pred) ** 2)))
    mae_prior = float(np.mean(np.abs(y_all - pr_all)))
    rmse_prior = float(np.sqrt(np.mean((y_all - pr_all) ** 2)))

    print(f"OOF MAE={mae_oof:.1f} \u00b1 {np.std(fold_maes):.1f} HV  "
          f"RMSE={rmse_oof:.1f} \u00b1 {np.std(fold_rmses):.1f} HV")
    print(f"Prior MAE={mae_prior:.1f} HV")

    df = df.copy()
    df["fold"] = fold_ids
    df["sigma_rom"] = df["phys_sigma"]
    df["sigma_used"] = oof_sigma
    if config.sigma_variant != "analytic":
        df["HV_prior_rom"] = df["HV_prior"]
        df["HV_prior"] = oof_prior
        if config.sigma_variant == "effective_volume":
            df["sigma_effective"] = oof_sigma
            df["sigma_effective_ratio"] = oof_sigma / df["sigma_rom"].to_numpy()
            for element_index, element in enumerate(SUPPORTED_ELEMENTS):
                df[f"effective_volume_{element}"] = oof_effective_volumes[:, element_index]
                df[f"delta_volume_{element}"] = oof_delta_volumes[:, element_index]
                df[f"sigma_contribution_{element}"] = oof_sigma_contributions[:, element_index]
    if config.use_dual_mechanism:
        df["edge_yield_strength_mpa"] = oof_edge_yield_strength_mpa
        df["screw_yield_strength_mpa"] = oof_screw_yield_strength_mpa
        df["edge_mechanism_probability"] = oof_edge_probability
        if not np.isnan(oof_edge_blend_weight).all():
            df["edge_blend_weight"] = oof_edge_blend_weight
        if not np.isnan(oof_mechanism_gate_strength).all():
            df["mechanism_gate_strength"] = oof_mechanism_gate_strength
        df["mechanism_delta"] = oof_mechanism_delta
        if not np.isnan(oof_mechanism_delta_threshold).all():
            df["mechanism_delta_threshold"] = oof_mechanism_delta_threshold
        if not np.isnan(oof_mechanism_delta_sharpness).all():
            df["mechanism_delta_sharpness"] = oof_mechanism_delta_sharpness
        df["controlling_is_screw"] = oof_controlling_is_screw
    df["HV_gpr_predicted"] = oof_pred
    df["HV_gpr_std"] = oof_std
    df["HV_gpr_pred_std"] = oof_pred_std

    result_df = df.rename(columns={"HV": "HV_actual"})
    if write_outputs:
        outfile = RESULTS_DIR / f"{config.name}_results_{dataset_label}.csv"
        result_df.to_csv(outfile, index=False)
        print(f"\nSaved {outfile} ({len(df)} rows)")

    if write_summary:
        _append_summary(pd.DataFrame([{
            "model": config.name, "dataset": dataset_label, "mode": f"{len(folds)}fold_cv",
            "n_samples": len(df), "n_valid": len(df),
            "prior_mae_hv": mae_prior, "prior_rmse_hv": rmse_prior,
            "gpr_mae_hv": mae_oof, "gpr_rmse_hv": rmse_oof, "seed": SEED,
        }]))

    return GpRunResult(
        results=result_df,
        prior_mae_hv=mae_prior,
        prior_rmse_hv=rmse_prior,
        gpr_mae_hv=mae_oof,
        gpr_rmse_hv=rmse_oof,
    )
