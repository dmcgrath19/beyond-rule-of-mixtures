"""Run the Vela, Gao RF and Gao SVR baselines on the grouped-validation folds.

Fold assignments are copied row by row from the PI-GP (analytic mean) prediction
file of the same dataset, grouping and seed, so every baseline is scored on
exactly the folds used for the GP arms and can be compared pairwise.

Vela's mutual-information feature selection is done inside each training fold,
so no baseline sees held-out rows during model selection.

Run:
    uv run python -m utils.run_grouped_baselines --dataset borg --group system --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing

from models.gao import make_gao_rf, make_gao_svr
from models.vela import fit_residual_gp_and_predict, select_vela_features
from run import (
    COMPOSITION_COLS,
    FEATURE_COLS,
    PHYS_COLS,
    _predict_sklearn_uncertainty,
    add_composition_intermediates,
    add_physics_intermediates,
    add_priors,
)
from utils.load_data import (
    BORG_NUMERIC_COLS,
    RADICAL_VELA_FEATURE_COLS,
    RADICAL_VELA_FORCED_FEATURE_COLS,
    VELA_FEATURE_COLS,
    VELA_FORCED_FEATURE_COLS,
    load_borg,
)
from utils.run_grouped_validation import load_cohort

OUT_DIR = Path("results/grouped_validation")
MODELS = ("vela", "gao_rf", "gao_svr")


def _test_type_code(value: object) -> int:
    return {"C": 0, "T": 1}.get(value, 2)


def row_key(df: pd.DataFrame, dataset: str) -> pd.Series:
    if dataset == "hadex":
        return df["sample_id"].astype(str)
    base = df["formula"].astype(str) + "|" + df["HV"].round(3).astype(str)
    return base + "#" + df.groupby(base).cumcount().astype(str)


def load_baseline_frame(dataset: str) -> pd.DataFrame:
    if dataset == "borg":
        df = load_borg()
        df["test_type_code"] = df["Type of test"].map(_test_type_code)
        df = df[np.isclose(df["test_temperature_K"], 298.15)].reset_index(drop=True)
    else:
        df = load_cohort("hadex", "analytic", aggregate_samples=True)
        # Replace the fold-dependent learned prior with the fixed analytical one.
        df = df.drop(columns=[c for c in df.columns if c in {"HV_prior", "fold"} or c.startswith("HV_gpr")])
    df = add_priors(df.drop(columns=["HV_prior"], errors="ignore"))
    df = add_physics_intermediates(df)
    if not set(COMPOSITION_COLS).issubset(df.columns):
        df = add_composition_intermediates(df)
    return df


def attach_folds(df: pd.DataFrame, dataset: str, group: str, seed: int) -> pd.DataFrame:
    ref_path = OUT_DIR / f"{dataset}_analytic_{group}_unique_samples_seed{seed}_predictions.csv"
    ref = pd.read_csv(ref_path).rename(columns={"HV_actual": "HV"})
    ref_folds = pd.Series(ref["fold"].to_numpy(), index=row_key(ref, dataset))
    df = df.copy()
    df["_key"] = row_key(df, dataset)
    df = df[df["_key"].isin(ref_folds.index)].reset_index(drop=True)
    if len(df) != len(ref):
        raise ValueError(f"matched {len(df)} of {len(ref)} reference rows for {ref_path.name}")
    df["fold"] = df["_key"].map(ref_folds).astype(int)
    return df.drop(columns="_key")


def predict_fold(model: str, dataset: str, train: pd.DataFrame, test: pd.DataFrame, seed: int):
    if model == "vela":
        cand, forced = ((VELA_FEATURE_COLS, VELA_FORCED_FEATURE_COLS) if dataset == "borg"
                        else (RADICAL_VELA_FEATURE_COLS, RADICAL_VELA_FORCED_FEATURE_COLS))
        cols = select_vela_features(df=train, target_col="HV", candidate_cols=cand, forced_cols=forced,
                                    max_feature_count=8, corr_threshold=0.85)
        pred = fit_residual_gp_and_predict(train_df=train, pred_df=test, feature_cols=cols,
                                           target_col="HV", prior_col="HV_prior", random_state=seed)
        return pred.mean, pred.std
    input_cols = FEATURE_COLS + PHYS_COLS
    sx = preprocessing.StandardScaler()
    X_tr = sx.fit_transform(train[input_cols].to_numpy())
    X_te = sx.transform(test[input_cols].to_numpy())
    y_tr = train["HV"].to_numpy()
    est = make_gao_rf() if model == "gao_rf" else make_gao_svr()
    est.fit(X_tr, y_tr)
    std = _predict_sklearn_uncertainty(model=est, sklearn_model=model, X_tr=X_tr, y_tr=y_tr, X_te=X_te)
    return est.predict(X_te), std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["borg"], required=True)
    parser.add_argument("--group", choices=["random", "formula", "reference", "system"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = attach_folds(load_baseline_frame(args.dataset), args.dataset, args.group, args.seed)
    for model in MODELS:
        vela_cols = VELA_FEATURE_COLS + VELA_FORCED_FEATURE_COLS if args.dataset == "borg" else RADICAL_VELA_FEATURE_COLS
        needed = (vela_cols if model == "vela" else FEATURE_COLS) + PHYS_COLS + ["HV", "HV_prior"]
        if base[needed].isna().any().any():
            raise ValueError(f"{model}: missing inputs in {base[needed].isna().any(axis=1).sum()} rows")
        df = base.copy()
        pred = np.full(len(df), np.nan)
        std = np.full(len(df), np.nan)
        for f in sorted(df["fold"].unique()):
            tr, te = df["fold"] != f, df["fold"] == f
            mu, sd = predict_fold(model, args.dataset, df[tr].reset_index(drop=True),
                                  df[te].reset_index(drop=True), seed=args.seed * 10 + int(f))
            pred[te.to_numpy()], std[te.to_numpy()] = mu, sd
        keep = [c for c in ["sample_id", "formula", "fold", "HV_prior", *BORG_NUMERIC_COLS, *COMPOSITION_COLS]
                if c in df.columns]
        out = df[keep].copy()
        out["HV_actual"] = df["HV"].to_numpy()
        out["HV_gpr_predicted"] = pred
        # Vela's GP includes a WhiteKernel and the Gao uncertainties are already predictive.
        out["HV_gpr_std"] = std
        out["HV_gpr_pred_std"] = std
        path = args.out_dir / f"{args.dataset}_{model}_{args.group}_unique_samples_seed{args.seed}_predictions.csv"
        out.to_csv(path, index=False)
        mae = float(np.mean(np.abs(out.HV_actual - out.HV_gpr_predicted)))
        print(f"{args.dataset} {args.group} seed{args.seed} {model}: n={len(out)} MAE={mae:.1f}")


if __name__ == "__main__":
    main()
