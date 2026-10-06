"""Run leakage-resistant validation protocols from retained result cohorts.

This utility deliberately consumes the retained CSV cohorts so a validation
rerun does not depend on live MongoDB access.  It never uses their old fold or
prediction columns; folds and predictions are rebuilt from scratch.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core.composition import Composition

from lib.features import FEATURE_NAMES, _compute_features_from_fracs
from models import get_model_config
from run import (
    COMPOSITION_COLS,
    PHYS_COLS,
    _feature_cols_for,
    run_gpytorch_gpr,
)


OUT_DIR = Path("results/grouped_validation")


def chemical_system(df: pd.DataFrame) -> pd.Series:
    return df[COMPOSITION_COLS].apply(
        lambda row: "-".join(
            sorted(col.removeprefix("comp_") for col, value in row.items() if float(value) > 1e-10)
        ),
        axis=1,
    )


def balanced_group_folds(groups: pd.Series, n_splits: int = 5, seed: int = 42):
    """Assign whole groups to approximately size-balanced folds."""
    groups = groups.astype(str).reset_index(drop=True)
    counts = groups.value_counts()
    rng = np.random.default_rng(seed)
    tie_break = {group: rank for rank, group in enumerate(rng.permutation(counts.index.to_numpy()))}
    ordered = sorted(counts.index, key=lambda group: (-int(counts[group]), tie_break[group]))
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for group in ordered:
        fold_id = int(np.argmin(fold_sizes))
        fold_groups[fold_id].append(group)
        fold_sizes[fold_id] += int(counts[group])

    all_indices = np.arange(len(groups))
    folds = []
    for test_groups in fold_groups:
        test_mask = groups.isin(test_groups).to_numpy()
        folds.append((all_indices[~test_mask], all_indices[test_mask]))
    return folds


def load_cohort(dataset: str, model_name: str, aggregate_samples: bool = False) -> pd.DataFrame:
    if dataset != "borg":
        raise ValueError("Only public Borg data are included")
    path = Path(__file__).resolve().parents[1] / "data/borg/cohort.csv"
    return pd.read_csv(path).rename(columns={"HV_actual": "HV"})


def bulk_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the core/shell reconstruction by the measured bulk composition (core = shell = bulk)."""
    df = df.copy()
    for i, formula in enumerate(df["formula"]):
        comp = Composition(formula)
        elements = [el.name for el in comp.elements]
        feats = _compute_features_from_fracs(elements, [comp.get_atomic_fraction(el) for el in elements])
        for name, val in zip(FEATURE_NAMES, feats if feats is not None else [np.nan] * len(FEATURE_NAMES)):
            df.loc[i, f"core_{name}"] = val
            df.loc[i, f"shell_{name}"] = val
    return df


def model_config(model_name: str):
    if model_name == "analytic":
        return get_model_config("gpytorch")
    if model_name == "composition":
        return replace(get_model_config("gpytorch"), name="composition_only_gp", mean_type="constant")
    if model_name == "composition_full":
        return replace(
            get_model_config("xrd_and_microstructure"),
            name="composition_only_xrd_and_microstructure_gp",
            mean_type="constant",
        )
    if model_name == "effective":
        return get_model_config("gpytorch_nonlinear_sigma")
    if model_name == "analytic_full":
        return get_model_config("xrd_and_microstructure")
    if model_name == "effective_full":
        return replace(
            get_model_config("gpytorch_nonlinear_sigma"),
            name="effective_volume_xrd_and_microstructure_grouped",
            include_xrd_peak_count=True,
            include_microstructure=True,
        )
    raise ValueError(model_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["borg"], required=True)
    parser.add_argument(
        "--model",
        choices=["analytic", "effective", "composition", "analytic_full", "effective_full", "composition_full"],
        required=True,
    )
    parser.add_argument("--group", choices=["random", "formula", "reference", "system"], required=True)
    parser.add_argument("--keep-repeated-samples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-filter",
        type=Path,
        help="CSV with a sample_id column; restrict HADEX to those specimens (label suffix = file stem)",
    )
    parser.add_argument("--bulk-microstructure", action="store_true",
                        help="replace the core/shell reconstruction by the bulk composition (cohort suffix _bulkmicro)")
    parser.add_argument("--exclude-imputed-temperature", action="store_true",
                        help="Borg: drop rows whose test temperature was imputed (cohort suffix _measuredtemp)")
    parser.add_argument("--init-seed", type=int,
                        help="seed torch before fitting, for initialisation-stability runs (label suffix _init<N>)")
    parser.add_argument("--log-bound", type=float,
                        help="override the correction bound beta of the effective-volume head (label suffix _beta<value>)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fold-file", type=Path, help="CSV with borg_row_id and fold; reuse archived folds")
    parser.add_argument("--smoke", action="store_true", help="2 steps per stage, first fold only; not paper results")
    args = parser.parse_args()
    import run as engine
    torch.set_num_threads(1)
    if args.smoke:
        engine.STAGE1_ITERS = 2
        engine.STAGE2_ITERS = 2
    # Historical runs did not record initialisation RNG state. New runs use a fixed seed.
    torch.manual_seed(args.init_seed if args.init_seed is not None else args.seed)
    np.random.seed(args.seed)
    if args.init_seed is not None:
        torch.manual_seed(args.init_seed)
        np.random.seed(args.init_seed)

    aggregate = args.dataset == "hadex" and not args.keep_repeated_samples
    df = load_cohort(args.dataset, args.model, aggregate_samples=aggregate)
    if args.bulk_microstructure:
        df = bulk_microstructure(df)
    if args.exclude_imputed_temperature:
        df = df[df["Test temperature"].notna()].reset_index(drop=True)
    config = model_config(args.model)
    if args.log_bound is not None:
        config = replace(config, sigma_log_bound=args.log_bound)
    required = _feature_cols_for(config) + PHYS_COLS + COMPOSITION_COLS + ["HV", "HV_prior"]
    df = df.dropna(subset=required).reset_index(drop=True)
    if args.sample_filter is not None:
        keep = set(pd.read_csv(args.sample_filter)["sample_id"].astype(str))
        df = df[df["sample_id"].astype(str).isin(keep)].reset_index(drop=True)
    df["chemical_system"] = chemical_system(df)

    if args.group == "random":
        # Shuffled-row protocol of the original submission, kept as a reference point.
        df["row_id"] = np.arange(len(df)).astype(str)
    group_col = {
        "random": "row_id",
        "formula": "formula",
        "reference": "reference_id",
        "sample": "sample_id",
        "system": "chemical_system",
    }[args.group]
    if group_col not in df or df[group_col].isna().any():
        raise ValueError(f"Grouping column {group_col!r} is unavailable or incomplete")
    if args.fold_file:
        saved = pd.read_csv(args.fold_file).set_index("borg_row_id")
        ids = df.borg_row_id
        if set(ids) != set(saved.index) or not saved.index.is_unique:
            raise ValueError("Fold manifest must contain each cohort row exactly once")
        fids = saved.loc[ids, "fold"].to_numpy()
        indices = np.arange(len(df))
        folds = [(indices[fids != f], indices[fids == f]) for f in sorted(set(fids))]
        if args.group != "random":
            check = pd.DataFrame({"group": df[group_col], "fold": fids})
            if check.groupby("group").fold.nunique().max() != 1:
                raise ValueError("Saved folds split a grouping unit")
    else:
        folds = balanced_group_folds(df[group_col], seed=args.seed)
    if args.smoke:
        # Restrict the smoke cohort to one valid fold so every output row has a prediction.
        tr, te = folds[0]
        pred, sd, prior, diag = engine._train_gp_fold(
            df.iloc[tr][engine._feature_cols_for(config) + ["phys_T"]].to_numpy(),
            df.iloc[te][engine._feature_cols_for(config) + ["phys_T"]].to_numpy(),
            df.iloc[tr][PHYS_COLS].to_numpy(), df.iloc[te][PHYS_COLS].to_numpy(),
            df.iloc[tr][COMPOSITION_COLS].to_numpy(), df.iloc[te][COMPOSITION_COLS].to_numpy(),
            df.iloc[tr].HV.to_numpy(), "smoke", config)
        assert np.isfinite(pred).all() and np.isfinite(sd).all()
        print(f"Smoke check passed: {len(pred)} finite predictions")
        return

    cohort = "rows" if args.keep_repeated_samples else "unique_samples"
    if args.sample_filter is not None:
        cohort += f"_{args.sample_filter.stem}"
    if args.bulk_microstructure:
        cohort += "_bulkmicro"
    if args.exclude_imputed_temperature:
        cohort += "_measuredtemp"
    label = f"{args.dataset}_{args.model}_{args.group}_{cohort}_seed{args.seed}"
    if args.init_seed is not None:
        label += f"_init{args.init_seed}"
    if args.log_bound is not None:
        label += f"_beta{args.log_bound:g}"
    print(
        f"COHORT dataset={args.dataset} model={args.model} rows={len(df)} "
        f"formulas={df.formula.nunique()} systems={df.chemical_system.nunique()} "
        f"groups={df[group_col].nunique()} group_col={group_col}"
    )
    result = run_gpytorch_gpr(
        config,
        dataset="borg" if args.dataset == "borg" else "radical",
        dataset_label=label,
        preloaded=df,
        folds=folds,
        write_outputs=False,
        write_summary=False,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = out_dir / f"{label}_predictions.csv"
    result.results.to_csv(predictions, index=False)
    fold_metrics = (
        result.results.assign(
            abs_error=lambda x: (x.HV_actual - x.HV_gpr_predicted).abs(),
            sq_error=lambda x: (x.HV_actual - x.HV_gpr_predicted) ** 2,
        )
        .groupby("fold", as_index=False)
        .agg(n=("HV_actual", "size"), mae_hv=("abs_error", "mean"), mse=("sq_error", "mean"))
    )
    fold_metrics["rmse_hv"] = np.sqrt(fold_metrics.pop("mse"))
    fold_metrics.insert(0, "dataset", args.dataset)
    fold_metrics.insert(1, "model", args.model)
    fold_metrics.insert(2, "group", args.group)
    fold_metrics.insert(3, "cohort", cohort)
    fold_metrics.to_csv(out_dir / f"{label}_fold_metrics.csv", index=False)

    summary = pd.DataFrame(
        [{
            "dataset": args.dataset,
            "model": args.model,
            "group": args.group,
            "cohort": cohort,
            "seed": args.seed,
            "n_rows": len(df),
            "n_formulas": df.formula.nunique(),
            "n_systems": df.chemical_system.nunique(),
            "n_groups": df[group_col].nunique(),
            "mae_hv": result.gpr_mae_hv,
            "median_ae_hv": float((result.results.HV_actual - result.results.HV_gpr_predicted).abs().median()),
            "rmse_hv": result.gpr_rmse_hv,
            "fold_mae_sd_hv": fold_metrics.mae_hv.std(ddof=0),
            "fold_rmse_sd_hv": fold_metrics.rmse_hv.std(ddof=0),
        }]
    )
    summary.to_csv(out_dir / f"{label}_summary.csv", index=False)
    print("FINAL_SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
