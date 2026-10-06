"""Score accuracy and predictive uncertainty for the grouped-validation runs.

Reads the out-of-fold prediction files written by utils/run_grouped_validation.py
and
writes summary tables to results/grouped_validation/uncertainty/.

Calibration uses ``HV_gpr_pred_std`` (latent GP variance plus likelihood noise),
which is the predictive distribution of a new measurement.  The latent-only
``HV_gpr_std`` is what earlier figures plotted as whiskers; its coverage is
reported alongside for comparison.

Run:
    uv run python -m utils.evaluate_uncertainty
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run import COMPOSITION_COLS
from utils.load_data import BORG_NUMERIC_COLS
from utils.run_grouped_validation import chemical_system

IN_DIR = Path("results/grouped_validation")
OUT_DIR = IN_DIR / "uncertainty"
Z68, Z95 = 1.0, 1.959964
N_BOOT = 10_000
SEEDS = range(5)  # repeated-CV seeds used by the sweep
FILE_RE = re.compile(
    r"(?P<dataset>borg)_(?P<model>.+)_(?P<group>random|formula|reference|sample|system)"
    r"_(?P<cohort>rows|unique_samples(?:_[a-z]+)?)_seed(?P<seed>\d+)_predictions\.csv"
)


# ── scoring rules ────────────────────────────────────────────────────────────

def gaussian_nlpd(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return 0.5 * np.log(2 * np.pi * sd**2) + 0.5 * ((y - mu) / sd) ** 2


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def score(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict[str, float]:
    err = y - mu
    z = np.abs(err) / sd
    return {
        "mae_hv": float(np.mean(np.abs(err))),
        "rmse_hv": float(np.sqrt(np.mean(err**2))),
        "median_ae_hv": float(np.median(np.abs(err))),
        "cov68": float(np.mean(z <= Z68)),
        "cov95": float(np.mean(z <= Z95)),
        "nlpd": float(np.mean(gaussian_nlpd(y, mu, sd))),
        "crps_hv": float(np.mean(gaussian_crps(y, mu, sd))),
        "mean_sd_hv": float(np.mean(sd)),
        "rms_z": float(np.sqrt(np.mean(z**2))),
        "spearman_sd_abs_err": float(stats.spearmanr(sd, np.abs(err)).statistic),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def nearest_train_distance(df: pd.DataFrame) -> np.ndarray:
    """Distance from each row to its nearest training row in standardised descriptor space."""
    x = df[BORG_NUMERIC_COLS].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    folds = df["fold"].to_numpy()
    out = np.empty(len(df))
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        d = np.linalg.norm(x[te, None, :] - x[None, tr, :], axis=-1)
        out[te] = d.min(axis=1)
    return out


def cross_fitted_scale(df: pd.DataFrame) -> np.ndarray:
    """Per-row variance scale s_f = RMS(z) over the *other* folds (no use of the row's own fold)."""
    z2 = ((df.HV_actual - df.HV_gpr_predicted) / df.HV_gpr_pred_std) ** 2
    folds = df["fold"].to_numpy()
    scale = np.empty(len(df))
    for f in np.unique(folds):
        scale[folds == f] = np.sqrt(z2[folds != f].mean())
    return scale


def cluster_bootstrap_diff(diff: np.ndarray, clusters: np.ndarray, seed: int = 0) -> tuple[float, float, float, float]:
    """Mean paired difference with a cluster bootstrap CI and two-sided p-value."""
    rng = np.random.default_rng(seed)
    labels, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv)
    draws = rng.integers(0, len(labels), size=(N_BOOT, len(labels)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return float(diff.mean()), float(lo), float(hi), float(max(p, 1 / N_BOOT))


def row_key(df: pd.DataFrame) -> pd.Series:
    if "borg_row_id" in df:
        return df["borg_row_id"].astype(str)
    if "sample_id" in df and df["sample_id"].notna().all():
        return df["sample_id"].astype(str)
    # Borg repeats formulas; cohort files share row order, so occurrence index disambiguates.
    return df["formula"].astype(str) + "#" + df.groupby("formula").cumcount().astype(str)


# ── cross-validation runs ────────────────────────────────────────────────────

def load_runs() -> pd.DataFrame:
    frames = []
    for path in sorted(IN_DIR.glob("*_predictions.csv")):
        m = FILE_RE.fullmatch(path.name)
        if m is None:
            continue
        df = pd.read_csv(path)
        if "HV_gpr_pred_std" not in df:
            continue  # pre-dates the predictive-std output; latent std only
        df = df.copy()
        meta = m.groupdict()
        meta["seed"] = int(meta["seed"])
        if meta["seed"] not in SEEDS:
            continue
        df = df.assign(**meta)
        df["key"] = row_key(df).astype(str)
        df["chemical_system"] = chemical_system(df) if set(COMPOSITION_COLS) <= set(df) else df["formula"]
        df["dist_nn"] = nearest_train_distance(df)
        df["scale"] = cross_fitted_scale(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def cv_tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed, strata = [], []
    for keys, g in runs.groupby(["dataset", "cohort", "group", "model", "seed"]):
        y, mu = g.HV_actual.to_numpy(), g.HV_gpr_predicted.to_numpy()
        row = dict(zip(["dataset", "cohort", "group", "model", "seed"], keys), n=len(g))
        row.update(score(y, mu, g.HV_gpr_pred_std.to_numpy()))
        latent = np.abs(y - mu) / g.HV_gpr_std.to_numpy()
        row["latent_cov68"] = float(np.mean(latent <= Z68))
        row["latent_cov95"] = float(np.mean(latent <= Z95))
        recal = score(y, mu, (g.HV_gpr_pred_std * g.scale).to_numpy())
        row.update({f"recal_{k}": recal[k] for k in ("cov68", "cov95", "nlpd", "crps_hv")})
        row["recal_scale"] = float(g.scale.mean())
        row["spearman_dist_sd"] = float(stats.spearmanr(g.dist_nn, g.HV_gpr_pred_std).statistic)
        row["spearman_dist_abs_err"] = float(stats.spearmanr(g.dist_nn, np.abs(y - mu)).statistic)
        per_seed.append(row)

        near = g.dist_nn <= g.dist_nn.median()
        for name, mask in (("near", near), ("far", ~near)):
            s = score(y[mask], mu[mask], g.HV_gpr_pred_std.to_numpy()[mask])
            strata.append(dict(zip(["dataset", "cohort", "group", "model", "seed"], keys), stratum=name,
                               mae_hv=s["mae_hv"], mean_sd_hv=s["mean_sd_hv"], cov68=s["cov68"],
                               cov95=s["cov95"], rms_z=s["rms_z"]))
    per_seed = pd.DataFrame(per_seed)
    strata = pd.DataFrame(strata)
    idx = ["dataset", "cohort", "group", "model"]
    metrics = [c for c in per_seed.columns if c not in idx + ["seed", "n"]]
    agg = per_seed.groupby(idx).agg(n=("n", "first"), n_seeds=("seed", "size"),
                                    **{f"{c}": (c, "mean") for c in metrics},
                                    **{f"{c}_sd": (c, "std") for c in ("mae_hv", "rmse_hv")}).reset_index()
    strata_agg = strata.groupby(idx + ["stratum"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    return agg, strata_agg


PAIRS = [
    ("effective", "analytic"),
    ("effective", "composition"),
    ("analytic", "composition"),
    ("effective_full", "analytic_full"),
    ("effective_full", "composition_full"),
    ("analytic_full", "composition_full"),
    ("effective_full", "effective"),
    ("composition_full", "composition"),
    ("effective", "vela"),
    ("effective", "gao_rf"),
    ("effective", "gao_svr"),
    ("analytic", "vela"),
    ("composition", "vela"),
    ("effective_full", "vela"),
]


def paired_tables(runs: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged absolute error per row, differenced between arms on shared rows and seeds."""
    runs = runs.assign(abs_err=(runs.HV_actual - runs.HV_gpr_predicted).abs())
    rows = []
    for (dataset, cohort, group), g in runs.groupby(["dataset", "cohort", "group"]):
        for a, b in PAIRS:
            ga, gb = g[g.model == a], g[g.model == b]
            seeds = sorted(set(ga.seed) & set(gb.seed))
            if not seeds:
                continue
            ea = ga[ga.seed.isin(seeds)].groupby("key").agg(e=("abs_err", "mean"), sys=("chemical_system", "first"))
            eb = gb[gb.seed.isin(seeds)].groupby("key").abs_err.mean()
            common = ea.index.intersection(eb.index)
            diff = (ea.loc[common, "e"] - eb.loc[common]).to_numpy()
            mean, lo, hi, p = cluster_bootstrap_diff(diff, ea.loc[common, "sys"].to_numpy())
            rows.append(dict(dataset=dataset, cohort=cohort, group=group, model_a=a, model_b=b,
                             n_rows=len(common), n_seeds=len(seeds), mae_a=float(ea.loc[common, "e"].mean()),
                             mae_b=float(eb.loc[common].mean()), diff_mae_hv=mean, ci_lo=lo, ci_hi=hi, p_boot=p,
                             frac_rows_a_better=float(np.mean(diff < 0))))
    return pd.DataFrame(rows)


# ── recalibration ────────────────────────────────────────────────────────────

CONFORMAL_LEVEL = 0.90


def conformal_quantile(scores: np.ndarray, level: float = CONFORMAL_LEVEL) -> float:
    n = len(scores)
    k = min(int(np.ceil((n + 1) * level)), n)
    return float(np.sort(scores)[k - 1])


def interval_half_widths(g: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Cross-fitted 90% interval half-widths for one run, plus each row's distance stratum.

    For each held-out fold, calibration scores |r|/sigma come from the other
    folds' out-of-fold predictions only.  The distance stratum depends on
    descriptors alone (never on hardness): rows nearer than the calibration
    median distance are "near", the rest "far".
    """
    r = (g.HV_actual - g.HV_gpr_predicted).abs().to_numpy()
    sd = g.HV_gpr_pred_std.to_numpy()
    dist = g.dist_nn.to_numpy()
    folds = g.fold.to_numpy()
    half = np.empty(len(g), dtype=object)
    width = {m: np.empty(len(g)) for m in ("raw", "global_scale", "global_conformal", "stratified_conformal")}
    zq = stats.norm.ppf(0.5 + CONFORMAL_LEVEL / 2)
    for f in np.unique(folds):
        te, cal = folds == f, folds != f
        z_cal = r[cal] / sd[cal]
        d_med = np.median(dist[cal])
        near_cal = dist[cal] <= d_med
        near_te = dist[te] <= d_med
        half[te] = np.where(near_te, "near", "far")
        width["raw"][te] = zq * sd[te]
        width["global_scale"][te] = zq * np.sqrt(np.mean(z_cal**2)) * sd[te]
        width["global_conformal"][te] = conformal_quantile(z_cal) * sd[te]
        q_near, q_far = conformal_quantile(z_cal[near_cal]), conformal_quantile(z_cal[~near_cal])
        width["stratified_conformal"][te] = np.where(near_te, q_near, q_far) * sd[te]
    return width, half


def recalibration_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Coverage and mean width of cross-fitted 90% intervals, by distance stratum."""
    rows = []
    for keys, g in runs.groupby(["dataset", "cohort", "group", "model", "seed"]):
        g = g.reset_index(drop=True)
        r = (g.HV_actual - g.HV_gpr_predicted).abs().to_numpy()
        width, half = interval_half_widths(g)
        for method, w in width.items():
            for regime in ("near", "far", "all"):
                mask = np.ones(len(g), bool) if regime == "all" else half == regime
                rows.append(dict(zip(["dataset", "cohort", "group", "model", "seed"], keys), method=method,
                                 regime=regime, coverage=float(np.mean(r[mask] <= w[mask])),
                                 mean_width_hv=float(np.mean(2 * w[mask])), n=int(mask.sum())))
    t = pd.DataFrame(rows)
    return (t.groupby(["dataset", "cohort", "group", "model", "method", "regime"])
             .agg(coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
                  mean_width_hv=("mean_width_hv", "mean"), n=("n", "mean")).reset_index())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if runs.empty:
        raise ValueError("No Borg predictions were found")
    cv, strata = cv_tables(runs)
    for name, frame in [("cv_summary", cv), ("cv_distance_strata", strata),
                        ("cv_paired_differences", paired_tables(runs)),
                        ("cv_recalibration_90", recalibration_table(runs))]:
        frame.to_csv(OUT_DIR / (name + ".csv"), index=False)
    print(cv[["group", "model", "n", "n_seeds", "mae_hv", "mae_hv_sd"]].round(3).to_string(index=False))

if __name__ == "__main__":
    main()
