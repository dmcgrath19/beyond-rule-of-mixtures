"""Borg sample-efficiency experiment with shared formula-grouped subsets."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from borg_experiments import Data  # noqa: E402
from calibration import ARMS, crps_gaussian  # noqa: E402
from effective_volume_gp import curtin_analytic_hv  # noqa: E402
from utils.paper_training import fit_fold, predict  # noqa: E402
from reframe_check import build_borg_recomputed  # noqa: E402

warnings.filterwarnings("ignore")

# fit_fold seeds torch and numpy, but multithreaded CPU reductions still change
# floating-point summation order between processes, and 650 Adam steps amplify that
# into a visibly different estimate. Two 40-repeat runs with identical arguments gave
# a pooled advantage of +5.9 HV (p=0.015) and +4.2 HV (p=0.067) for this reason.
# Pinning to one thread makes a run reproducible so the estimate can be trusted.
import torch  # noqa: E402

torch.set_num_threads(1)


def rule(t: str) -> None:
    print(f"\n{'=' * 92}\n{t}\n{'=' * 92}")


def subsample_groups(pool_idx, groups, target, rng):
    """Take whole formulae from the pool until about `target` rows are collected."""
    # sorted: a set of strings iterates in PYTHONHASHSEED-dependent order
    uniq = sorted({groups[i] for i in pool_idx})
    rng.shuffle(uniq)
    taken, rows = set(), []
    for g in uniq:
        gi = [i for i in pool_idx if groups[i] == g]
        if rows and len(rows) + len(gi) > target:
            continue
        taken.add(g)
        rows += gi
        if len(rows) >= target:
            break
    return np.array(sorted(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=150)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[25, 35, 50, 74])
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--pool-sizes", type=int, nargs="*", default=[25, 35, 50],
                    help="sizes forming the pre-specified mid-range hypothesis")
    ap.add_argument("--plot-out", default="results/learning_curve/learning_curve.png")
    ap.add_argument("--csv-out", default="results/learning_curve/learning_curve.csv")
    ap.add_argument("--diff-out", default="results/learning_curve/learning_curve_diffs.csv")
    ap.add_argument("--split-out", default="results/learning_curve/splits.json")
    ap.add_argument("--split-in", help="reuse a saved split/subset manifest")
    ap.add_argument("--n1", type=int, default=200)
    ap.add_argument("--n2", type=int, default=800)
    a = ap.parse_args(argv)
    import json
    for path in [a.plot_out, a.csv_out, a.diff_out, a.split_out]:
        if path: Path(path).parent.mkdir(parents=True, exist_ok=True)
    records = []
    saved = json.loads(Path(a.split_in).read_text()) if a.split_in else None

    d = Data()
    Xk, phys, comp, y, formulas = build_borg_recomputed(d.basis)
    v_base = d.v_base()
    groups = np.array(formulas)
    print(f"Borg n={len(y)}, {len(set(formulas))} unique formulae, "
          f"recomputed features used for both training and prediction")
    print(f"{a.repeats} repeats, grouped {int(100*a.test_frac)}% held-out test set, "
          f"training sizes {a.sizes}")

    # analytic prior needs no training: a horizontal reference line
    hv_prior = np.asarray(curtin_analytic_hv(phys), dtype=float)
    ok = np.isfinite(hv_prior)
    print(f"Curtin analytical prior (no training): "
          f"MAE {np.abs(hv_prior[ok]-y[ok]).mean():.1f} HV over {ok.sum()} rows\n")

    # results[arm][size] -> list of MAE, one per repeat
    mae = {arm: {n: [] for n in a.sizes} for arm in ARMS}
    crp = {arm: {n: [] for n in a.sizes} for arm in ARMS}
    # paired per-repeat differences, composition-only minus effective volume
    diff = {n: [] for n in a.sizes}
    actual = {n: [] for n in a.sizes}

    gss = GroupShuffleSplit(n_splits=a.repeats, test_size=a.test_frac,
                            random_state=0)
    split_iter = [(np.asarray(r["pool"]), np.asarray(r["test"])) for r in saved] if saved is not None else gss.split(Xk, y, groups)
    for rep, (pool, test) in enumerate(split_iter):
        record = {"repeat": rep, "seed": rep, "pool": pool.tolist(), "test": test.tolist(), "training": {}}
        records.append(record)
        rng = np.random.default_rng(100 + rep)
        for n in a.sizes:
            tr = np.asarray(saved[rep]["training"][str(n)], dtype=int) if saved is not None else subsample_groups(pool, groups, n, rng)
            if not set(tr) <= set(pool) or set(groups[tr]) & set(groups[test]):
                raise ValueError("Saved split contains leakage or invalid training rows")
            record["training"][str(n)] = tr.tolist()
            if len(tr) < 8:
                continue
            actual[n].append(len(tr))
            per_arm = {}
            for arm, kw in ARMS.items():
                fit = fit_fold(Xk[tr], phys[tr], comp[tr], y[tr], v_base,
                               seed=rep, n1=a.n1, n2=a.n2, **kw)
                mu, sd = predict(fit, Xk[test], phys[test], comp[test])
                e = float(np.abs(mu - y[test]).mean())
                c = crps_gaussian(y[test], mu, sd)
                mae[arm][n].append(e)
                crp[arm][n].append(c)
                per_arm[arm] = e
            diff[n].append(per_arm["Composition-only GP"]
                           - per_arm["PI-GP + effective volume"])
        print(f"  repeat {rep+1}/{a.repeats} done "
              f"(test n={len(test)}, train sizes "
              f"{[actual[n][-1] if actual[n] else 0 for n in a.sizes]})")

    if a.split_out:
        Path(a.split_out).write_text(json.dumps(records, indent=2) + "\n")
    rule("LEARNING CURVE -- test MAE (HV), mean +/- sd over repeats")
    hdr = f"{'train n':>8} " + " ".join(f"{k[:22]:>24}" for k in ARMS)
    print(hdr)
    print("-" * len(hdr))
    for n in a.sizes:
        if not mae["Composition-only GP"][n]:
            continue
        row = f"{int(np.mean(actual[n])):>8} "
        for arm in ARMS:
            v = np.array(mae[arm][n])
            row += f"{v.mean():>16.1f} +/-{v.std(ddof=1):5.1f}"
        print(row)

    rule("THE CLAIM UNDER TEST -- composition-only minus effective volume")
    print("positive => the physics prior helps at that sample size.")
    print("paired across repeats, so each repeat sees identical data for both arms.\n")
    print(f"{'train n':>8} {'diff (HV)':>11} {'95% CI':>22} {'p':>9} {'wins':>7}")
    print("-" * 62)
    verdict = []
    for n in a.sizes:
        dv = np.array(diff[n])
        if len(dv) < 3:
            continue
        m = dv.mean()
        se = dv.std(ddof=1) / np.sqrt(len(dv))
        t = stats.t.ppf(0.975, len(dv) - 1)
        lo, hi = m - t * se, m + t * se
        p = stats.ttest_1samp(dv, 0).pvalue
        print(f"{int(np.mean(actual[n])):>8} {m:>+11.1f} "
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>22} {p:>9.3f} "
              f"{int((dv>0).sum())}/{len(dv):<5}")
        verdict.append((int(np.mean(actual[n])), m, lo, hi, p))

    # A per-size scan over k sizes is k tests; one p<0.05 among them is not
    # evidence. The data-efficiency claim is specifically that the prior helps in
    # the mid-small-data regime and washes out asymptotically, so test that as a
    # single pre-specified hypothesis: average each repeat's advantage over the
    # mid-range sizes, then one t-test on those per-repeat means.
    if a.pool_sizes and all(len(diff[n]) >= 3 for n in a.pool_sizes if n in diff):
        rule("POOLED PRE-SPECIFIED TEST -- one hypothesis, no multiplicity")
        pool_n = [n for n in a.pool_sizes if len(diff[n]) >= 3]
        k = min((len(diff[n]) for n in pool_n), default=0)
        per_rep = np.mean([np.array(diff[n][:k]) for n in pool_n], axis=0)
        m = per_rep.mean()
        se = per_rep.std(ddof=1) / np.sqrt(len(per_rep))
        t = stats.t.ppf(0.975, len(per_rep) - 1)
        p = stats.ttest_1samp(per_rep, 0).pvalue
        w = stats.wilcoxon(per_rep).pvalue if len(per_rep) > 5 else float("nan")
        print(f"sizes pooled: {pool_n}, {len(per_rep)} repeats")
        print(f"mean advantage of the physics prior: {m:+.1f} HV")
        print(f"95% CI [{m - t*se:+.1f}, {m + t*se:+.1f}], "
              f"t-test p={p:.4f}, signed-rank p={w:.4f}")
        print(f"positive in {int((per_rep>0).sum())} of {len(per_rep)} repeats")
        print(f"effect size (Cohen's d): {m/per_rep.std(ddof=1):+.2f}")
        # convergence check: is the advantage smaller at the largest size?
        big = a.sizes[-1]
        if len(diff[big]) >= 3:
            db = np.array(diff[big])
            print(f"\nat the largest size (n~{int(np.mean(actual[big]))}) the advantage is "
                  f"{db.mean():+.1f} HV (p={stats.ttest_1samp(db,0).pvalue:.3f}) "
                  f"-- largest-size comparison")

    # per-repeat differences, so independent runs can be pooled rather than compared
    if a.diff_out:
        pd.DataFrame({str(n): pd.Series(diff[n]) for n in a.sizes}).to_csv(
            a.diff_out, index_label="repeat")
        print(f"\nwrote {a.diff_out}")

    if a.csv_out:
        rows = []
        for n in a.sizes:
            for arm in ARMS:
                v = np.array(mae[arm][n])
                if not len(v):
                    continue
                rows.append(dict(train_n=int(np.mean(actual[n])), arm=arm,
                                 mae_mean=v.mean(), mae_sd=v.std(ddof=1),
                                 crps_mean=np.mean(crp[arm][n]),
                                 repeats=len(v)))
        pd.DataFrame(rows).to_csv(a.csv_out, index=False)
        print(f"\nwrote {a.csv_out}")

    if a.plot_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns = [int(np.mean(actual[n])) for n in a.sizes if mae["Composition-only GP"][n]]
        keep = [n for n in a.sizes if mae["Composition-only GP"][n]]
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.4))
        style = {"PI-GP (analytic mean)": ("o-", "tab:blue"),
                 "PI-GP + effective volume": ("s-", "tab:orange"),
                 "Composition-only GP": ("^-", "tab:green")}
        for arm in ARMS:
            m = np.array([np.mean(mae[arm][n]) for n in keep])
            se = np.array([np.std(mae[arm][n], ddof=1) / np.sqrt(len(mae[arm][n]))
                           for n in keep])
            mk, c = style[arm]
            ax.errorbar(ns, m, yerr=se, fmt=mk, color=c, capsize=3, ms=5, lw=1.6,
                        label=arm)
        ax.set_xlabel("training set size (rows)")
        ax.set_ylabel("held-out test MAE (HV)")
        ax.set_title("Learning curves, Borg")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        adv = np.array([np.mean(diff[n]) for n in keep])
        advse = np.array([np.std(diff[n], ddof=1) / np.sqrt(len(diff[n]))
                          for n in keep])
        bx.axhline(0, color="k", lw=1, ls="--")
        bx.errorbar(ns, adv, yerr=1.96 * advse, fmt="D-", color="tab:red",
                    capsize=3, ms=5, lw=1.6)
        bx.set_xlabel("training set size (rows)")
        bx.set_ylabel("advantage of the physics prior (HV)")
        bx.set_title("Composition-only minus effective volume")
        bx.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.plot_out, dpi=160)
        print(f"wrote {a.plot_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
