"""Compute Borg volume stability and fold-level error ranges from archived runs."""
from pathlib import Path
import numpy as np
import pandas as pd
from lib.physics import SUPPORTED_ELEMENTS, ELEMENTAL_BCC_VOLUMES
GV = Path("results/grouped_validation")
def volume_stability() -> pd.DataFrame:
    """Across-refit spread of the learned per-element volume correction on Borg (formula-grouped)."""
    base = dict(zip(SUPPORTED_ELEMENTS, ELEMENTAL_BCC_VOLUMES))
    frames = []
    for seed in range(5):
        d = pd.read_csv(GV / f"borg_effective_formula_unique_samples_seed{seed}_predictions.csv")
        d["row"] = d["formula"] + "#" + d.groupby("formula").cumcount().astype(str)
        d["seed"] = seed
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    rows = []
    for el in SUPPORTED_ELEMENTS:
        present = d[f"comp_{el}"] > 1e-10
        if present.sum() == 0:
            continue
        rel = (d.loc[present, f"effective_volume_{el}"] / base[el] - 1) * 100  # % correction
        g = pd.DataFrame({"row": d.loc[present, "row"], "rel": rel})
        per_row = g.groupby("row").rel.agg(["mean", "std"])
        sign_agree = g.groupby("row").rel.apply(lambda x: (np.sign(x) == np.sign(x.mean())).mean())
        rows.append({
            "element": el,
            "n_alloys": len(per_row),
            "mean_correction_pct": float(per_row["mean"].mean()),
            "mean_abs_correction_pct": float(per_row["mean"].abs().mean()),
            "across_refit_sd_pct": float(per_row["std"].mean()),
            "sign_agreement": float(sign_agree.mean()),
        })
    return pd.DataFrame(rows).sort_values("n_alloys", ascending=False)


def fold_level() -> pd.DataFrame:
    frames = []
    for path in GV.glob("*_seed[0-4]_fold_metrics.csv"):
        m = pd.read_csv(path)
        m["seed"] = int(path.name.split("_seed")[1][0])
        frames.append(m)
    f = pd.concat(frames, ignore_index=True)
    f = f[f.cohort == "unique_samples"]
    return (f.groupby(["dataset", "group", "model"])
             .agg(n_folds=("mae_hv", "size"), fold_mae_min=("mae_hv", "min"), fold_mae_median=("mae_hv", "median"),
                  fold_mae_max=("mae_hv", "max"), fold_mae_sd=("mae_hv", "std")).reset_index())

def main():
    out=GV/"uncertainty";out.mkdir(parents=True,exist_ok=True)
    volume_stability().to_csv(out/"volume_stability.csv",index=False)
    fold_level().to_csv(out/"fold_level.csv",index=False)
    print("Wrote Borg diagnostics")
if __name__=="__main__":main()
