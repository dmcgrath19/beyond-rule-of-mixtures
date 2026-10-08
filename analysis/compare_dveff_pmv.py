"""Compare learned effective-volume misfits of Borg alloys in the Mo-Nb-Ta-Ti-W space with MLIP partial molar volumes."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import predict_misfit  # noqa: E402
from volume_surface import BASIS, parse_composition  # noqa: E402

warnings.filterwarnings("ignore")

QUINARY = set(BASIS)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dv-eff", default="results/learned_volumes.csv",
                   help="output of effective_volume_gp.py --dump-volumes")
    p.add_argument("-o", "--out", help="write the joined comparison here")
    a = p.parse_args(argv)

    learned = pd.read_csv(a.dv_eff)
    surf, fits, model = predict_misfit.build()

    rows = []
    skipped = []
    for formula, grp in learned.groupby("formula"):
        els = set(grp.element)
        if not els <= QUINARY:
            skipped.append((formula, "outside Mo-Nb-Ta-Ti-W"))
            continue
        c = parse_composition(formula, tuple(BASIS))
        pmv = {r["element"]: r for r in predict_misfit.evaluate(surf, fits, model, c)}
        supported = all(r["supported"] for r in pmv.values())
        for _, r in grp.iterrows():
            ref = pmv.get(r.element)
            if ref is None:
                continue
            rows.append(dict(
                formula=formula, element=r.element, x=r.x,
                dV_eff=r.dV_eff, dV_pmv=ref["dV_pmv"], dV_pmv_unc=ref["dV_pmv_unc"],
                dV_rom=ref["dV_rom"], supported=supported,
                HV_exp=r.HV_exp, HV_oof=r.HV_oof,
            ))

    if not rows:
        print("no Borg alloys fall inside the MLIP quinary; nothing to compare")
        return 1

    df = pd.DataFrame(rows)
    df["gap"] = df.dV_eff - df.dV_pmv
    df["sign_agrees"] = np.sign(df.dV_eff) == np.sign(df.dV_pmv)

    print("=" * 92)
    print("LEARNED EFFECTIVE-VOLUME MISFITS vs MLIP PARTIAL MOLAR VOLUMES")
    print("=" * 92)
    print(f"Borg formulae inside Mo-Nb-Ta-Ti-W : {df.formula.nunique()}")
    print(f"  of which the MLIP surface supports: {df[df.supported].formula.nunique()}")
    if skipped:
        print(f"  skipped (outside the quinary)    : {len(skipped)}")

    show = df.copy()
    show["dV_pmv"] = [f"{v:+.3f}+/-{u:.3f}" for v, u in zip(df.dV_pmv, df.dV_pmv_unc)]
    cols = ["formula", "element", "x", "dV_eff", "dV_pmv", "dV_rom", "gap",
            "sign_agrees", "supported"]
    print()
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    ok = df[df.supported]
    print()
    print("-" * 92)
    print("SUMMARY (data-supported rows only)")
    print("-" * 92)
    if len(ok) < 3:
        print(f"only {len(ok)} supported alloy-element rows -- too few for a correlation")
    else:
        r = np.corrcoef(ok.dV_eff, ok.dV_pmv)[0, 1]
        r_rom = np.corrcoef(ok.dV_eff, ok.dV_rom)[0, 1]
        print(f"rows                       : {len(ok)} over {ok.formula.nunique()} alloys")
        print(f"corr(dV_eff, dV_pmv)       : {r:+.3f}")
        print(f"corr(dV_eff, dV_rom)       : {r_rom:+.3f}")
        print(f"sign agreement             : {int(ok.sign_agrees.sum())}/{len(ok)} "
              f"({100 * ok.sign_agrees.mean():.0f}%)")
        print(f"median |dV_eff - dV_pmv|   : {ok.gap.abs().median():.3f} A^3/atom")
        print(f"max    |dV_eff - dV_pmv|   : {ok.gap.abs().max():.3f} A^3/atom")
        print(f"median PMV uncertainty     : {ok.dV_pmv_unc.median():.3f} A^3/atom")
        print(f"RMS dV_eff / RMS dV_pmv    : "
              f"{np.sqrt((ok.dV_eff**2).mean()) / np.sqrt((ok.dV_pmv**2).mean()):.3f}")

    if a.out:
        df.to_csv(a.out, index=False)
        print(f"\nwrote {a.out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
