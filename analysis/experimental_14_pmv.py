"""Public analysis of cached MLIP volumes and published validation compositions."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from curtin_ys_prior import LATTICE_CONSTANTS_BCC_EXP
from effective_volume_gp import ELEMENTAL_HV
from misfit_from_pmv import fit_surface, rom_elastic, strength, vegard
from partial_molar_volumes import SURFACES, sigma_from_volumes

T_ROOM = 298.15

# Table 2 of the submission: XRF compositions in at.%, measured HV, and the
# paper's own Curtin / Vela / model predictions for reference.
ALLOYS = [
    ("W28.2Ta70.6", {"W": 28.2, "Ta": 70.6}, 399, 192.0, 371.0, 376.0),
    ("W67.0Ta31.8", {"W": 67.0, "Ta": 31.8}, 488, 302.0, 445.0, 498.0),
    ("Mo81.5Ta17.9", {"Mo": 81.5, "Ta": 17.9}, 344, 248.0, 395.0, 428.0),
    ("Mo55.2Ti44.8", {"Mo": 55.2, "Ti": 44.8}, 385, 298.0, 448.0, 360.0),
    ("Mo62.4Ti19.9Ta17.5", {"Mo": 62.4, "Ti": 19.9, "Ta": 17.5}, 486, 315.0, 467.0, 499.0),
    ("Mo41.8Ti40.7Ta17.3", {"Mo": 41.8, "Ti": 40.7, "Ta": 17.3}, 473, 255.0, 401.0, 446.0),
    ("Mo20.5Ti61.2Ta17.8", {"Mo": 20.5, "Ti": 61.2, "Ta": 17.8}, 346, 132.0, 312.0, 396.0),
    ("Nb49.1Ti16.1W34.6", {"Nb": 49.1, "Ti": 16.1, "W": 34.6}, 455, 190.0, 379.0, 608.0),
    ("Nb24.8Ti40.2W34.8", {"Nb": 24.8, "Ti": 40.2, "W": 34.8}, 491, 199.0, 488.0, 482.0),
    ("Nb12.6Ti52.9W34.4", {"Nb": 12.6, "Ti": 52.9, "W": 34.4}, 484, 199.0, 373.0, 394.0),
    ("Mo12.1Nb35.1Ta52.8", {"Mo": 12.1, "Nb": 35.1, "Ta": 52.8}, 327, 90.6, 403.0, 297.0),
    ("Mo20.4Nb21.7Ti40.0Ta17.4",
     {"Mo": 20.4, "Nb": 21.7, "Ti": 40.0, "Ta": 17.4}, 356, 131.0, 383.0, 500.0),
    ("Mo31.0Nb23.4Ti28.6Ta17.0",
     {"Mo": 31.0, "Nb": 23.4, "Ti": 28.6, "Ta": 17.0}, 418, 192.0, 435.0, 402.0),
    ("Mo42.5Nb13.2Ti35.7Ta8.6",
     {"Mo": 42.5, "Nb": 13.2, "Ti": 35.7, "Ta": 8.6}, 426, 248.0, 361.0, 480.0),
]


def normalize(frac: dict[str, float]) -> dict[str, float]:
    """Table 2 compositions do not all sum to 100, so renormalize."""
    tot = sum(frac.values())
    return {k: v / tot for k, v in frac.items()}


def pick_surface(frac: dict[str, float], models: dict) -> str:
    """Smallest cached surface whose element set covers the alloy."""
    els = set(frac)
    covering = [n for n in models if els <= set(models[n][0].elements)]
    if not covering:
        raise ValueError(f"no cached MLIP surface covers {sorted(els)}")
    return min(covering, key=lambda n: (len(models[n][0].elements), -len(models[n][0].v)))


def hv_from(G, nu, sigma, vbar):
    """Vickers hardness from the pure solid-solution prior at room temperature."""
    ys, _, _ = strength(G, nu, sigma, vbar, temps=(T_ROOM,))
    return float(ys[0]) * 3 / 9.81


def main():
    models = {name: fit_surface(name) for name in SURFACES}
    print("fitted MLIP surfaces:")
    for name, (surf, model) in models.items():
        print(
            f"   {name:<14} {len(surf.v):3d} points  order={model.order} "
            f"ternary={model.ternary}"
        )

    pure_lit = {el: a**3 / 2 for el, a in LATTICE_CONSTANTS_BCC_EXP.items()}

    rows = []
    for label, raw, exp_hv, curtin_paper, vela, ours in ALLOYS:
        frac = normalize(raw)
        G, nu = rom_elastic(frac)
        name = pick_surface(frac, models)
        surf, model = models[name]
        els = surf.elements
        pure_mlip = dict(zip(els, surf.pure_volumes()))

        c = np.array([[frac.get(e, 0.0) for e in els]], dtype=float)
        V_mlip, vi = model.partial_molar_volumes(c)
        sig_pmv = float(sigma_from_volumes(c, vi, V_mlip)[0])
        V_mlip = float(V_mlip[0])

        vb_lit, sig_lit = vegard(frac, pure_lit)
        vb_ml, sig_ml = vegard(frac, pure_mlip)

        rows.append(
            dict(
                composition=label,
                surface=name,
                exp=exp_hv,
                curtin_paper=curtin_paper,
                rom_table=hv_from(G, nu, sig_lit, vb_lit),
                rom_mlip=hv_from(G, nu, sig_ml, vb_ml),
                mlip_mean=hv_from(G, nu, sig_ml, V_mlip),
                pmv=hv_from(G, nu, sig_pmv, V_mlip),
                vela=vela,
                paper_model=ours,
                sigma_table=sig_lit,
                sigma_mlip=sig_ml,
                sigma_pmv=sig_pmv,
            )
        )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 104)
    print("Reproducing the paper's Curtin column (validation of the ROM path)")
    print("=" * 104)
    d = (df["rom_table"] - df["curtin_paper"]).abs()
    print(
        df[["composition", "curtin_paper", "rom_table"]].assign(diff=d).to_string(
            index=False, float_format=lambda x: f"{x:9.1f}"
        )
    )
    print(f"\nmax |difference| from the published Curtin column: {d.max():.1f} HV")

    print("\n" + "=" * 104)
    print("Predicted Vickers hardness, varying only where the volumes come from")
    print("=" * 104)
    print(
        df[
            ["composition", "surface", "exp", "rom_table", "rom_mlip", "mlip_mean", "pmv"]
        ].to_string(index=False, float_format=lambda x: f"{x:9.1f}")
    )

    print("\n" + "=" * 104)
    print("Error against measured hardness (HV)")
    print("=" * 104)
    summary = []
    for col, tag in (
        ("rom_table", "ROM, paper's Table 5 constants"),
        ("rom_mlip", "ROM, MLIP pure-element anchors"),
        ("mlip_mean", "ROM misfit + MLIP mean volume"),
        ("pmv", "MLIP partial molar volumes"),
    ):
        err = df[col] - df["exp"]
        summary.append(
            dict(
                variant=tag,
                MAE=err.abs().mean(),
                RMSE=np.sqrt((err**2).mean()),
                bias=err.mean(),
                n_within_100=(err.abs() < 100).sum(),
            )
        )
    print(
        pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:9.1f}")
    )

    print("\nfor reference, the fitted models on the same 14 alloys:")
    for col, tag in (("vela", "Vela"), ("paper_model", "this paper's full model")):
        err = df[col] - df["exp"]
        print(f"   {tag:<26} MAE {err.abs().mean():6.1f} HV   bias {err.mean():+6.1f} HV")

    variants = [
        ("rom_table", "ROM, paper's Table 5 constants"),
        ("rom_mlip", "ROM, MLIP pure-element anchors"),
        ("mlip_mean", "ROM misfit + MLIP mean volume"),
        ("pmv", "MLIP partial molar volumes"),
    ]

    # Every variant underpredicts by ~200 HV because Eq. (2) is the pure
    # solid-solution term: it has no intrinsic-hardness floor. Appendix F.3 does
    # include one. Adding it back makes the comparison between volume treatments
    # meaningful instead of being swamped by a shared offset.
    print("\n" + "=" * 104)
    print("With the intrinsic-hardness term of Appendix F.3 restored")
    print("=" * 104)
    intr = np.array(
        [
            sum(x * ELEMENTAL_HV.get(el, 0.0) for el, x in normalize(raw).items())
            for _, raw, *_ in ALLOYS
        ]
    )
    df["intrinsic"] = intr
    rows_i = []
    for col, tag in variants:
        err = (df[col] + intr) - df["exp"]
        rows_i.append(
            dict(variant=tag, MAE=err.abs().mean(), RMSE=np.sqrt((err**2).mean()),
                 bias=err.mean())
        )
    print(pd.DataFrame(rows_i).to_string(index=False, float_format=lambda x: f"{x:9.1f}"))

    # A constant offset is absorbable by any calibration, so the question that
    # actually matters is whether the volume treatment improves the
    # composition-dependent part. Remove each variant's own mean error and look
    # at what is left.
    print("\n" + "=" * 104)
    print("Bias-corrected: does the volume treatment improve the composition-dependent part?")
    print("=" * 104)
    rows_b = []
    for col, tag in variants:
        err = df[col] - df["exp"]
        centred = err - err.mean()
        rows_b.append(
            dict(
                variant=tag,
                resid_MAE=centred.abs().mean(),
                resid_SD=centred.std(ddof=1),
                pearson_r=np.corrcoef(df[col], df["exp"])[0, 1],
                spearman_r=pd.Series(df[col]).corr(pd.Series(df["exp"]), method="spearman"),
            )
        )
    print(pd.DataFrame(rows_b).to_string(index=False, float_format=lambda x: f"{x:9.3f}"))

    out = Path(__file__).resolve().parents[1] / "results/experimental_14_pmv.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
