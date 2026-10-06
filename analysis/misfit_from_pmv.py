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

from curtin_ys_prior import ELASTIC_TENSOR_DICT, LATTICE_CONSTANTS_BCC_EXP
from partial_molar_volumes import (
    RedlichKister,
    load_surface,
    select_order,
    sigma_from_volumes,
)

KB = 8.617333262e-5
EV_TO_GPA = 160.21766208
STRAIN_RATE_LOG = np.log(1e7)

# Table 7 of the submission, all inside the Nb-Ti-W surface.
TABLE7 = [
    ("Ti59W41", {"Ti": 0.59, "W": 0.41}, -2.67, 1.440, 14.9, (13.7, 15.2, 20.1)),
    ("Ti70W30", {"Ti": 0.70, "W": 0.30}, -2.27, 1.243, 12.8, (11.8, 13.2, 18.3)),
    ("Ti80W20", {"Ti": 0.80, "W": 0.20}, -1.79, 0.967, 10.5, (9.5, 11.0, 16.0)),
    ("Ti59Nb20W20", {"Ti": 0.59, "Nb": 0.20, "W": 0.20}, -1.47, 0.918, 7.4, (7.0, 8.0, 11.4)),
    ("Ti91Nb9", {"Ti": 0.91, "Nb": 0.09}, -1.45, 0.009, 775.0, (333.0, 2001.0, 4e5)),
]


def rom_elastic(frac: dict[str, float]) -> tuple[float, float]:
    """(G, nu) from rule-of-mixtures elastic constants, as the prior does."""
    c11, c12, c44 = (
        sum(x * ELASTIC_TENSOR_DICT[el][i] for el, x in frac.items()) for i in range(3)
    )
    B = (c11 + 2 * c12) / 3
    G = np.sqrt(c44 * (c11 - c12) / 2)
    return float(G), float((3 * B - 2 * G) / (2 * (3 * B + G)))


def strength(G: float, nu: float, sigma: float, vbar: float, temps=(0.0, 300.0, 1500.0)):
    """Maresca-Curtin yield stress in MPa at each temperature, plus b and Ea."""
    b = (2 * vbar) ** (1 / 3) * np.sqrt(3) / 2
    ela = ((1 + nu) / (1 - nu)) ** (4 / 3)
    ys0 = 0.04 * (1 / 12) ** (-1 / 3) * G * ela * (sigma / b**6) ** (2 / 3) * 1000 * 3
    Ea = (
        2.0
        * (1 / 12) ** (1 / 3)
        * G
        * b**3
        * ((1 + nu) / (1 - nu)) ** (2 / 3)
        * (sigma / b**6) ** (1 / 3)
        / EV_TO_GPA
    )
    out = []
    for T in temps:
        if T == 0:
            out.append(ys0)
        else:
            out.append(
                ys0 * np.exp(-1 / 0.55 * ((KB * T / Ea) * STRAIN_RATE_LOG) ** 0.91)
            )
    return np.array(out), b, Ea


def vegard(frac: dict[str, float], pure: dict[str, float]) -> tuple[float, float]:
    """(Vbar, Sigma) under linear mixing of the given pure-element volumes."""
    vb = sum(x * pure[el] for el, x in frac.items())
    sig = sum(x * (pure[el] - vb) ** 2 for el, x in frac.items())
    return float(vb), float(sig)


def fit_surface(name: str):
    surf = load_surface(name)
    best = select_order(surf).iloc[0]
    model = RedlichKister(
        surf.elements, surf.pure_volumes(), int(best["order"]), bool(best["ternary"])
    ).fit(surf.c, surf.v)
    return surf, model


def evaluate(model, elements, frac: dict[str, float]):
    """Vbar, partial molar volumes and Sigma_pmv at one composition."""
    c = np.array([[frac.get(el, 0.0) for el in elements]], dtype=float)
    c = c / c.sum()
    V, vi = model.partial_molar_volumes(c)
    return float(V[0]), vi[0], float(sigma_from_volumes(c, vi, V)[0])


def main():
    surf, model = fit_surface("Nb-Ti-W")
    els = surf.elements
    pure_mlip = dict(zip(els, surf.pure_volumes()))
    pure_lit = {el: LATTICE_CONSTANTS_BCC_EXP[el] ** 3 / 2 for el in els}

    print("Nb-Ti-W pure-element atomic volumes (A^3/atom)")
    for el in els:
        print(
            f"   {el:>2}  MLIP {pure_mlip[el]:7.3f}   experimental-table "
            f"{pure_lit[el]:7.3f}   diff {100 * (pure_mlip[el] / pure_lit[el] - 1):+6.2f}%"
        )

    rows = []
    for label, frac, f_paper, sig0_paper, dsig_paper, dsy_paper in TABLE7:
        G, nu = rom_elastic(frac)
        Vbar_mlip, vi, sig_pmv = evaluate(model, els, frac)

        vb_lit, sig_lit = vegard(frac, pure_lit)
        vb_ml, sig_ml = vegard(frac, pure_mlip)
        f_meas = Vbar_mlip / vb_ml - 1  # bowing vs the MLIP Vegard plane

        # the paper's two conventions, both anchored on the MLIP Vegard baseline
        sig_shift = sig_ml + (Vbar_mlip - vb_ml) ** 2
        sig_rescale = (1 + f_meas) ** 2 * sig_ml

        ys_base, _, _ = strength(G, nu, sig_ml, vb_ml)
        ys_pmv, _, _ = strength(G, nu, sig_pmv, Vbar_mlip)
        ys_shift, _, _ = strength(G, nu, sig_shift, Vbar_mlip)

        rows.append(
            dict(
                composition=label,
                f_pct=100 * f_meas,
                f_paper=f_paper,
                sigma0_lit=sig_lit,
                sigma0_paper=sig0_paper,
                sigma0_mlip=sig_ml,
                sigma_pmv=sig_pmv,
                dsigma_pmv_pct=100 * (sig_pmv / sig_ml - 1),
                dsigma_shift_pct=100 * (sig_shift / sig_ml - 1),
                dsigma_paper_pct=dsig_paper,
                dsy0_pmv_pct=100 * (ys_pmv[0] / ys_base[0] - 1),
                dsy300_pmv_pct=100 * (ys_pmv[1] / ys_base[1] - 1),
                dsy1500_pmv_pct=100 * (ys_pmv[2] / ys_base[2] - 1),
                dsy0_shift_pct=100 * (ys_shift[0] / ys_base[0] - 1),
                dsy0_paper_pct=dsy_paper[0],
                dsy300_paper_pct=dsy_paper[1],
                # absolute changes, which the reviewer asked for explicitly
                sy300_base_MPa=ys_base[1],
                sy300_pmv_MPa=ys_pmv[1],
                dHV300=(ys_pmv[1] - ys_base[1]) * 3 / 9.81,
                pmv=" ".join(f"{el}:{v:.2f}" for el, v in zip(els, vi) if frac.get(el)),
            )
        )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("Sigma: paper's conventions vs partial molar volumes  (all Sigma in A^6)")
    print("=" * 100)
    print(
        df[
            [
                "composition",
                "sigma0_paper",
                "sigma0_lit",
                "sigma0_mlip",
                "sigma_pmv",
                "dsigma_paper_pct",
                "dsigma_shift_pct",
                "dsigma_pmv_pct",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:9.3f}")
    )

    print("\n" + "=" * 100)
    print("Bowing and induced strength deviation (%)")
    print("=" * 100)
    print(
        df[
            [
                "composition",
                "f_paper",
                "f_pct",
                "dsy0_paper_pct",
                "dsy0_shift_pct",
                "dsy0_pmv_pct",
                "dsy300_paper_pct",
                "dsy300_pmv_pct",
                "dsy1500_pmv_pct",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:9.2f}")
    )

    print("\n" + "=" * 100)
    print("Absolute changes at 300 K (reviewer point 6, final paragraph)")
    print("=" * 100)
    print(
        df[["composition", "sy300_base_MPa", "sy300_pmv_MPa", "dHV300"]].to_string(
            index=False, float_format=lambda x: f"{x:10.1f}"
        )
    )

    print("\npartial molar volumes at each composition (A^3/atom):")
    for r in rows:
        print(f"   {r['composition']:<12} {r['pmv']}")

    out = Path(__file__).resolve().parent / "misfit_from_pmv_table7.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
