"""Checks of the partial molar volume pipeline: surface fit, autograd derivatives and dilute limits."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from curtin_ys_prior import LATTICE_CONSTANTS_BCC_EXP
from misfit_from_pmv import fit_surface
from partial_molar_volumes import REPO, sigma_from_volumes


def check_surface_reproduction(surf, model):
    """RK prediction vs raw MLIP volumes, and vs the CSV's own bowing column."""
    pred = model.predict(surf.c)
    resid = pred - surf.v
    print("1) RK surface vs raw MLIP grid")
    print(
        f"   max |residual| {np.abs(resid).max():.4f} A^3/atom "
        f"({100 * np.abs(resid).max() / surf.v.mean():.3f}% of mean volume)"
    )

    df = pd.read_csv(REPO / "nb_ti_w_eos_ternary_with_bowing.csv")
    if "delta_V_mlip_percent" in df:
        pure = dict(zip(surf.elements, surf.pure_volumes()))
        veg = surf.c @ np.array([pure[e] for e in surf.elements])
        ours = 100 * (surf.v / veg - 1)
        theirs = df["delta_V_mlip_percent"].to_numpy(float)[: len(ours)]
        good = np.isfinite(theirs)
        print(
            f"   our bowing vs CSV delta_V_mlip_percent: max diff "
            f"{np.abs(ours[good] - theirs[good]).max():.4f} percentage points"
        )


def check_autograd(model, elements, n=200, h=1e-5, seed=0):
    """Autograd partial molar volumes vs central finite differences."""
    rng = np.random.default_rng(seed)
    c = rng.dirichlet(np.ones(len(elements)), size=n)
    V, vi = model.partial_molar_volumes(c)

    # finite-difference d(nV)/dn_i at n = 1
    fd = np.empty_like(vi)
    for i in range(len(elements)):
        num = c.copy()
        num[:, i] += h
        tot_p = num.sum(axis=1)
        Vp = model.predict(num / tot_p[:, None]) * tot_p
        num = c.copy()
        num[:, i] -= h
        tot_m = num.sum(axis=1)
        Vm = model.predict(num / tot_m[:, None]) * tot_m
        fd[:, i] = (Vp - Vm) / (2 * h)

    print("\n2) autograd vs finite-difference partial molar volumes")
    print(f"   max |difference| {np.abs(vi - fd).max():.2e} A^3/atom")
    print(
        f"   Gibbs-Duhem max |sum_i c_i v_i - V| {np.abs((c * vi).sum(1) - V).max():.2e}"
    )


def check_dilute_corner(model, elements):
    """Is the dilute-solute contraction real, or an extrapolation artifact?"""
    i_ti, i_nb = elements.index("Ti"), elements.index("Nb")
    print("\n3) dilute Nb-in-Ti behaviour (is Ti91Nb9 trustworthy?)")
    print(f"   {'x_Nb':>6} {'v_Nb':>8} {'v_Ti':>8} {'Vbar':>8}")
    for x in (0.02, 0.05, 0.09, 0.15, 0.25, 0.5):
        c = np.zeros((1, len(elements)))
        c[0, i_nb], c[0, i_ti] = x, 1 - x
        V, vi = model.partial_molar_volumes(c)
        print(f"   {x:6.2f} {vi[0, i_nb]:8.3f} {vi[0, i_ti]:8.3f} {V[0]:8.3f}")
    print("   (pure Nb on this surface is 18.128 A^3/atom)")

    # how many grid points actually constrain the Nb-Ti edge?
    surf, _ = fit_surface("Nb-Ti-W")
    on_edge = surf.c[:, elements.index("W")] < 1e-6
    print(f"   grid points with x_W = 0 constraining the Nb-Ti edge: {on_edge.sum()}")


def sweep(model, elements, step=0.02):
    """Sigma_pmv / Sigma_ROM across the simplex, on both anchor conventions."""
    pure_mlip = dict(zip(elements, model.pure))
    pure_lit = {el: LATTICE_CONSTANTS_BCC_EXP[el] ** 3 / 2 for el in elements}

    grid = []
    n = int(round(1 / step))
    for a in range(n + 1):
        for b in range(n + 1 - a):
            grid.append([a * step, b * step, 1 - a * step - b * step])
    c = np.array(grid)

    V, vi = model.partial_molar_volumes(c)
    sig_pmv = sigma_from_volumes(c, vi, V)

    vm = np.array([pure_mlip[e] for e in elements])
    vl = np.array([pure_lit[e] for e in elements])
    veg_m = c @ vm
    veg_l = c @ vl
    sig_rom_m = (c * (vm[None, :] - veg_m[:, None]) ** 2).sum(1)
    sig_rom_l = (c * (vl[None, :] - veg_l[:, None]) ** 2).sum(1)

    interior = c.min(axis=1) > 1e-9
    ratio_m = sig_pmv[interior] / sig_rom_m[interior]
    ratio_l = sig_pmv[interior] / sig_rom_l[interior]

    print(f"\n4) full-simplex sweep, {interior.sum()} interior compositions")
    for tag, r in (("vs MLIP-anchored ROM", ratio_m), ("vs tabulated ROM", ratio_l)):
        print(
            f"   Sigma_pmv / Sigma_ROM {tag:<22} "
            f"median {np.median(r):5.2f}   "
            f"10th {np.percentile(r, 10):5.2f}   90th {np.percentile(r, 90):5.2f}   "
            f"fraction > 1: {100 * (r > 1).mean():4.1f}%"
        )

    out = pd.DataFrame(
        {
            **{f"x_{e}": c[:, i] for i, e in enumerate(elements)},
            "Vbar_A3": V,
            **{f"v_{e}_A3": vi[:, i] for i, e in enumerate(elements)},
            "sigma_pmv_A6": sig_pmv,
            "sigma_rom_mlip_A6": sig_rom_m,
            "sigma_rom_papertable_A6": sig_rom_l,
        }
    )
    path = Path(__file__).resolve().parents[1] / "results/nb_ti_w_partial_molar_volumes.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    print(f"   saved -> {path}")


def main():
    surf, model = fit_surface("Nb-Ti-W")
    els = surf.elements
    check_surface_reproduction(surf, model)
    check_autograd(model, els)
    check_dilute_corner(model, els)
    sweep(model, els)


if __name__ == "__main__":
    main()
