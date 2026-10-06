"""Misfit volumes at any composition in the Mo-Nb-Ta-Ti-W space.

The general entry point onto the pooled MLIP volume surface. Given any
composition it returns the element-resolved misfit volume under three
treatments, so the GP's learned correction can be set against both the baseline
it was initialised from and the physically consistent target:

    dV_i^paper = V_i^Table5 - sum_j c_j V_j^Table5   the prior's own baseline
    dV_i^ROM   = V_i^MLIP   - sum_j c_j V_j^MLIP     linear mixing, MLIP anchors
    dV_i^PMV   = v_i - Vbar                          partial molar, MLIP surface

Separating the last two matters. dV^paper differs from dV^ROM only through the
elemental constants (the beta-Ti entry above all), while dV^ROM differs from
dV^PMV only through the non-linearity of the volume surface. Comparing the GP
against dV^paper alone would confound the two.

Usage
-----
    python predict_misfit.py --comp "Nb49.1 Ti16.1 W34.6"
    python predict_misfit.py --comp "Mo20.4 Nb21.7 Ti40.0 Ta17.4" --comp "W28.2 Ta70.6"
    python predict_misfit.py --csv alloys.csv --column composition -o out.csv
    python predict_misfit.py --grid Nb-Ti-W --step 2 -o nbtiw_dense.csv
    python predict_misfit.py --grid all --step 5 -o dense_reference.csv

Compositions may be given as "Nb49.1 Ti16.1 W34.6" or "Nb49.1Ti16.1W34.6"; they
are renormalised, so XRF totals that fall short of 100 are handled.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

from volume_surface import (
    BASIS,
    PooledSurface,
    RedlichKister,
    VolumeModel,
    parse_composition,
    sigma,
)

ORDERS = (2, 3, 4)
BEST_ORDER = 4

# Table 5 of the manuscript, a_i in Angstrom -> V_i = a_i^3 / 2.
PAPER_LATTICE = dict(Mo=3.147, Nb=3.300, Ta=3.301, Ti=3.320, W=3.165)
PAPER_VOLUME = np.array([PAPER_LATTICE[el] ** 3 / 2 for el in BASIS])


def build(order: int = BEST_ORDER):
    surf = PooledSurface.load()
    fits = {o: RedlichKister(surf.basis, surf.pure, o, True).fit(surf.c, surf.v)
            for o in sorted(set(ORDERS) | {order})}
    return surf, fits, VolumeModel(surf, fits[order])


def evaluate(surf, fits, model, c: np.ndarray, order: int = BEST_ORDER) -> list[dict]:
    """One row per present element."""
    c2 = c[None, :]
    V, vi = fits[order].partial_molar_volumes(c2)
    Vbar_pmv, dv_pmv = float(V[0]), vi[0] - V[0]

    stack = []
    for o in ORDERS:
        Vo, vio = fits[o].partial_molar_volumes(c2)
        stack.append(vio[0] - Vo[0])
    unc = np.array(stack).max(axis=0) - np.array(stack).min(axis=0)

    vbar_rom = float(surf.vegard(c2)[0])
    dv_rom = surf.pure - vbar_rom
    vbar_paper = float(c @ PAPER_VOLUME)
    dv_paper = PAPER_VOLUME - vbar_paper

    sup = model.support(c)
    label = "".join(f"{el}{100 * x:.4g}" for el, x in zip(BASIS, c) if x > 1e-6)

    rows = []
    for i, el in enumerate(BASIS):
        if c[i] <= 1e-6:
            continue
        rows.append(dict(
            composition=label,
            element=el,
            x=c[i],
            Vbar_paper=vbar_paper,
            Vbar_rom=vbar_rom,
            Vbar_pmv=Vbar_pmv,
            bowing_pct=100 * (Vbar_pmv - vbar_rom) / vbar_rom,
            v_i_pmv=Vbar_pmv + dv_pmv[i],
            dV_paper=dv_paper[i],
            dV_rom=dv_rom[i],
            dV_pmv=dv_pmv[i],
            dV_pmv_unc=unc[i],
            sigma_paper=float((c * dv_paper**2).sum()),
            sigma_rom=float((c * dv_rom**2).sum()),
            sigma_pmv=float(sigma(c2, vi, V)[0]),
            supported=sup.ok,
            support_note=sup.describe(),
        ))
    return rows


def simplex_grid(elements: list[str], step_pct: float) -> list[np.ndarray]:
    """All compositions on `elements` at `step_pct` at.% spacing, corners included."""
    k = len(elements)
    n = int(round(100 / step_pct))
    out = []
    for counts in itertools.product(range(n + 1), repeat=k - 1):
        rest = n - sum(counts)
        if rest < 0:
            continue
        full = list(counts) + [rest]
        c = np.zeros(len(BASIS))
        for el, m in zip(elements, full):
            c[BASIS.index(el)] = m / n
        out.append(c)
    return out


SUBSYSTEMS = {
    "Nb-Ti-W": ["Nb", "Ti", "W"],
    "Mo-Ti-Ta": ["Mo", "Ti", "Ta"],
    "Nb-Ta-W": ["Nb", "Ta", "W"],
    "Mo-Nb-Ta": ["Mo", "Nb", "Ta"],
    "Mo-Nb-Ti": ["Mo", "Nb", "Ti"],
    "Nb-Ta-Ti": ["Nb", "Ta", "Ti"],
    "Mo-Nb-Ta-Ti": ["Mo", "Nb", "Ta", "Ti"],
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comp", action="append", default=[],
                   help="composition string, repeatable")
    p.add_argument("--csv", help="read compositions from a CSV")
    p.add_argument("--column", default="composition", help="composition column in --csv")
    p.add_argument("--grid", help=f"dense grid: 'all' or one of {sorted(SUBSYSTEMS)}")
    p.add_argument("--step", type=float, default=5.0, help="grid spacing in at.%% (default 5)")
    p.add_argument("--order", type=int, default=BEST_ORDER, help="RK order (default 4)")
    p.add_argument("-o", "--out", help="write CSV here")
    p.add_argument("--wide", action="store_true",
                   help="one row per composition instead of per alloy-element")
    a = p.parse_args(argv)

    surf, fits, model = build(a.order)

    comps: list[np.ndarray] = []
    if a.grid:
        names = sorted(SUBSYSTEMS) if a.grid == "all" else [a.grid]
        for nm in names:
            if nm not in SUBSYSTEMS:
                p.error(f"unknown subsystem {nm!r}; choose from {sorted(SUBSYSTEMS)} or 'all'")
            comps += simplex_grid(SUBSYSTEMS[nm], a.step)
        # dedupe shared edges between subsystems
        seen, uniq = set(), []
        for c in comps:
            k = tuple(np.round(c, 8))
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        comps = uniq
    if a.csv:
        df = pd.read_csv(a.csv)
        if a.column not in df.columns:
            p.error(f"column {a.column!r} not in {a.csv}; have {list(df.columns)}")
        comps += [parse_composition(s) for s in df[a.column]]
    for s in a.comp:
        comps.append(parse_composition(s))

    if not comps:
        p.error("give at least one of --comp, --csv or --grid")

    rows: list[dict] = []
    for c in comps:
        rows += evaluate(surf, fits, model, c, a.order)
    out = pd.DataFrame(rows)

    if a.wide:
        idx = ["composition", "Vbar_paper", "Vbar_rom", "Vbar_pmv", "bowing_pct",
               "sigma_paper", "sigma_rom", "sigma_pmv", "supported"]
        wide = out.pivot_table(index=idx, columns="element",
                               values=["dV_rom", "dV_pmv", "dV_pmv_unc"]).reset_index()
        wide.columns = [c if isinstance(c, str) else f"{c[0]}_{c[1]}" for c in wide.columns]
        out = wide

    if a.out:
        out.to_csv(a.out, index=False)
        n_unsup = int((~out["supported"]).sum()) if "supported" in out else 0
        print(f"wrote {a.out}: {len(out)} rows, {out['composition'].nunique()} compositions"
              + (f", {n_unsup} rows EXTRAPOLATING" if n_unsup else ""))
    else:
        pd.set_option("display.width", 220)
        cols = [c for c in ["composition", "element", "x", "dV_paper", "dV_rom",
                            "dV_pmv", "dV_pmv_unc", "bowing_pct", "sigma_rom",
                            "sigma_pmv", "supported"] if c in out.columns]
        print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if "supported" in out and not out["supported"].all():
            print("\nWARNING: some compositions extrapolate beyond the cached surfaces:")
            for note in sorted(out.loc[~out["supported"], "support_note"].unique()):
                print(f"  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
