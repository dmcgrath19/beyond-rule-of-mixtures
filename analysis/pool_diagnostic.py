"""Public analysis of cached MLIP volumes and published validation compositions."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itertools import combinations

import numpy as np
import pandas as pd

from partial_molar_volumes import SURFACES, load_surface

BASIS = ["Mo", "Nb", "Ta", "Ti", "W"]

# XRF at.% from Table 2 of the manuscript. These do not sum to 100 (minor
# unreported elements); they are renormalised over the refractory basis.
ALLOYS_14 = {
    "W28.2Ta70.6": dict(W=28.2, Ta=70.6),
    "W67.0Ta31.8": dict(W=67.0, Ta=31.8),
    "Mo81.5Ta17.9": dict(Mo=81.5, Ta=17.9),
    "Mo55.2Ti44.8": dict(Mo=55.2, Ti=44.8),
    "Mo62.4Ti19.9Ta17.5": dict(Mo=62.4, Ti=19.9, Ta=17.5),
    "Mo41.8Ti40.7Ta17.3": dict(Mo=41.8, Ti=40.7, Ta=17.3),
    "Mo20.5Ti61.2Ta17.8": dict(Mo=20.5, Ti=61.2, Ta=17.8),
    "Nb49.1Ti16.1W34.6": dict(Nb=49.1, Ti=16.1, W=34.6),
    "Nb24.8Ti40.2W34.8": dict(Nb=24.8, Ti=40.2, W=34.8),
    "Nb12.6Ti52.9W34.4": dict(Nb=12.6, Ti=52.9, W=34.4),
    "Mo12.1Nb35.1Ta52.8": dict(Mo=12.1, Nb=35.1, Ta=52.8),
    "Mo20.4Nb21.7Ti40.0Ta17.4": dict(Mo=20.4, Nb=21.7, Ti=40.0, Ta=17.4),
    "Mo31.0Nb23.4Ti28.6Ta17.0": dict(Mo=31.0, Nb=23.4, Ti=28.6, Ta=17.0),
    "Mo42.5Nb13.2Ti35.7Ta8.6": dict(Mo=42.5, Nb=13.2, Ti=35.7, Ta=8.6),
}


def to_basis(frac: dict[str, float]) -> np.ndarray:
    c = np.array([frac.get(el, 0.0) for el in BASIS], dtype=float)
    return c / c.sum()


def main() -> None:
    surfaces = {name: load_surface(name) for name in SURFACES}

    # ---------------- 1. pure-element consistency ----------------
    print("=" * 78)
    print("1. PURE-ELEMENT VOLUMES ACROSS SURFACES (A^3/atom)")
    print("=" * 78)
    pure_by_el: dict[str, dict[str, float]] = {el: {} for el in BASIS}
    for name, surf in surfaces.items():
        for el, V in zip(surf.elements, surf.pure_volumes()):
            pure_by_el[el][name] = V

    rows = []
    for el in BASIS:
        vals = pure_by_el[el]
        if not vals:
            rows.append(dict(element=el, n_surfaces=0, mean=np.nan, spread=np.nan))
            continue
        v = np.array(list(vals.values()))
        rows.append(
            dict(
                element=el,
                n_surfaces=len(v),
                mean=v.mean(),
                spread=v.max() - v.min(),
                spread_pct=100 * (v.max() - v.min()) / v.mean(),
                **{n: f"{x:.4f}" for n, x in vals.items()},
            )
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------- 2. shared-edge consistency ----------------
    print()
    print("=" * 78)
    print("2. SHARED BINARY EDGES: do two surfaces agree where they overlap?")
    print("=" * 78)
    # collect every point as (frozenset of present elements, basis vector, V, surface)
    pooled = []
    for name, surf in surfaces.items():
        for ci, vi in zip(surf.c, surf.v):
            full = np.zeros(len(BASIS))
            for el, x in zip(surf.elements, ci):
                full[BASIS.index(el)] = x
            pooled.append((name, full, vi))

    tol = 1e-6
    seen: dict[tuple, list[tuple[str, float]]] = {}
    for name, c, v in pooled:
        key = tuple(np.round(c, 4))
        seen.setdefault(key, []).append((name, v))

    dupes = {k: v for k, v in seen.items() if len({n for n, _ in v}) > 1}
    if not dupes:
        print("no compositions are shared between surfaces (grids do not coincide)")
    else:
        drows = []
        for key, entries in sorted(dupes.items()):
            vals = np.array([v for _, v in entries])
            comp = " ".join(
                f"{el}{100 * x:.0f}" for el, x in zip(BASIS, key) if x > tol
            )
            drows.append(
                dict(
                    composition=comp,
                    n=len(entries),
                    surfaces=",".join(sorted({n for n, _ in entries})),
                    V_mean=vals.mean(),
                    V_spread=vals.max() - vals.min(),
                    spread_pct=100 * (vals.max() - vals.min()) / vals.mean(),
                )
            )
        d = pd.DataFrame(drows).sort_values("V_spread", ascending=False)
        print(f"{len(d)} shared compositions")
        print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()
        print(
            f"worst disagreement {d['V_spread'].max():.4f} A^3/atom "
            f"({d['spread_pct'].max():.3f}%), median {d['V_spread'].median():.4f}"
        )

    # ---------------- 3. interaction coverage ----------------
    print()
    print("=" * 78)
    print("3. INTERACTION COVERAGE OVER THE Mo-Nb-Ta-Ti-W BASIS")
    print("=" * 78)
    C = np.vstack([c for _, c, _ in pooled])
    present = C > 1e-6

    print(f"pooled points: {len(C)}  (unique: {len(seen)})")
    print()
    print("binary pairs (n points where BOTH elements are present):")
    prows = []
    for i, j in combinations(range(len(BASIS)), 2):
        n = int((present[:, i] & present[:, j]).sum())
        prows.append(dict(pair=f"{BASIS[i]}-{BASIS[j]}", n_points=n,
                          status="OK" if n else "*** NO DATA ***"))
    print(pd.DataFrame(prows).to_string(index=False))

    print()
    print("ternary triples (n points where ALL THREE are present):")
    trows = []
    for i, j, k in combinations(range(len(BASIS)), 3):
        n = int((present[:, i] & present[:, j] & present[:, k]).sum())
        trows.append(dict(triple=f"{BASIS[i]}-{BASIS[j]}-{BASIS[k]}", n_points=n,
                          status="OK" if n else "*** NO DATA ***"))
    print(pd.DataFrame(trows).to_string(index=False))

    # ---------------- 4. do the 14 alloys need anything missing? ----------------
    print()
    print("=" * 78)
    print("4. THE 14 EXPERIMENTAL ALLOYS: are their interactions covered?")
    print("=" * 78)
    pair_n = {
        (i, j): int((present[:, i] & present[:, j]).sum())
        for i, j in combinations(range(len(BASIS)), 2)
    }
    triple_n = {
        (i, j, k): int((present[:, i] & present[:, j] & present[:, k]).sum())
        for i, j, k in combinations(range(len(BASIS)), 3)
    }

    arows = []
    for label, frac in ALLOYS_14.items():
        c = to_basis(frac)
        idx = [n for n, x in enumerate(c) if x > 1e-6]
        missing_pairs = [
            f"{BASIS[i]}-{BASIS[j]}"
            for i, j in combinations(idx, 2)
            if pair_n[(i, j)] == 0
        ]
        missing_triples = [
            f"{BASIS[i]}-{BASIS[j]}-{BASIS[k]}"
            for i, j, k in combinations(idx, 3)
            if triple_n[(i, j, k)] == 0
        ]
        # is this exact composition inside a single cached surface's element set?
        host = [
            n for n, s in surfaces.items()
            if set(BASIS[i] for i in idx) <= set(s.elements)
        ]
        arows.append(
            dict(
                alloy=label,
                elements="".join(BASIS[i] for i in idx),
                renorm_sum=sum(frac.values()),
                single_surface=host[0] if host else "-- none --",
                missing_pairs=",".join(missing_pairs) or "none",
                missing_triples=",".join(missing_triples) or "none",
            )
        )
    print(pd.DataFrame(arows).to_string(index=False))


if __name__ == "__main__":
    main()
