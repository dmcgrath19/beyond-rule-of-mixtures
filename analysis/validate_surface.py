"""Public analysis of cached MLIP volumes and published validation compositions."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pool_diagnostic import ALLOYS_14, to_basis
from volume_surface import (
    BASIS,
    PooledSurface,
    RedlichKister,
    VolumeModel,
    as_vector,
    sigma,
)

pd.set_option("display.width", 200)
FMT = lambda x: f"{x:.5f}"  # noqa: E731


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
def select_order(surf: PooledSurface, max_order: int = 4, n_folds: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(surf)), n_folds)
    rows = []
    for order in range(max_order + 1):
        for ternary in (False, True):
            errs = []
            for te in folds:
                tr = np.setdiff1d(np.arange(len(surf)), te)
                m = RedlichKister(surf.basis, surf.pure, order, ternary)
                m.fit(surf.c[tr], surf.v[tr])
                errs.append(m.predict(surf.c[te]) - surf.v[te])
            e = np.concatenate(errs)
            full = RedlichKister(surf.basis, surf.pure, order, ternary)
            full.fit(surf.c, surf.v)
            rows.append(
                dict(
                    order=order,
                    ternary=ternary,
                    n_terms=full.n_terms,
                    n_dead=len(full.unconstrained),
                    cv_rmse=float(np.sqrt((e**2).mean())),
                    cv_max=float(np.abs(e).max()),
                    fit_rmse=float(np.sqrt(((full.predict(surf.c) - surf.v) ** 2).mean())),
                )
            )
    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)


def finite_difference_pmv(model: RedlichKister, c: np.ndarray, h: float = 1e-6):
    """v_i by central differences, for comparison with autograd."""
    c = np.atleast_2d(c)
    V = model.predict(c)
    g = np.empty_like(c)
    for i in range(c.shape[1]):
        cp, cm = c.copy(), c.copy()
        cp[:, i] += h
        cm[:, i] -= h
        g[:, i] = (model.predict(cp) - model.predict(cm)) / (2 * h)
    return V, V[:, None] + g - (c * g).sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
def main() -> None:
    surf = PooledSurface.load()

    rule("POOLED SURFACE")
    print(f"basis        : {'-'.join(surf.basis)}")
    print(f"unique points: {len(surf)}")
    print("by source    : " + ", ".join(
        f"{k} {v}" for k, v in pd.Series(surf.source).value_counts().sort_index().items()
    ))
    print("pure volumes : " + ", ".join(
        f"{el} {V:.4f}" for el, V in zip(surf.basis, surf.pure)
    ))

    # ---------------- 1. order selection ----------------
    rule("1. REDLICH-KISTER ORDER SELECTION (5-fold CV over pooled points)")
    table = select_order(surf)
    print(table.to_string(index=False, float_format=FMT))
    best = table.iloc[0]
    order, ternary = int(best["order"]), bool(best["ternary"])
    print(f"\nchosen: order={order} ternary={ternary}")

    rk = RedlichKister(surf.basis, surf.pure, order, ternary).fit(surf.c, surf.v)
    pred = rk.predict(surf.c)
    resid = pred - surf.v
    print(
        f"fit RMSE {np.sqrt((resid**2).mean()) * 1e3:.3f} x10^-3 A^3/atom "
        f"({100 * np.sqrt((resid**2).mean()) / surf.v.mean():.4f}% of mean volume), "
        f"max |resid| {np.abs(resid).max() * 1e3:.3f} x10^-3"
    )
    if rk.unconstrained:
        print(f"\n{len(rk.unconstrained)} interaction terms have NO data and are pinned to zero:")
        for lab in rk.unconstrained:
            print(f"    {lab}")
        print("  -> any query exercising these is extrapolation; see support flags.")

    # ---------------- 2. corners ----------------
    rule("2. SIMPLEX CORNERS")
    eye = np.eye(len(surf.basis))
    err = rk.predict(eye) - surf.pure
    print(pd.DataFrame(dict(
        element=list(surf.basis), pure=surf.pure, predicted=rk.predict(eye), error=err
    )).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"max corner error {np.abs(err).max():.2e} A^3")

    # ---------------- 3. Gibbs-Duhem ----------------
    rule("3. GIBBS-DUHEM IDENTITY")
    V, vi = rk.partial_molar_volumes(surf.c)
    gd = np.abs((surf.c * vi).sum(axis=1) - V)
    print(f"max |sum_i c_i v_i - V| over {len(surf)} points = {gd.max():.3e} A^3")
    print("(exact by construction; a non-zero value would mean a bug in the derivative)")

    # ---------------- 4. autograd vs finite difference ----------------
    rule("4. AUTOGRAD vs CENTRAL FINITE DIFFERENCE")
    # No pooled point carries all five elements (the widest surface is
    # quaternary), so sample the multi-component points rather than requiring a
    # strictly interior simplex point.
    multi = (surf.c > 1e-6).sum(axis=1) >= 3
    sample = surf.c[multi][:: max(1, int(multi.sum()) // 40)]
    _, vi_ad = rk.partial_molar_volumes(sample)
    _, vi_fd = finite_difference_pmv(rk, sample)
    d = np.abs(vi_ad - vi_fd)
    print(f"checked {len(sample)} compositions, max |v_i(autograd) - v_i(FD)| = {d.max():.3e} A^3")

    # ---------------- 4b. sensitivity of the DERIVATIVES to RK order ----------------
    rule("4b. ARE THE MISFIT VOLUMES STABLE ACROSS RK ORDER?")
    print("CV selects the order on fitted volumes, but we consume derivatives.")
    print("Differentiating a high-order polynomial amplifies noise, so the")
    print("spread of dV_i and Sigma across orders is the honest uncertainty.\n")
    orders = [2, 3, 4]
    fits = {}
    for o in orders:
        m = RedlichKister(surf.basis, surf.pure, o, True).fit(surf.c, surf.v)
        fits[o] = m
    rows = []
    for label, frac in ALLOYS_14.items():
        c = to_basis(frac)[None, :]
        active = [i for i, x in enumerate(c[0]) if x > 1e-6]
        dvs, sigs = [], []
        for o in orders:
            V, vi = fits[o].partial_molar_volumes(c)
            dvs.append((vi[0] - V[0])[active])
            sigs.append(float(sigma(c, vi, V)[0]))
        dvs = np.array(dvs)
        rows.append(dict(
            alloy=label,
            **{f"sigma_o{o}": s for o, s in zip(orders, sigs)},
            sigma_spread_pct=100 * (max(sigs) - min(sigs)) / np.mean(sigs),
            max_dV_spread=float((dvs.max(axis=0) - dvs.min(axis=0)).max()),
        ))
    osens = pd.DataFrame(rows)
    print(osens.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"max Sigma spread across orders : {osens['sigma_spread_pct'].max():.1f}%")
    print(f"max dV_i spread across orders  : {osens['max_dV_spread'].max():.4f} A^3/atom")

    # ---------------- 4c. derivatives against the raw grid ----------------
    rule("4c. RK DERIVATIVES vs FINITE DIFFERENCES ON THE RAW MLIP GRID")
    print("The strongest available derivative test. Along a binary edge the")
    print("partial molar volumes follow from a single slope,")
    print("    dV_A = -x_B dV/dx_B,   dV_B = (1-x_B) dV/dx_B,")
    print("so central-differencing the raw 10 at.% points gives a model-free")
    print("reference for exactly the quantity the RK expansion supplies.\n")

    rows = []
    npts = (surf.c > 1e-6).sum(axis=1)
    for a, b in [(i, j) for i in range(len(BASIS)) for j in range(i + 1, len(BASIS))]:
        on_edge = (npts == 2) & (surf.c[:, a] > 1e-6) & (surf.c[:, b] > 1e-6)
        if on_edge.sum() < 5:
            continue
        xb = surf.c[on_edge, b]
        vv = surf.v[on_edge]
        o = np.argsort(xb)
        xb, vv = xb[o], vv[o]

        # central differences at interior grid nodes
        slope_fd = (vv[2:] - vv[:-2]) / (xb[2:] - xb[:-2])
        xmid = xb[1:-1]
        cmid = np.zeros((len(xmid), len(BASIS)))
        cmid[:, a] = 1 - xmid
        cmid[:, b] = xmid

        for o_rk in orders:
            V, vi = fits[o_rk].partial_molar_volumes(cmid)
            dv_b_rk = vi[:, b] - V
            dv_b_fd = (1 - xmid) * slope_fd
            err = dv_b_rk - dv_b_fd
            rows.append(dict(
                edge=f"{BASIS[a]}-{BASIS[b]}",
                n_nodes=len(xmid),
                order=o_rk,
                rmse_dV=float(np.sqrt((err**2).mean())),
                max_dV=float(np.abs(err).max()),
            ))
    fd = pd.DataFrame(rows)
    piv = fd.pivot_table(index=["edge", "n_nodes"], columns="order",
                         values="rmse_dV").reset_index()
    piv.columns = [f"rmse_o{c}" if isinstance(c, int) else c for c in piv.columns]
    print("RMSE of dV_i against raw-grid central differences (A^3/atom):")
    print(piv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    best_by_order = fd.groupby("order")["rmse_dV"].agg(["mean", "max"])
    print("aggregated over all edges:")
    print(best_by_order.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("NOTE: central differences on a 10 at.% grid carry their own O(h^2)")
    print("truncation error, so these numbers bound the RK error rather than")
    print("measuring it exactly. They are still decisive for ranking orders.")

    # ---------------- 5. leave-one-surface-out ----------------
    rule("5. LEAVE-ONE-SURFACE-OUT")
    print("fit on three surfaces, predict the held-out one. Points shared with a")
    print("retained surface are excluded so the test is genuinely out-of-sample.\n")
    rows = []
    for name in sorted(set(surf.source)):
        te = surf.source == name
        tr = ~te
        # drop held-out points whose composition also appears in the training set
        keep = np.array([
            not (np.abs(surf.c[tr] - ci).sum(axis=1) < 1e-6).any() for ci in surf.c[te]
        ])
        idx = np.where(te)[0][keep]
        if len(idx) == 0:
            rows.append(dict(held_out=name, n_test=0, rmse=np.nan, max_err=np.nan))
            continue
        m = RedlichKister(surf.basis, surf.pure, order, ternary).fit(surf.c[tr], surf.v[tr])
        e = m.predict(surf.c[idx]) - surf.v[idx]
        rows.append(dict(
            held_out=name,
            n_test=len(idx),
            rmse=float(np.sqrt((e**2).mean())),
            max_err=float(np.abs(e).max()),
            bias=float(e.mean()),
        ))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------- 6. pooled vs per-subsystem at the 14 alloys ----------------
    rule("6. POOLED QUINARY FIT vs PER-SUBSYSTEM FIT AT THE 14 ALLOYS")
    print("For each alloy, refit RK using only the points that lie inside that")
    print("alloy's own element set, then compare misfit volumes and Sigma.\n")

    rows = []
    for label, frac in ALLOYS_14.items():
        c = to_basis(frac)
        active = [i for i, x in enumerate(c) if x > 1e-6]
        els = tuple(BASIS[i] for i in active)

        # pooled
        Vp, vip = rk.partial_molar_volumes(c[None, :])
        dv_pooled = (vip[0] - Vp[0])[active]
        sig_pooled = float(sigma(c[None, :], vip, Vp)[0])

        # subsystem-only: points using no element outside `els`
        mask = np.ones(len(surf), dtype=bool)
        for i in range(len(BASIS)):
            if i not in active:
                mask &= surf.c[:, i] < 1e-6
        sub_c = surf.c[mask][:, active]
        sub_v = surf.v[mask]
        sub_pure = surf.pure[active]
        m = RedlichKister(els, sub_pure, order, ternary and len(els) >= 3)
        m.fit(sub_c, sub_v)
        cs = c[active][None, :]
        Vs, vis = m.partial_molar_volumes(cs)
        dv_sub = vis[0] - Vs[0]
        sig_sub = float(sigma(cs, vis, Vs)[0])

        rows.append(dict(
            alloy=label,
            system="".join(els),
            n_sub=int(mask.sum()),
            Vbar_pooled=float(Vp[0]),
            Vbar_sub=float(Vs[0]),
            dVbar=float(Vp[0] - Vs[0]),
            sigma_pooled=sig_pooled,
            sigma_sub=sig_sub,
            sigma_ratio=sig_pooled / sig_sub if sig_sub else np.nan,
            max_dV_diff=float(np.abs(dv_pooled - dv_sub).max()),
        ))
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"max |Vbar_pooled - Vbar_sub|      : {cmp['dVbar'].abs().max():.4f} A^3/atom")
    print(f"max |dV_i difference| any element : {cmp['max_dV_diff'].max():.4f} A^3/atom")
    print(f"Sigma ratio range                 : "
          f"{cmp['sigma_ratio'].min():.3f} .. {cmp['sigma_ratio'].max():.3f}")

    # ---------------- summary ----------------
    rule("SUMMARY")
    vm = VolumeModel(surf, rk)
    res = vm.evaluate_many(ALLOYS_14)
    n_ok = int(res["supported"].sum())
    print(f"all {len(res)} experimental alloys evaluated; {n_ok}/{len(res)} fully data-supported")
    if n_ok < len(res):
        print(res.loc[~res["supported"], ["alloy", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
