"""Redlich-Kister fits of the cached MLIP volume surfaces and the partial molar volumes they imply."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent / "data/simulation"

# Cached EGIP-inf surfaces: 3x3x3 BCC SQS (54 atoms), icet seed 42,
# hydrostatic pre-relax, 15-point isotropic scan, Birch-Murnaghan fit.
SURFACES = {
    "Nb-Ti-W": ("nb_ti_w_eos_ternary_with_bowing.csv", ["Nb", "Ti", "W"]),
    "Mo-Ti-Ta": ("mo_ti_ta_eos_ternary_with_bowing.csv", ["Mo", "Ti", "Ta"]),
    "Nb-Ta-W": ("nb_ta_w_eos_ternary.csv", ["Nb", "Ta", "W"]),
    "Nb-Ta-Ti-Mo": (
        "nb_ta_ti_mo_eos_quaternary_with_bowing.csv",
        ["Nb", "Ta", "Ti", "Mo"],
    ),
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+\.?\d*)")


def parse_composition(s: str, elements: list[str]) -> np.ndarray:
    """Atomic fractions from strings like 'W90.7407 Ti9.2593'.

    The x_*_realized columns are unreliable (the Nb-Ti-W file has them
    mis-assigned, e.g. pure W reported as x_Nb_realized = 1.0), so the realized
    composition string is the source of truth.
    """
    found = {el: float(v) for el, v in _TOKEN.findall(str(s))}
    unknown = set(found) - set(elements)
    if unknown:
        raise ValueError(f"composition {s!r} has elements outside {elements}: {unknown}")
    c = np.array([found.get(el, 0.0) for el in elements], dtype=float)
    total = c.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"composition {s!r} did not parse to a positive total")
    return c / total


@dataclass
class Surface:
    name: str
    elements: list[str]
    c: np.ndarray  # (n, k) realized atomic fractions
    v: np.ndarray  # (n,) MLIP equilibrium volume per atom, A^3

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def pure_volumes(self) -> np.ndarray:
        """Volume at each simplex corner, taken from the nearest grid point."""
        out = np.empty(self.n_elements)
        for i in range(self.n_elements):
            j = int(np.argmax(self.c[:, i]))
            if self.c[j, i] < 0.999:
                raise ValueError(
                    f"{self.name}: no pure-{self.elements[i]} point on the surface "
                    f"(max fraction {self.c[j, i]:.3f})"
                )
            out[i] = self.v[j]
        return out


def load_surface(name: str) -> Surface:
    fname, elements = SURFACES[name]
    df = pd.read_csv(REPO / fname)
    c = np.vstack([parse_composition(s, elements) for s in df["realized_composition"]])
    v = df["volume_per_atom_A3"].to_numpy(dtype=float)
    ok = np.isfinite(v)
    return Surface(name, elements, c[ok], v[ok])


class RedlichKister:
    """V(c) = sum_i c_i V_i + sum_{i<j} c_i c_j sum_k L_ij^k (c_i - c_j)^k [+ ternary]

    Pure-element volumes V_i are pinned, not fitted. Only the excess terms are
    least-squares fitted, so the expansion reproduces the corners exactly.
    """

    def __init__(self, elements: list[str], pure: np.ndarray, order: int, ternary: bool):
        self.elements = elements
        self.pure = np.asarray(pure, dtype=float)
        self.order = order
        self.ternary = ternary
        self.pairs = list(combinations(range(len(elements)), 2))
        self.triples = list(combinations(range(len(elements)), 3)) if ternary else []
        self.coef: np.ndarray | None = None

    def _basis(self, c: np.ndarray) -> np.ndarray:
        """Excess-term design matrix, shape (n, n_terms)."""
        cols = []
        for i, j in self.pairs:
            prod = c[:, i] * c[:, j]
            for k in range(self.order + 1):
                cols.append(prod * (c[:, i] - c[:, j]) ** k)
        for i, j, k in self.triples:
            prod = c[:, i] * c[:, j] * c[:, k]
            # one ternary interaction parameter per corner element
            cols.extend([prod * c[:, i], prod * c[:, j], prod * c[:, k]])
        return np.column_stack(cols) if cols else np.zeros((len(c), 0))

    @property
    def n_terms(self) -> int:
        return len(self.pairs) * (self.order + 1) + len(self.triples) * 3

    def fit(self, c: np.ndarray, v: np.ndarray) -> "RedlichKister":
        residual = v - c @ self.pure
        A = self._basis(c)
        self.coef, *_ = np.linalg.lstsq(A, residual, rcond=None)
        return self

    def predict(self, c: np.ndarray) -> np.ndarray:
        assert self.coef is not None, "call fit() first"
        return c @ self.pure + self._basis(c) @ self.coef

    # ---- torch mirror, used only to get exact derivatives via autograd ----
    def _predict_torch(self, c: torch.Tensor) -> torch.Tensor:
        pure = torch.as_tensor(self.pure, dtype=c.dtype)
        out = c @ pure
        coef = torch.as_tensor(self.coef, dtype=c.dtype)
        t = 0
        for i, j in self.pairs:
            prod = c[:, i] * c[:, j]
            for k in range(self.order + 1):
                out = out + coef[t] * prod * (c[:, i] - c[:, j]) ** k
                t += 1
        for i, j, k in self.triples:
            prod = c[:, i] * c[:, j] * c[:, k]
            for m in (i, j, k):
                out = out + coef[t] * prod * c[:, m]
                t += 1
        return out

    def partial_molar_volumes(self, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (V, v_i) with sum_i c_i v_i == V to machine precision.

        v_i = V + dV/dc_i - sum_j c_j dV/dc_j. This is invariant to how V is
        extended off the simplex, so differentiating the polynomial form as if
        the c_i were independent is legitimate.
        """
        ct = torch.tensor(c, dtype=torch.float64, requires_grad=True)
        V = self._predict_torch(ct)
        (grad,) = torch.autograd.grad(V.sum(), ct)
        g = grad.detach().numpy()
        Vn = V.detach().numpy()
        return Vn, Vn[:, None] + g - (c * g).sum(axis=1, keepdims=True)


def sigma_from_volumes(c: np.ndarray, vi: np.ndarray, vbar: np.ndarray) -> np.ndarray:
    """Reduced misfit parameter sum_i c_i (v_i - Vbar)^2, in A^6."""
    return (c * (vi - vbar[:, None]) ** 2).sum(axis=1)


def select_order(surf: Surface, max_order: int = 3, n_folds: int = 5, seed: int = 0):
    """Pick the RK order by k-fold CV on the surface points."""
    pure = surf.pure_volumes()
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(surf.v)), n_folds)
    rows = []
    for order in range(max_order + 1):
        for ternary in (False, True):
            if ternary and surf.n_elements < 3:
                continue
            errs = []
            for te in folds:
                tr = np.setdiff1d(np.arange(len(surf.v)), te)
                m = RedlichKister(surf.elements, pure, order, ternary).fit(
                    surf.c[tr], surf.v[tr]
                )
                errs.append(m.predict(surf.c[te]) - surf.v[te])
            e = np.concatenate(errs)
            full = RedlichKister(surf.elements, pure, order, ternary).fit(surf.c, surf.v)
            rows.append(
                dict(
                    order=order,
                    ternary=ternary,
                    n_terms=full.n_terms,
                    cv_rmse=float(np.sqrt((e**2).mean())),
                    fit_rmse=float(np.sqrt(((full.predict(surf.c) - surf.v) ** 2).mean())),
                )
            )
    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)


def main():
    for name in SURFACES:
        surf = load_surface(name)
        pure = surf.pure_volumes()
        print(f"\n{'=' * 74}\n{name}   n={len(surf.v)} points, {surf.n_elements} elements")
        print(
            "pure-element MLIP volumes (A^3/atom): "
            + ", ".join(f"{el} {V:.3f}" for el, V in zip(surf.elements, pure))
        )

        table = select_order(surf)
        print("\nRedlich-Kister order selection (5-fold CV over surface points):")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

        best = table.iloc[0]
        model = RedlichKister(
            surf.elements, pure, int(best["order"]), bool(best["ternary"])
        ).fit(surf.c, surf.v)
        V, vi = model.partial_molar_volumes(surf.c)

        gibbs_duhem = np.abs((surf.c * vi).sum(axis=1) - V).max()
        print(
            f"\nchosen: order={int(best['order'])} ternary={bool(best['ternary'])}  "
            f"fit RMSE {best['fit_rmse'] * 1000:.2f} x10^-3 A^3/atom "
            f"({100 * best['fit_rmse'] / surf.v.mean():.3f}% of mean volume)"
        )
        print(f"Gibbs-Duhem residual max|sum_i c_i v_i - V| = {gibbs_duhem:.2e} A^3")

        interior = surf.c.min(axis=1) > 0.05
        spread = (vi.max(axis=1) - vi.min(axis=1))[interior]
        print(
            f"partial molar volume spread over {interior.sum()} interior points: "
            f"median {np.median(spread):.3f}, max {spread.max():.3f} A^3/atom"
        )


if __name__ == "__main__":
    main()
