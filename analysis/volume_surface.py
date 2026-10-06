"""Unified Mo-Nb-Ta-Ti-W molar-volume surface and partial molar volumes.

The cached MLIP surfaces (three ternaries + one quaternary, 484 points, 398
unique) all use the same generation protocol -- EGIP-inf, 3x3x3 BCC SQS with
icet seed 42, hydrostatic pre-relax, 15-point isotropic scan, Birch-Murnaghan
fit -- and are bit-identical on the 83 compositions where two surfaces overlap.
They can therefore be pooled into a single volume surface over the quinary
Mo-Nb-Ta-Ti-W basis without any cross-calibration.

Pooling matters because it gives one globally consistent set of partial molar
volumes. Fitting each subsystem separately would give a different v_Ti in
Mo-Ti-Ta than in Nb-Ti-W, which is exactly the inconsistency the learned
effective volumes are being tested against.

    v_i = V + dV/dc_i - sum_j c_j dV/dc_j,    sum_i c_i v_i = V   (Gibbs-Duhem)

so Sigma_pmv = sum_i c_i (v_i - V)^2 is a properly centred variance with no
convention left to choose, and dV_i = v_i - V is directly comparable to the
learned effective-volume misfit dV_i^eff of the GP.

Coverage is not uniform. Mo-W is absent from every cached surface, as are the
Mo-Nb-W, Mo-Ta-W, Mo-Ti-W and Ta-Ti-W ternaries. Those interaction terms have
identically-zero design columns, so the fit silently assigns them zero excess
volume. Every query is therefore screened and returned with an explicit support
flag rather than being allowed to extrapolate unannounced.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent / "data/simulation"

BASIS = ("Mo", "Nb", "Ta", "Ti", "W")

# Cached EGIP-inf surfaces. Identical protocol across all four files.
SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Nb-Ti-W": ("nb_ti_w_eos_ternary_with_bowing.csv", ("Nb", "Ti", "W")),
    "Mo-Ti-Ta": ("mo_ti_ta_eos_ternary_with_bowing.csv", ("Mo", "Ti", "Ta")),
    "Nb-Ta-W": ("nb_ta_w_eos_ternary.csv", ("Nb", "Ta", "W")),
    "Nb-Ta-Ti-Mo": (
        "nb_ta_ti_mo_eos_quaternary_with_bowing.csv",
        ("Nb", "Ta", "Ti", "Mo"),
    ),
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+\.?\d*)")
_ZERO = 1e-9


def parse_composition(s: str, basis: tuple[str, ...] = BASIS) -> np.ndarray:
    """Atomic fractions on `basis` from strings like 'W90.7407 Ti9.2593'.

    The x_*_realized columns in the cached CSVs are unreliable (the Nb-Ti-W file
    has them mis-assigned, e.g. pure W reported as x_Nb_realized = 1.0), so the
    realized composition string is the source of truth.
    """
    found = {el: float(v) for el, v in _TOKEN.findall(str(s))}
    unknown = set(found) - set(basis)
    if unknown:
        raise ValueError(f"composition {s!r} has elements outside {basis}: {sorted(unknown)}")
    c = np.array([found.get(el, 0.0) for el in basis], dtype=float)
    total = c.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"composition {s!r} did not parse to a positive total")
    return c / total


def as_vector(comp: dict[str, float], basis: tuple[str, ...] = BASIS) -> np.ndarray:
    """Renormalised fraction vector from a {element: amount} mapping.

    Accepts at.% or fractions; XRF totals that fall short of 100 are
    renormalised over the refractory basis.
    """
    unknown = {k for k, v in comp.items() if k not in basis and v > 0}
    if unknown:
        raise ValueError(f"composition has elements outside {basis}: {sorted(unknown)}")
    c = np.array([float(comp.get(el, 0.0)) for el in basis])
    if (c < 0).any():
        raise ValueError(f"negative fraction in {comp}")
    total = c.sum()
    if total <= 0:
        raise ValueError(f"composition {comp} sums to zero")
    return c / total


# ---------------------------------------------------------------------------
# pooled data
# ---------------------------------------------------------------------------


@dataclass
class PooledSurface:
    basis: tuple[str, ...]
    c: np.ndarray  # (n, k) atomic fractions
    v: np.ndarray  # (n,) MLIP equilibrium volume per atom, A^3
    source: np.ndarray  # (n,) originating surface name
    pure: np.ndarray = field(init=False)  # (k,) pure-element volumes

    def __post_init__(self) -> None:
        self.pure = self._pure_volumes()

    def _pure_volumes(self) -> np.ndarray:
        out = np.full(len(self.basis), np.nan)
        for i, el in enumerate(self.basis):
            hit = self.c[:, i] > 0.999
            if not hit.any():
                raise ValueError(f"no pure-{el} point in the pooled data")
            vals = self.v[hit]
            if vals.max() - vals.min() > 1e-6:
                raise ValueError(
                    f"pure-{el} volume disagrees between surfaces: "
                    f"{vals.min():.6f}..{vals.max():.6f} A^3"
                )
            out[i] = vals.mean()
        return out

    @classmethod
    def load(cls, basis: tuple[str, ...] = BASIS, dedupe: bool = True) -> "PooledSurface":
        cs, vs, src = [], [], []
        for name, (fname, elements) in SOURCES.items():
            df = pd.read_csv(REPO / fname)
            v = df["volume_per_atom_A3"].to_numpy(dtype=float)
            ok = np.isfinite(v)
            for s, vi in zip(df["realized_composition"][ok], v[ok]):
                sub = parse_composition(s, elements)
                full = np.zeros(len(basis))
                for el, x in zip(elements, sub):
                    full[basis.index(el)] = x
                cs.append(full)
                vs.append(vi)
                src.append(name)
        c = np.vstack(cs)
        v = np.array(vs)
        s = np.array(src)

        if dedupe:
            # Overlapping surfaces repeat 83 compositions with identical values;
            # keeping duplicates would silently up-weight shared edges and faces.
            _, keep = np.unique(np.round(c, 6), axis=0, return_index=True)
            keep.sort()
            spread = _duplicate_spread(c, v)
            if spread > 1e-6:
                raise ValueError(
                    f"duplicated compositions disagree by up to {spread:.2e} A^3; "
                    "surfaces are not mutually consistent and must not be pooled"
                )
            c, v, s = c[keep], v[keep], s[keep]

        return cls(basis=basis, c=c, v=v, source=s)

    def __len__(self) -> int:
        return len(self.v)

    def vegard(self, c: np.ndarray) -> np.ndarray:
        return c @ self.pure


def _duplicate_spread(c: np.ndarray, v: np.ndarray) -> float:
    keys: dict[tuple, list[float]] = {}
    for ci, vi in zip(np.round(c, 6), v):
        keys.setdefault(tuple(ci), []).append(vi)
    return max((max(g) - min(g) for g in keys.values() if len(g) > 1), default=0.0)


# ---------------------------------------------------------------------------
# Redlich-Kister model
# ---------------------------------------------------------------------------


class RedlichKister:
    """V(c) = sum_i c_i V_i + sum_{i<j} c_i c_j sum_k L_ij^k (c_i-c_j)^k [+ ternary]

    Pure-element volumes are pinned, not fitted, so the expansion reproduces the
    simplex corners exactly and only has to describe the excess volume.

    Terms whose design column is identically zero over the training data (an
    interaction no cached surface exercises) are recorded in `unconstrained` and
    forced to zero rather than being left to the least-squares minimum-norm
    solution by accident.
    """

    def __init__(
        self,
        basis: tuple[str, ...],
        pure: np.ndarray,
        order: int = 2,
        ternary: bool = True,
    ) -> None:
        self.basis = tuple(basis)
        self.pure = np.asarray(pure, dtype=float)
        self.order = int(order)
        self.ternary = bool(ternary)
        self.pairs = list(combinations(range(len(basis)), 2))
        self.triples = list(combinations(range(len(basis)), 3)) if ternary else []
        self.coef: np.ndarray | None = None
        self.unconstrained: list[str] = []

    # -- term bookkeeping ---------------------------------------------------
    def term_labels(self) -> list[str]:
        out = []
        for i, j in self.pairs:
            for k in range(self.order + 1):
                out.append(f"L[{self.basis[i]}-{self.basis[j]}]^{k}")
        for i, j, k in self.triples:
            tri = f"{self.basis[i]}-{self.basis[j]}-{self.basis[k]}"
            for m in (i, j, k):
                out.append(f"M[{tri}]:{self.basis[m]}")
        return out

    @property
    def n_terms(self) -> int:
        return len(self.pairs) * (self.order + 1) + len(self.triples) * 3

    def _basis_matrix(self, c: np.ndarray) -> np.ndarray:
        cols = []
        for i, j in self.pairs:
            prod = c[:, i] * c[:, j]
            for k in range(self.order + 1):
                cols.append(prod * (c[:, i] - c[:, j]) ** k)
        for i, j, k in self.triples:
            prod = c[:, i] * c[:, j] * c[:, k]
            cols.extend([prod * c[:, i], prod * c[:, j], prod * c[:, k]])
        return np.column_stack(cols) if cols else np.zeros((len(c), 0))

    # -- fitting ------------------------------------------------------------
    def fit(self, c: np.ndarray, v: np.ndarray) -> "RedlichKister":
        A = self._basis_matrix(c)
        residual = v - c @ self.pure

        dead = np.abs(A).max(axis=0) < _ZERO
        labels = self.term_labels()
        self.unconstrained = [lab for lab, d in zip(labels, dead) if d]

        coef = np.zeros(A.shape[1])
        if (~dead).any():
            sol, *_ = np.linalg.lstsq(A[:, ~dead], residual, rcond=None)
            coef[~dead] = sol
        self.coef = coef
        return self

    def predict(self, c: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("call fit() first")
        return c @ self.pure + self._basis_matrix(c) @ self.coef

    # -- exact derivatives via autograd -------------------------------------
    def _predict_torch(self, c: torch.Tensor) -> torch.Tensor:
        pure = torch.as_tensor(self.pure, dtype=c.dtype)
        coef = torch.as_tensor(self.coef, dtype=c.dtype)
        out = c @ pure
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
        """Return (V, v_i) satisfying sum_i c_i v_i == V to machine precision.

        v_i = V + dV/dc_i - sum_j c_j dV/dc_j is invariant to how V is extended
        off the simplex, so differentiating the polynomial as if the c_i were
        independent is legitimate.
        """
        c = np.atleast_2d(np.asarray(c, dtype=float))
        ct = torch.tensor(c, dtype=torch.float64, requires_grad=True)
        V = self._predict_torch(ct)
        (grad,) = torch.autograd.grad(V.sum(), ct)
        g = grad.detach().numpy()
        Vn = V.detach().numpy()
        return Vn, Vn[:, None] + g - (c * g).sum(axis=1, keepdims=True)


def sigma(c: np.ndarray, vi: np.ndarray, vbar: np.ndarray) -> np.ndarray:
    """Reduced misfit parameter sum_i c_i (v_i - Vbar)^2, in A^6."""
    return (c * (vi - np.asarray(vbar)[:, None]) ** 2).sum(axis=1)


# ---------------------------------------------------------------------------
# support screening
# ---------------------------------------------------------------------------


@dataclass
class Support:
    """Whether a query composition is backed by data."""

    missing_pairs: list[str]
    missing_triples: list[str]
    nearest_distance: float  # L1 distance to closest training composition

    @property
    def ok(self) -> bool:
        return not self.missing_pairs and not self.missing_triples

    def describe(self) -> str:
        if self.ok:
            return f"supported (nearest datum {self.nearest_distance:.3f} L1)"
        bits = []
        if self.missing_pairs:
            bits.append("pairs " + ",".join(self.missing_pairs))
        if self.missing_triples:
            bits.append("triples " + ",".join(self.missing_triples))
        return "EXTRAPOLATING: no data for " + "; ".join(bits)


class VolumeModel:
    """Pooled surface + fitted RK expansion + support screening."""

    def __init__(self, surface: PooledSurface, model: RedlichKister) -> None:
        self.surface = surface
        self.model = model
        present = surface.c > 1e-6
        self._pair_n = {
            (i, j): int((present[:, i] & present[:, j]).sum())
            for i, j in combinations(range(len(surface.basis)), 2)
        }
        self._triple_n = {
            (i, j, k): int((present[:, i] & present[:, j] & present[:, k]).sum())
            for i, j, k in combinations(range(len(surface.basis)), 3)
        }

    @classmethod
    def build(cls, order: int = 2, ternary: bool = True) -> "VolumeModel":
        surf = PooledSurface.load()
        rk = RedlichKister(surf.basis, surf.pure, order, ternary).fit(surf.c, surf.v)
        return cls(surf, rk)

    def support(self, c: np.ndarray) -> Support:
        b = self.surface.basis
        idx = [i for i, x in enumerate(c) if x > 1e-6]
        mp = [
            f"{b[i]}-{b[j]}"
            for i, j in combinations(idx, 2)
            if self._pair_n[(i, j)] == 0
        ]
        mt = [
            f"{b[i]}-{b[j]}-{b[k]}"
            for i, j, k in combinations(idx, 3)
            if self._triple_n[(i, j, k)] == 0
        ]
        d = float(np.abs(self.surface.c - c).sum(axis=1).min())
        return Support(mp, mt, d)

    def evaluate(self, comp: dict[str, float] | np.ndarray) -> dict:
        """Misfit volumes and misfit parameter at one composition.

        Returns both the partial-molar and rule-of-mixtures treatments so the
        two can be compared directly against the GP's learned correction.
        """
        c = comp if isinstance(comp, np.ndarray) else as_vector(comp, self.surface.basis)
        c2 = c[None, :]
        V, vi = self.model.partial_molar_volumes(c2)
        vbar_rom = self.surface.vegard(c2)
        dv_rom = self.surface.pure - vbar_rom[0]

        return dict(
            basis=self.surface.basis,
            c=c,
            Vbar_pmv=float(V[0]),
            Vbar_rom=float(vbar_rom[0]),
            bowing_pct=float(100 * (V[0] - vbar_rom[0]) / vbar_rom[0]),
            v_i=vi[0],
            dV_pmv=vi[0] - V[0],
            dV_rom=dv_rom,
            sigma_pmv=float(sigma(c2, vi, V)[0]),
            sigma_rom=float((c * dv_rom**2).sum()),
            gibbs_duhem=float(abs((c * vi[0]).sum() - V[0])),
            support=self.support(c),
        )

    def evaluate_many(self, comps: dict[str, dict[str, float]]) -> pd.DataFrame:
        rows = []
        for label, comp in comps.items():
            r = self.evaluate(comp)
            row = dict(
                alloy=label,
                Vbar_rom=r["Vbar_rom"],
                Vbar_pmv=r["Vbar_pmv"],
                bowing_pct=r["bowing_pct"],
                sigma_rom=r["sigma_rom"],
                sigma_pmv=r["sigma_pmv"],
                sigma_ratio=r["sigma_pmv"] / r["sigma_rom"] if r["sigma_rom"] else np.nan,
                supported=r["support"].ok,
                note=r["support"].describe(),
            )
            for el, dp, dr in zip(r["basis"], r["dV_pmv"], r["dV_rom"]):
                if r["c"][list(r["basis"]).index(el)] > 1e-6:
                    row[f"dV_pmv_{el}"] = dp
                    row[f"dV_rom_{el}"] = dr
            rows.append(row)
        return pd.DataFrame(rows)
