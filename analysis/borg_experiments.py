"""Borg matrices and elemental reference volumes for public analyses."""
import numpy as np
import torch
from pymatgen.core.composition import Composition
from curtin_ys_prior import LATTICE_CONSTANTS_BCC_EXP as LATT
from effective_volume_gp import DT, metrics
from utils.paper_training import Data as PaperData, cross_val
from sklearn.model_selection import KFold, GroupKFold
import pandas as pd
QUINARY = ("Mo", "Nb", "Ta", "Ti", "W")
MLIP_VOLUME = dict(Mo=15.859, Nb=18.128, Ta=18.556, Ti=17.305, W=16.144)
BETA = 0.05
class Data(PaperData):
    """Borg design matrices plus the masks the Ti analysis needs."""

    def __init__(self):
        super().__init__()
        self.ti = self.basis.index("Ti")
        inq = np.array([
            {e.name for e in Composition(f).elements} <= set(QUINARY)
            for f in self.formulas
        ])
        self.ti_overlap = (self.comp[:, self.ti] > 1e-6) & inq
        self.groups = pd.factorize(np.asarray(self.formulas))[0]

    def v_base(self, anchor="current", a_ti=None):
        v = [LATT[e] ** 3 / 2 for e in self.basis]
        if anchor == "mlip":
            for i, e in enumerate(self.basis):
                if e in MLIP_VOLUME:
                    v[i] = MLIP_VOLUME[e]
        if a_ti is not None:
            v[self.ti] = a_ti**3 / 2
        return torch.tensor(v, dtype=DT)

    def run(self, v_base, *, seeds=3, collect=False, **kw):
        """Mean MAE/RMSE over seeds, plus Ti diagnostics in the overlap set."""
        protocol = kw.pop("protocol", "shuffled")
        n_splits = kw.pop("n_splits", 5)
        if protocol not in ("shuffled", "grouped"):
            raise ValueError(protocol)
        splits = list(KFold(n_splits, shuffle=True, random_state=42).split(self.y)) if protocol == "shuffled" else list(GroupKFold(n_splits).split(self.Xk, self.y, self.groups))
        maes, rmses, pos, meds = [], [], [], []
        for s in range(seeds):
            oof, _, vols = cross_val(
                self.Xk, self.phys, self.comp, self.y, self.formulas,
                v_base, self.basis, splits=splits, seed=s,
                collect_volumes=collect, **kw,
            )
            m, r = metrics(oof, self.y)
            maes.append(m)
            rmses.append(r)
            if collect and vols is not None:
                t = vols[self.ti_overlap, self.ti]
                pos.append(int((t > 0).sum()))
                meds.append(float(np.median(t)))
        out = dict(mae=np.mean(maes), sd=np.std(maes, ddof=1) if seeds > 1 else 0.0,
                   rmse=np.mean(rmses), n_ti=int(self.ti_overlap.sum()))
        if pos:
            out.update(ti_pos=np.mean(pos), ti_med=np.mean(meds))
        return out
