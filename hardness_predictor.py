"""Hierarchical hardness feature for the ductility model.

Factors the hardness GP from `gpytorch-lab.py` into a reusable predictor:
train on Antimatter (qness) + Borg hardness data, then predict Vickers hardness
(HV) from composition alone. The ductility notebook feeds the PREDICTED hardness
in as a feature (hardness and ductility are inversely correlated), which keeps
the ductility model deployable pre-synthesis since hardness itself is predicted
from composition.

Model is identical to gpytorch-lab.py: Curtin-Varvenne physics mean + ARD-RBF
kernel. Composition features are computed locally (no Ray) via
composition_to_features, so the predictor is self-contained.
"""
from __future__ import annotations

import math
import numpy as np
import gpytorch
import torch
from sklearn import preprocessing
from pymatgen.core.composition import Composition
from atlas.workflows.hea.composition_to_features import composition_to_features
from curtin_ys_prior import (
    compute_yield_strength,
    ELASTIC_TENSOR_DICT,
    LATTICE_CONSTANTS_BCC_EXP,
)

# 8 composition kernel features, in the SAME order the hardness GP uses
# (gpytorch-lab.py hv_feat_idx / borg_numeric_cols).
_KERNEL_KEYS = ["R_Var", "R_pm", "B_GPa", "G_GPa", "Poisson_Delt", "VEC", "Tm_K", "R_Delt"]
_BORG_NUM_COLS = ["R Var", "R", "B_avgr", "G_avgr", "V_Delt", "VEC Avg", "Tm Avg", "R_Delt"]
_PROC_MAP = {"CAST": 0, "ANNEAL": 1, "WROUGHT": 2, "OTHER": 3}

ELEMENTAL_HV = {
    "W": 350, "Mo": 183, "Re": 275, "Hf": 175, "Mn": 250,
    "Cr": 110, "Ta": 90, "Co": 110, "Ti": 90, "Fe": 70,
    "Ni": 70, "Zr": 182, "Nb": 70, "V": 65, "Cu": 45,
    "Al": 17, "Si": 1100,
}


class CurtinHardnessMean(gpytorch.means.Mean):
    """Curtin-Varvenne yield-strength -> Vickers hardness prior (see gpytorch-lab.py)."""

    def __init__(self, physics_start: int):
        super().__init__()
        self.p = physics_start
        self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(0.04), dtype=torch.float64))
        self.log_taylor = torch.nn.Parameter(torch.tensor(math.log(3.0), dtype=torch.float64))
        self.log_hv_scale = torch.nn.Parameter(torch.tensor(math.log(3.0 / 9.81), dtype=torch.float64))
        self.log_thermal_inv_c = torch.nn.Parameter(torch.tensor(math.log(1.0 / 0.55), dtype=torch.float64))
        self.log_thermal_exp = torch.nn.Parameter(torch.tensor(math.log(0.91), dtype=torch.float64))
        self.log_line_tension = torch.nn.Parameter(torch.tensor(math.log(1.0 / 12.0), dtype=torch.float64))

    def forward(self, x):
        alpha = self.log_alpha.exp()
        taylor = self.log_taylor.exp()
        hv_sc = self.log_hv_scale.exp()
        inv_c = self.log_thermal_inv_c.exp()
        t_exp = self.log_thermal_exp.exp()
        lt = self.log_line_tension.exp()

        G = x[..., self.p]
        nu = x[..., self.p + 1]
        b = x[..., self.p + 2]
        Sigma = x[..., self.p + 3].clamp(min=1e-30)
        T = x[..., self.p + 4]
        hv_intrinsic = x[..., self.p + 5]

        ys_0K = (
            alpha * lt.pow(-1.0 / 3.0) * G
            * ((1 + nu) / (1 - nu)) ** (4.0 / 3.0)
            * (Sigma / b.pow(6)).pow(2.0 / 3.0) * 1000.0 * taylor
        )
        Ea = (
            2.0 * lt.pow(1.0 / 3.0) * G * b.pow(3)
            * ((1 + nu) / (1 - nu)) ** (2.0 / 3.0)
            * (Sigma / b.pow(6)).pow(1.0 / 3.0) / 160.21766208
        ).clamp(min=1e-30)
        thermal_arg = ((8.617333262e-5 * T / Ea) * math.log(1e7)).clamp(min=1e-30)
        ys_sss = ys_0K * torch.exp(-inv_c * thermal_arg.pow(t_exp))
        return hv_intrinsic + ys_sss * hv_sc


class HardnessGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_features, physics_start, y_mean, y_std):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = CurtinHardnessMean(physics_start)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_features, active_dims=list(range(n_features)))
        )

    def forward(self, x):
        raw_mean = self.mean_module(x)
        mean_x = (raw_mean - self.y_mean) / self.y_std
        return gpytorch.distributions.MultivariateNormal(mean_x, self.covar_module(x))


def _curtin_intermediates(formula, temperature):
    """(G_rom, poisson, burgers, reduced_misfit_vol, T, hv_intrinsic) or None on failure."""
    try:
        comp = Composition(formula)
        eq_vol = sum(comp.get_atomic_fraction(el) * (LATTICE_CONSTANTS_BCC_EXP[el.name] ** 3) / 2
                     for el in comp.elements)
        Sigma = sum(comp.get_atomic_fraction(el)
                    * ((LATTICE_CONSTANTS_BCC_EXP[el.name] ** 3) / 2 - eq_vol) ** 2
                    for el in comp.elements)
        C11 = sum(comp.get_atomic_fraction(el) * ELASTIC_TENSOR_DICT[el.name][0] for el in comp.elements)
        C12 = sum(comp.get_atomic_fraction(el) * ELASTIC_TENSOR_DICT[el.name][1] for el in comp.elements)
        C44 = sum(comp.get_atomic_fraction(el) * ELASTIC_TENSOR_DICT[el.name][2] for el in comp.elements)
        B = (C11 + 2 * C12) / 3
        G = math.sqrt(C44 * (C11 - C12) / 2)
        nu = (3 * B - 2 * G) / (2 * (3 * B + G))
        a = (eq_vol * 2) ** (1 / 3)
        b = a * math.sqrt(3) / 2
        hv_intrinsic = sum(comp.get_atomic_fraction(el) * ELEMENTAL_HV.get(el.name, 0.0)
                           for el in comp.elements)
        return (G, nu, b, Sigma, temperature, hv_intrinsic)
    except Exception:
        return None


def _comp_kernel_feats(comp_str):
    """8 composition kernel features in _KERNEL_KEYS order, or None if any missing."""
    try:
        feats = composition_to_features(comp_str, return_dict=True)
    except Exception:
        return None
    vals = [feats.get(k) for k in _KERNEL_KEYS]
    if any(v is None for v in vals):
        return None
    return [float(v) for v in vals]


def _get_measured_composition(collection, sample_id):
    sem_doc = collection.find_one({"sample_id": sample_id, "type": "sem"})
    if sem_doc and "data" in sem_doc:
        eds_list = (sem_doc["data"].get("big_data") or {}).get("eds_measurements") or []
        if eds_list:
            map_list = eds_list[-1].get("eds_map_measurements") or []
            if map_list and map_list[-1].get("measured_global_composition_at"):
                return map_list[-1]["measured_global_composition_at"]
    xrf_doc = collection.find_one({"sample_id": sample_id, "type": "xrf"})
    if xrf_doc and "data" in xrf_doc:
        d = xrf_doc["data"]
        return d.get("measured_composition_at") or d.get("actual_composition_at")
    return None


def _xrd_peak_count(collection, sample_id):
    xrd_doc = collection.find_one({"sample_id": sample_id, "type": "xrd"})
    if not xrd_doc or "data" not in xrd_doc:
        return None
    data = xrd_doc["data"]
    scans = data.get("scans") or (data.get("big_data") or {}).get("scans") or []
    if not scans:
        return None
    peaks = scans[0].get("peaks") if isinstance(scans[0], dict) else []
    return len(peaks) if peaks else 0


class HardnessPredictor:
    """Train the hardness GP, then predict HV from composition strings."""

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None
        self.model = None
        self.likelihood = None
        self._n_feat = None

    def fit(self, collection, borg_xlsx="./Borg_Datase_PUB.xlsx", rai_start=400,
            max_xrd_peaks=20, n_stage1=150, n_stage2=500, verbose=True):
        import pandas as pd

        # --- Antimatter (qness) ---
        qness = list(collection.find({"type": "qness", "data.mean_hardness_hv": {"$exists": True, "$ne": None}}))
        hv_by_sample = {}
        for qdoc in qness:
            sid = qdoc.get("sample_id")
            try:
                if not sid or int(sid.split("-")[-1]) < rai_start:
                    continue
            except Exception:
                continue
            hv = qdoc.get("data", {}).get("mean_hardness_hv")
            if hv is not None:
                hv_by_sample.setdefault(sid, []).append(float(hv))

        Xk, phys, y, src = [], [], [], []
        for sid, hvs in hv_by_sample.items():
            npk = _xrd_peak_count(collection, sid)
            if npk is not None and npk > max_xrd_peaks:
                continue
            comp = _get_measured_composition(collection, sid)
            if not comp:
                continue
            kf = _comp_kernel_feats(comp)
            ph = _curtin_intermediates(comp, 298.15)
            if kf is None or ph is None:
                continue
            Xk.append(kf + [_PROC_MAP["CAST"], 298.15])
            phys.append(list(ph))
            y.append(sum(hvs) / len(hvs))
            src.append("AM")
        n_am = len(y)

        # --- Borg ---
        try:
            df = pd.read_excel(borg_xlsx)
            df = df.dropna(subset=_BORG_NUM_COLS + ["HV", "FORMULA", "Processing method"])
            for _, r in df.iterrows():
                ph = _curtin_intermediates(
                    r["FORMULA"],
                    float(r["Test temperature"]) + 273.15 if not pd.isna(r["Test temperature"]) else 298.15,
                )
                if ph is None:
                    continue
                Xk.append([float(r[c]) for c in _BORG_NUM_COLS]
                          + [_PROC_MAP.get(r["Processing method"], 3), ph[4]])
                phys.append(list(ph))
                y.append(float(r["HV"]))
                src.append("Borg")
        except FileNotFoundError:
            if verbose:
                print("  [hardness] Borg Excel not found; training on Antimatter only")
        n_bg = len(y) - n_am

        Xk = np.asarray(Xk, dtype=float)
        phys = np.asarray(phys, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = ~(np.isnan(y) | np.any(np.isnan(Xk), axis=1) | np.any(np.isnan(phys), axis=1))
        Xk, phys, y = Xk[ok], phys[ok], y[ok]
        if verbose:
            print(f"  [hardness] training on {ok.sum()} samples ({n_am} Antimatter, {n_bg} Borg)")

        self.scaler = preprocessing.StandardScaler()
        Xk_std = self.scaler.fit_transform(Xk)
        X_aug = np.hstack([Xk_std, phys])
        self._n_feat = Xk.shape[1]

        self._ymu, self._ysd = float(np.mean(y)), float(np.std(y))
        tx = torch.tensor(X_aug, dtype=torch.float64, device=self.device)
        ty = torch.tensor((y - self._ymu) / self._ysd, dtype=torch.float64, device=self.device)
        ymu_t = torch.tensor(self._ymu, dtype=torch.float64, device=self.device)
        ysd_t = torch.tensor(self._ysd, dtype=torch.float64, device=self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double().to(self.device)
        self.model = HardnessGP(tx, ty, self.likelihood, self._n_feat, self._n_feat, ymu_t, ysd_t).double().to(self.device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train(); self.likelihood.train()
        self.model.mean_module.requires_grad_(False)
        opt1 = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad]
                                + list(self.likelihood.parameters()), lr=0.05)
        for _ in range(n_stage1):
            opt1.zero_grad(); loss = -mll(self.model(tx), ty); loss.backward(); opt1.step()
        self.model.mean_module.requires_grad_(True)
        opt2 = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for _ in range(n_stage2):
            opt2.zero_grad(); loss = -mll(self.model(tx), ty); loss.backward(); opt2.step()
        if verbose:
            print(f"  [hardness] trained (final NLL={loss.item():.3f})")
        return self

    def predict(self, comp_strings, temperature_K=298.15):
        """Predict HV for each composition; NaN where features are unavailable."""
        assert self.model is not None, "call fit() first"
        rows, valid = [], []
        for c in comp_strings:
            kf = _comp_kernel_feats(c)
            ph = _curtin_intermediates(c, temperature_K)
            if kf is None or ph is None:
                valid.append(False)
                rows.append(None)
            else:
                valid.append(True)
                rows.append((kf + [_PROC_MAP["CAST"], temperature_K], list(ph)))
        out = np.full(len(comp_strings), np.nan)
        idx = [i for i, v in enumerate(valid) if v]
        if not idx:
            return out
        Xk = np.array([rows[i][0] for i in idx], dtype=float)
        phys = np.array([rows[i][1] for i in idx], dtype=float)
        X_aug = np.hstack([self.scaler.transform(Xk), phys])
        tx = torch.tensor(X_aug, dtype=torch.float64, device=self.device)
        self.model.eval(); self.likelihood.eval()
        with torch.no_grad():
            mean = self.model(tx).mean.cpu().numpy() * self._ysd + self._ymu
        for j, i in enumerate(idx):
            out[i] = float(mean[j])
        return out
