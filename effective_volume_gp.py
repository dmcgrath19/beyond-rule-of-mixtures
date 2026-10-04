"""Physics-informed GP for MPEA hardness with a learnable effective-volume misfit.

The model is a Gaussian process whose prior mean is the Maresca--Curtin
solid-solution-strengthening prediction converted to Vickers hardness, with an
ARD kernel over compositional descriptors learning the residual. Three arms share
the kernel and differ only in how the reduced misfit parameter sigma entering the
prior is formed:

    analytic            sigma = sigma_ROM                                  (rule of mixtures)
    shared_sigma        sigma = sigma_ROM * exp(beta tanh g_theta(c))
    effective_volume    V_i^eff = V_i^base * exp(beta tanh h_theta,i(c))
                        V_eq^eff = sum_i c_i V_i^eff
                        dV_i^eff = V_i^eff - V_eq^eff
                        sigma    = sum_i c_i (dV_i^eff)^2

with V_i^base = a_i^3/2 from the tabulated BCC lattice constants. Only sigma is
replaced inside the prior; G, nu, b and T are unchanged. The centring identity
sum_i c_i dV_i^eff = 0 holds exactly, so sigma is a variance, and the correction
heads start with a zeroed output layer, so at initialisation every arm equals the
analytic-mean model. A composition-only GP with the same kernel and a constant
mean (`variant="constant_mean"`) is available as the control.

Training is staged marginal likelihood: kernel hyperparameters first with the
mean frozen, then everything jointly. The physics prefactors (alpha, M, f_L and
the thermal-activation constants) are learnable log-parameters.

Usage
-----
    python effective_volume_gp.py                      # all three arms, formula-grouped CV
    python effective_volume_gp.py --protocol shuffled --seeds 5
    python effective_volume_gp.py --beta 0.05 --hidden 16
    python effective_volume_gp.py --dump-volumes learned_volumes.csv
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import gpytorch
import numpy as np
import pandas as pd
import torch
from pymatgen.core.composition import Composition
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold, KFold

from curtin_ys_prior import ELASTIC_TENSOR_DICT, LATTICE_CONSTANTS_BCC_EXP

warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64

BORG_XLSX = REPO / "Borg_Datase_PUB.xlsx"

VARIANTS = ("analytic", "shared_sigma", "effective_volume")

# Kernel descriptors as named in the Borg spreadsheet, and the processing code.
BORG_NUM_COLS = ["R Var", "R", "B_avgr", "G_avgr", "V_Delt", "VEC Avg", "Tm Avg", "R_Delt"]
PROC_MAP = {"CAST": 0, "ANNEAL": 1, "WROUGHT": 2, "OTHER": 3}

# Elemental Vickers hardness (HV), used only when `use_intrinsic=True`.
ELEMENTAL_HV = {
    "W": 350, "Mo": 183, "Re": 275, "Hf": 175, "Mn": 250,
    "Cr": 110, "Ta": 90, "Co": 110, "Ti": 90, "Fe": 70,
    "Ni": 70, "Zr": 182, "Nb": 70, "V": 65, "Cu": 45,
    "Al": 17, "Si": 1100,
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def curtin_intermediates(formula: str, temperature: float):
    """(G_rom, poisson, burgers, sigma_ROM, T, hv_intrinsic) for a formula, or None.

    Rule-of-mixtures elastic constants and atomic volumes from the elemental
    tables in `curtin_ys_prior`; G in GPa, b in Angstrom, sigma_ROM in A^6.
    """
    try:
        comp = Composition(formula)
        eq_vol = sum(comp.get_atomic_fraction(el) * (LATTICE_CONSTANTS_BCC_EXP[el.name] ** 3) / 2
                     for el in comp.elements)
        sigma = sum(comp.get_atomic_fraction(el)
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
        return (G, nu, b, sigma, temperature, hv_intrinsic)
    except Exception:
        return None


def build_borg(basis: tuple[str, ...] | None = None, xlsx: Path = BORG_XLSX):
    """Borg rows with complete hardness, descriptors, formula and processing method.

    Returns Xk (kernel features), phys (G, nu, b, sigma_ROM, T, HV_intr),
    comp (atomic fractions on `basis`), y, formulas, basis.
    """
    df = pd.read_excel(xlsx).dropna(subset=BORG_NUM_COLS + ["HV", "FORMULA", "Processing method"])
    Xk, phys, y, formulas, comps = [], [], [], [], []
    for _, r in df.iterrows():
        temp = float(r["Test temperature"]) + 273.15 if not pd.isna(r["Test temperature"]) else 298.15
        ph = curtin_intermediates(r["FORMULA"], temp)
        if ph is None:
            continue
        Xk.append([float(r[c]) for c in BORG_NUM_COLS] + [PROC_MAP.get(r["Processing method"], 3), ph[4]])
        phys.append(list(ph))
        y.append(float(r["HV"]))
        formulas.append(str(r["FORMULA"]))
        comps.append(Composition(str(r["FORMULA"])))

    if basis is None:
        present = sorted({el.name for c in comps for el in c.elements})
        basis = tuple(el for el in present if el in LATTICE_CONSTANTS_BCC_EXP)

    comp_mat = np.zeros((len(comps), len(basis)), dtype=float)
    for j, c in enumerate(comps):
        for i, el in enumerate(basis):
            try:
                comp_mat[j, i] = c.get_atomic_fraction(el)
            except (ValueError, KeyError):
                comp_mat[j, i] = 0.0
        tot = comp_mat[j].sum()
        if tot > 0:
            comp_mat[j] /= tot

    return np.asarray(Xk, float), np.asarray(phys, float), comp_mat, np.asarray(y, float), formulas, basis


def curtin_analytic_hv(phys: np.ndarray, use_intrinsic: bool = False) -> np.ndarray:
    """Bare Maresca--Curtin prior in HV at fixed constants.

    alpha = 0.04, f_L = 1/12, M = 3, ln(rate ratio) = ln 1e7, Tabor factor 3/9.81.
    """
    G, nu, b, sigma, T, hv_intr = (phys[:, i] for i in range(6))
    ys_0K = (
        0.04 * (1 / 12) ** (-1 / 3) * G
        * ((1 + nu) / (1 - nu)) ** (4 / 3)
        * (sigma / b**6) ** (2 / 3) * 1000.0 * 3.0
    )
    Ea = (
        2.0 * (1 / 12) ** (1 / 3) * G * b**3
        * ((1 + nu) / (1 - nu)) ** (2 / 3)
        * (sigma / b**6) ** (1 / 3) / 160.21766208
    )
    thermal = np.exp(-1 / 0.55 * ((8.617333262e-5 * T / Ea) * np.log(1e7)) ** 0.91)
    base = ys_0K * thermal * (3.0 / 9.81)
    return base + hv_intr if use_intrinsic else base


# ---------------------------------------------------------------------------
# learnable misfit heads
# ---------------------------------------------------------------------------
class SigmaHead(torch.nn.Module):
    """g_theta: composition -> one bounded log-scale on sigma_ROM.

    `bounded=False` removes the beta*tanh envelope (the unbounded ablation).
    """

    def __init__(self, n_el: int, hidden: int = 32, beta: float = 0.02, bounded: bool = True):
        super().__init__()
        self.beta = beta
        self.bounded = bounded
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_el, hidden, dtype=DT),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1, dtype=DT),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        h = self.net(c)
        h = self.beta * torch.tanh(h) if self.bounded else h
        return torch.exp(h.clamp(-8.0, 8.0)).squeeze(-1)


class VolumeHead(torch.nn.Module):
    """h_theta,i: composition -> per-element bounded log-volume shift.

    One network with N_el outputs. The output layer starts at zero, so
    V_i^eff = V_i^base and the arm is initially identical to the analytic-mean model.
    """

    def __init__(self, n_el: int, hidden: int = 32, beta: float = 0.02, bounded: bool = True):
        super().__init__()
        self.beta = beta
        self.bounded = bounded
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_el, hidden, dtype=DT),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_el, dtype=DT),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        h = self.net(c)
        return self.beta * torch.tanh(h) if self.bounded else h.clamp(-8.0, 8.0)


def reconstruct_misfit(c: torch.Tensor, v_eff: torch.Tensor):
    """Centred effective misfits and the reduced misfit parameter.

    Returns (sigma, dV_eff, V_eq_eff); sum_i c_i dV_i^eff = 0 by construction.
    """
    v_eq = (c * v_eff).sum(-1, keepdim=True)
    dv = v_eff - v_eq
    sigma = (c * dv.pow(2)).sum(-1)
    return sigma, dv, v_eq.squeeze(-1)


class EffVolCurtinMean(gpytorch.means.Mean):
    """Maresca--Curtin prior mean with an optional learnable misfit parameter.

    Feature layout: [kernel features][phys: G nu b sigma T HV_intr][comp].
    Only sigma is replaced by the learned quantity; G, nu, b and T are untouched.
    """

    def __init__(
        self,
        physics_start: int,
        comp_start: int,
        v_base: torch.Tensor,
        variant: str = "analytic",
        beta: float = 0.02,
        hidden: int = 32,
        use_intrinsic: bool = False,
        b_from_eff: bool = False,
        bounded: bool = True,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.p = physics_start
        self.cs = comp_start
        self.variant = variant
        self.use_intrinsic = use_intrinsic
        self.b_from_eff = b_from_eff
        self.register_buffer("v_base", v_base)

        self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(0.04), dtype=DT))
        self.log_taylor = torch.nn.Parameter(torch.tensor(math.log(3.0), dtype=DT))
        self.log_hv_scale = torch.nn.Parameter(torch.tensor(math.log(3.0 / 9.81), dtype=DT))
        self.log_thermal_inv_c = torch.nn.Parameter(torch.tensor(math.log(1.0 / 0.55), dtype=DT))
        self.log_thermal_exp = torch.nn.Parameter(torch.tensor(math.log(0.91), dtype=DT))
        self.log_line_tension = torch.nn.Parameter(torch.tensor(math.log(1.0 / 12.0), dtype=DT))

        n_el = v_base.numel()
        self.head = None
        if variant == "shared_sigma":
            self.head = SigmaHead(n_el, hidden, beta, bounded)
        elif variant == "effective_volume":
            self.head = VolumeHead(n_el, hidden, beta, bounded)

    def misfit(self, x: torch.Tensor):
        """(sigma, dV_eff or None, b_override or None) for this batch."""
        sigma_rom = x[..., self.p + 3].clamp(min=1e-30)
        if self.variant == "analytic":
            return sigma_rom, None, None

        c = x[..., self.cs : self.cs + self.v_base.numel()]
        if self.variant == "shared_sigma":
            return (sigma_rom * self.head(c)).clamp(min=1e-30), None, None

        v_eff = self.v_base * torch.exp(self.head(c))
        sigma, dv, v_eq = reconstruct_misfit(c, v_eff)
        b_over = None
        if self.b_from_eff:
            b_over = (2.0 * v_eq).pow(1.0 / 3.0) * (math.sqrt(3.0) / 2.0)
        return sigma.clamp(min=1e-30), dv, b_over

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp()
        taylor = self.log_taylor.exp()
        hv_sc = self.log_hv_scale.exp()
        inv_c = self.log_thermal_inv_c.exp()
        t_exp = self.log_thermal_exp.exp()
        lt = self.log_line_tension.exp()

        G = x[..., self.p]
        nu = x[..., self.p + 1]
        b = x[..., self.p + 2]
        T = x[..., self.p + 4]
        hv_intr = x[..., self.p + 5]

        sigma, _, b_over = self.misfit(x)
        if b_over is not None:
            b = b_over

        nu_fac = (1 + nu) / (1 - nu)
        red = (sigma / b.pow(6)).clamp(min=1e-30)
        ys_0K = alpha * lt.pow(-1.0 / 3.0) * G * nu_fac.pow(4.0 / 3.0) * red.pow(2.0 / 3.0) * 1000.0 * taylor
        Ea = (
            2.0 * lt.pow(1.0 / 3.0) * G * b.pow(3) * nu_fac.pow(2.0 / 3.0) * red.pow(1.0 / 3.0) / 160.21766208
        ).clamp(min=1e-30)
        thermal_arg = ((8.617333262e-5 * T / Ea) * math.log(1e7)).clamp(min=1e-30)
        hv = ys_0K * torch.exp(-inv_c * thermal_arg.pow(t_exp)) * hv_sc
        return hv + hv_intr if self.use_intrinsic else hv


class EffVolGP(gpytorch.models.ExactGP):
    """Physics prior mean plus an ARD residual kernel over the standardized descriptors."""

    def __init__(self, train_x, train_y, lik, n_feat, mean_module, kernel="matern52", mean_in_hv=True):
        super().__init__(train_x, train_y, lik)
        self.mean_module = mean_module
        self.mean_in_hv = mean_in_hv
        dims = list(range(n_feat))
        base = (
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_feat, active_dims=dims)
            if kernel == "matern52"
            else gpytorch.kernels.RBFKernel(ard_num_dims=n_feat, active_dims=dims)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(base)
        self.register_buffer("y_mean", torch.tensor(0.0, dtype=DT))
        self.register_buffer("y_std", torch.tensor(1.0, dtype=DT))

    def set_y_norm(self, mu, sd):
        self.y_mean.fill_(mu)
        self.y_std.fill_(sd)

    def forward(self, x):
        raw = self.mean_module(x)
        # A physics mean produces HV and must be standardized; a ConstantMean is
        # already learned in standardized space.
        mean_x = (raw - self.y_mean) / self.y_std if self.mean_in_hv else raw
        return gpytorch.distributions.MultivariateNormal(mean_x, self.covar_module(x))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _augment(scaler, Xk, phys, comp, fit_scaler=False):
    Xs = scaler.fit_transform(Xk) if fit_scaler else scaler.transform(Xk)
    return np.hstack([Xs, phys, comp])


def fit_fold(
    Xk, phys, comp, y, v_base, *, variant="effective_volume", kernel="matern52",
    beta=0.02, hidden=32, use_intrinsic=False, b_from_eff=False, bounded=True,
    residual=True, seed=0, n1=150, n2=500, lr1=0.05, lr2=0.01,
):
    """Staged marginal-likelihood training on one training set.

    Stage 1 freezes the mean (physics prefactors and head) and fits the kernel
    hyperparameters for `n1` steps; stage 2 unfreezes everything for `n2` steps.

    `variant="constant_mean"` is the composition-only control. `residual=False`
    fits the physics mean and head alone by least squares, without the GP residual.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = preprocessing.StandardScaler()
    X_aug = _augment(scaler, Xk, phys, comp, fit_scaler=True)
    n_feat = Xk.shape[1]
    ymu, ysd = float(y.mean()), float(y.std())

    tx = torch.tensor(X_aug, dtype=DT, device=DEV)
    ty = torch.tensor((y - ymu) / ysd, dtype=DT, device=DEV)

    if variant == "constant_mean":
        mean = gpytorch.means.ConstantMean().double()
    else:
        mean = EffVolCurtinMean(
            physics_start=n_feat,
            comp_start=n_feat + phys.shape[1],
            v_base=v_base.to(DEV),
            variant=variant,
            beta=beta,
            hidden=hidden,
            use_intrinsic=use_intrinsic,
            b_from_eff=b_from_eff,
            bounded=bounded,
        )

    if not residual:
        if variant == "constant_mean":
            raise ValueError("residual=False is meaningless for the constant-mean control")
        mean = mean.double().to(DEV)
        opt = torch.optim.Adam(mean.parameters(), lr=lr2)
        ty_hv = torch.tensor(y, dtype=DT, device=DEV)
        for _ in range(n1 + n2):
            opt.zero_grad()
            torch.nn.functional.mse_loss(mean(tx), ty_hv).backward()
            opt.step()
        return dict(model=None, mean=mean, lik=None, scaler=scaler,
                    ymu=ymu, ysd=ysd, n_feat=n_feat, residual=False)

    lik = gpytorch.likelihoods.GaussianLikelihood().double().to(DEV)
    model = EffVolGP(tx, ty, lik, n_feat, mean, kernel, mean_in_hv=(variant != "constant_mean")).double().to(DEV)
    model.set_y_norm(ymu, ysd)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    model.train()
    lik.train()

    model.mean_module.requires_grad_(False)
    stage1 = [p for p in model.parameters() if p.requires_grad] + list(lik.parameters())
    o1 = torch.optim.Adam(stage1, lr=lr1)
    for _ in range(n1):
        o1.zero_grad()
        (-mll(model(tx), ty)).backward()
        o1.step()

    model.mean_module.requires_grad_(True)
    o2 = torch.optim.Adam(model.parameters(), lr=lr2)
    for _ in range(n2):
        o2.zero_grad()
        (-mll(model(tx), ty)).backward()
        o2.step()

    model.eval()
    lik.eval()
    return dict(model=model, lik=lik, scaler=scaler, ymu=ymu, ysd=ysd, n_feat=n_feat, residual=True)


def predict(fit, Xk, phys, comp):
    """Predictive mean and standard deviation (HV), the latter including observation noise."""
    tx = torch.tensor(_augment(fit["scaler"], Xk, phys, comp), dtype=DT, device=DEV)
    with torch.no_grad():
        if not fit.get("residual", True):
            pred = fit["mean"](tx).cpu().numpy()
            return pred, np.zeros_like(pred)
        post = fit["model"](tx)
        mean = post.mean.cpu().numpy() * fit["ysd"] + fit["ymu"]
        sd = fit["lik"](post).stddev.cpu().numpy() * fit["ysd"]
    return mean, sd


def learned_volumes(fit, Xk, phys, comp, basis):
    """Element-resolved dV_i^eff at the given rows, or None for arms without a volume head."""
    mean = fit["mean"] if not fit.get("residual", True) else fit["model"].mean_module
    if not isinstance(mean, EffVolCurtinMean) or mean.variant != "effective_volume":
        return None
    tx = torch.tensor(_augment(fit["scaler"], Xk, phys, comp), dtype=DT, device=DEV)
    with torch.no_grad():
        _, dv, _ = mean.misfit(tx)
    return dv.cpu().numpy()


def cross_val(
    Xk, phys, comp, y, formulas, v_base, basis, *, variant="effective_volume",
    kernel="matern52", protocol="grouped", beta=0.02, hidden=32,
    use_intrinsic=False, b_from_eff=False, bounded=True, residual=True,
    seed=0, n_splits=5, n1=150, n2=500, lr1=0.05, lr2=0.01,
    collect_volumes=False, splits=None,
):
    """Out-of-fold predictions under the requested fold protocol.

    `protocol="grouped"` keeps every row of a formula in one fold; `"shuffled"` is
    plain row-level KFold. Pass `splits` to score several arms on identical folds.
    """
    if splits is not None:
        splits = list(splits)
    elif protocol == "grouped":
        groups = pd.factorize(np.asarray(formulas))[0]
        splits = list(GroupKFold(n_splits=n_splits).split(Xk, y, groups))
    elif protocol == "shuffled":
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Xk))
    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    oof = np.full(len(y), np.nan)
    oof_sd = np.full(len(y), np.nan)
    vols = np.full((len(y), len(basis)), np.nan) if collect_volumes else None

    for tr, te in splits:
        fit = fit_fold(
            Xk[tr], phys[tr], comp[tr], y[tr], v_base,
            variant=variant, kernel=kernel, beta=beta, hidden=hidden,
            use_intrinsic=use_intrinsic, b_from_eff=b_from_eff,
            bounded=bounded, residual=residual,
            seed=seed, n1=n1, n2=n2, lr1=lr1, lr2=lr2,
        )
        oof[te], oof_sd[te] = predict(fit, Xk[te], phys[te], comp[te])
        if collect_volumes:
            dv = learned_volumes(fit, Xk[te], phys[te], comp[te], basis)
            if dv is not None:
                vols[te] = dv

    return oof, oof_sd, vols


def metrics(pred, y):
    """(MAE, RMSE) in HV."""
    e = pred - y
    return float(np.abs(e).mean()), float(np.sqrt((e**2).mean()))


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS + ("constant_mean",))
    p.add_argument("--kernel", default="matern52", choices=("matern52", "rbf"))
    p.add_argument("--protocol", default="grouped", choices=("grouped", "shuffled"))
    p.add_argument("--beta", type=float, default=0.02, help="bound on the log volume correction")
    p.add_argument("--hidden", type=int, default=32, help="hidden width of the correction heads")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--n1", type=int, default=150, help="warm-up steps (kernel only)")
    p.add_argument("--n2", type=int, default=500, help="joint steps")
    p.add_argument("--threads", type=int, default=1,
                   help="torch CPU threads; 1 makes runs bitwise reproducible")
    p.add_argument("--intrinsic", action="store_true", help="add the elemental-HV term to the prior mean")
    p.add_argument("--b-from-eff", action="store_true", help="also take the Burgers vector from V_eq^eff")
    p.add_argument("--dump-volumes", help="write per-row learned dV_i^eff to this CSV")
    a = p.parse_args(argv)

    torch.set_num_threads(a.threads)

    Xk, phys, comp, y, formulas, basis = build_borg()
    v_base = torch.tensor([LATTICE_CONSTANTS_BCC_EXP[el] ** 3 / 2 for el in basis], dtype=DT)

    print(f"Borg rows after filter : {len(y)}")
    print(f"unique formulae        : {len(set(formulas))}")
    print(f"element basis ({len(basis)})     : {' '.join(basis)}")
    print(f"kernel / protocol      : {a.kernel} / {a.protocol} 5-fold"
          f"{f', {a.seeds} seeds' if a.seeds > 1 else ''}")
    print(f"beta / hidden          : {a.beta} / {a.hidden}"
          f"   (max volume shift {100 * (math.exp(a.beta) - 1):.1f}%)")
    print(f"intrinsic HV term      : {a.intrinsic}")

    prior = curtin_analytic_hv(phys, use_intrinsic=a.intrinsic)
    mae, rmse = metrics(prior, y)
    print(f"\n{'model':<34} {'MAE':>7} {'RMSE':>7}")
    print("-" * 52)
    print(f"{'Curtin prior (analytical)':<34} {mae:7.1f} {rmse:7.1f}")

    labels = {
        "analytic": "PI-GP (analytic mean)",
        "shared_sigma": "PI-GP + shared-sigma",
        "effective_volume": "PI-GP + effective volume",
        "constant_mean": "Composition-only GP",
    }
    dumps = {}
    for variant in a.variants:
        maes, rmses = [], []
        for s in range(a.seeds):
            oof, _, vols = cross_val(
                Xk, phys, comp, y, formulas, v_base, basis,
                variant=variant, kernel=a.kernel, protocol=a.protocol,
                beta=a.beta, hidden=a.hidden, use_intrinsic=a.intrinsic,
                b_from_eff=a.b_from_eff, seed=s, n1=a.n1, n2=a.n2,
                collect_volumes=bool(a.dump_volumes) and s == 0,
            )
            m, r = metrics(oof, y)
            maes.append(m)
            rmses.append(r)
            if vols is not None and s == 0:
                dumps[variant] = (oof, vols)
        spread = f"   seed sd +/-{np.std(maes, ddof=1):.1f}" if len(maes) > 1 else ""
        print(f"{labels[variant]:<34} {np.mean(maes):7.1f} {np.mean(rmses):7.1f}{spread}")

    if a.dump_volumes and "effective_volume" in dumps:
        oof, vols = dumps["effective_volume"]
        rows = []
        for j, f in enumerate(formulas):
            for i, el in enumerate(basis):
                if comp[j, i] > 1e-6:
                    rows.append(dict(formula=f, element=el, x=comp[j, i],
                                     dV_eff=vols[j, i], HV_exp=y[j], HV_oof=oof[j]))
        pd.DataFrame(rows).to_csv(a.dump_volumes, index=False)
        print(f"\nwrote {a.dump_volumes}: {len(rows)} alloy-element rows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
