"""Common paper training engine for standalone composition prediction."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import gpytorch
import numpy as np
import pandas as pd
import torch

from lib.physics import ELEMENTAL_BCC_VOLUMES, SUPPORTED_ELEMENTS
from models import get_model_config
from run import _train_gp_fold

from hardness_predictor import build_borg_recomputed

ARMS = {"PI-GP (analytic mean)": dict(variant="analytic"),
        "PI-GP + effective volume": dict(variant="effective_volume"),
        "Composition-only GP": dict(variant="constant_mean")}


class Data:
    def __init__(self):
        self.basis = list(SUPPORTED_ELEMENTS)
        self.Xk, self.phys, self.comp, self.y, self.formulas = build_borg_recomputed(self.basis)
        self.ti = self.basis.index("Ti")
        self.groups = pd.factorize(np.asarray(self.formulas))[0]

    def v_base(self, anchor="current", a_ti=None):
        volumes = np.array(ELEMENTAL_BCC_VOLUMES, dtype=float)
        if anchor == "mlip":
            for element, value in dict(Mo=15.859, Nb=18.128, Ta=18.556, Ti=17.305, W=16.144).items():
                volumes[self.basis.index(element)] = value
        elif anchor != "current":
            raise ValueError(anchor)
        if a_ti is not None:
            volumes[self.ti] = a_ti**3 / 2
        return torch.tensor(volumes, dtype=torch.float64)


def fit_fold(Xk, phys, comp, y, v_base=None, *, variant="effective_volume", seed=0,
             beta=.05, hidden=4, kernel="rbf", n1=None, n2=None, **kwargs):
    if kernel not in ("rbf", "matern52") or hidden != 4:
        raise ValueError("Paper engine supports RBF/Matérn-5/2 and correction width 4")
    if kwargs:
        raise ValueError(f"Unsupported training overrides: {kwargs}")
    config = get_model_config("gpytorch_nonlinear_sigma" if variant in ("effective_volume", "shared_sigma") else "gpytorch")
    if variant == "constant_mean":
        config = replace(config, mean_type="constant")
    elif variant == "shared_sigma":
        config = replace(config, sigma_variant="shared")
    elif variant not in ("analytic", "effective_volume"):
        raise ValueError(variant)
    config = replace(config, sigma_log_bound=beta, kernel=kernel)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    result = {}

    def capture(model, likelihood, scaler, y_mean, y_std):
        result.update(model=model, lik=likelihood, scaler=scaler, ymu=y_mean, ysd=y_std)

    anchors = None if v_base is None or variant not in ("effective_volume", "shared_sigma") else np.asarray(v_base, dtype=float)
    if variant == "shared_sigma":
        anchors = None
    import run as engine
    saved_steps = engine.STAGE1_ITERS, engine.STAGE2_ITERS
    try:
        if n1 is not None:engine.STAGE1_ITERS = int(n1)
        if n2 is not None:engine.STAGE2_ITERS = int(n2)
        if engine.STAGE1_ITERS < 0 or engine.STAGE2_ITERS < 0:raise ValueError("Training steps must be nonnegative")
        _train_gp_fold(Xk, Xk, phys[:, :5], phys[:, :5], comp, comp, y, f"{variant} seed={seed}",
                       config, capture_fit=capture, volume_anchors=anchors)
    finally:
        engine.STAGE1_ITERS, engine.STAGE2_ITERS = saved_steps
    result["variant"] = variant
    return result


def augmented(fit, Xk, phys, comp):
    return torch.tensor(np.hstack([fit["scaler"].transform(Xk), phys[:, :5], comp]), dtype=torch.float64)


def predict(fit, Xk, phys, comp):
    with torch.no_grad():
        posterior = fit["model"](augmented(fit, Xk, phys, comp))
        mu = posterior.mean.numpy() * fit["ysd"] + fit["ymu"]
        sd = fit["lik"](posterior).stddev.numpy() * fit["ysd"]
    return mu, sd


def cross_val(Xk, phys, comp, y, formulas, v_base, basis, *, splits, seed=0,
              collect_volumes=False, **kwargs):
    if list(basis) != list(SUPPORTED_ELEMENTS):
        raise ValueError("Element basis differs from the grouped training engine")
    mu, sd = np.empty(len(y)), np.empty(len(y))
    volumes = np.empty_like(comp) if collect_volumes else None
    for train, test in splits:
        fit = fit_fold(Xk[train], phys[train], comp[train], y[train], v_base, seed=seed, **kwargs)
        mu[test], sd[test] = predict(fit, Xk[test], phys[test], comp[test])
        if collect_volumes:
            with torch.no_grad():
                diag = fit["model"].mean_module.sigma_diagnostics(augmented(fit, Xk[test], phys[test], comp[test]))
            if diag.delta_volumes is None:
                volumes = None
            elif volumes is not None:
                volumes[test] = diag.delta_volumes.numpy()
    return mu, sd, volumes
