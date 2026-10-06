"""Gaussian scoring helpers for the public learning-curve experiment."""
import numpy as np
from scipy import stats
from borg_experiments import BETA
ARMS = {"PI-GP (analytic mean)": dict(variant="analytic"), "PI-GP + effective volume": dict(variant="effective_volume", beta=BETA), "Composition-only GP": dict(variant="constant_mean")}
def nlpd(y, mu, sd):
    sd = np.clip(sd, 1e-9, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + (y - mu) ** 2 / (2 * sd**2)))

def crps_gaussian(y, mu, sd):
    """Closed-form CRPS for a Gaussian predictive distribution (lower is better)."""
    sd = np.clip(sd, 1e-9, None)
    z = (y - mu) / sd
    return float(np.mean(
        sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    ))

def score_block(y, mu, sd):
    return dict(
        mae=float(np.abs(mu - y).mean()),
        cov68=100 * coverage(y, mu, sd, 0.68),
        cov95=100 * coverage(y, mu, sd, 0.95),
        nlpd=nlpd(y, mu, sd),
        crps=crps_gaussian(y, mu, sd),
        sharp=float(np.mean(sd)),
    )
