#!/usr/bin/env python3
"""
Map a chemical composition (formula) onto scalar features:
  melting point, VEC, bulk modulus, shear modulus, Poisson ratio,
  Allen electronegativity, configurational entropy, density, and
  isotropic elastic descriptors (B, G, E, nu).

- Configurational entropy: dimensionless S/R = -sum(x_i * ln(x_i)).
- VEC: standard HEA group-number convention (Guo & Liu 2011).

Uses pymatgen for composition parsing and elemental properties.
Optional: add wch3n scripts to path for consistency with that pipeline.

Requirements: pymatgen, numpy
  pip install pymatgen numpy

Elastic constants (C11, C12, C44) from a built-in BCC single-crystal
dictionary; B = (C11+2C12)/3, G = sqrt(C44*(C11-C12)/2).
Falls back to pymatgen elemental moduli for unlisted elements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from pymatgen.core import Element, Composition
except ImportError:
    raise ImportError("Install pymatgen: pip install pymatgen") from None

K_EV = 8.61733262e-5  # Boltzmann constant (eV/K), kept for reference

# Single-crystal elastic constants [C11, C12, C44] in GPa for BCC (or BCC-reduced) phases.
ELASTIC_TENSOR_DICT: dict[str, list[float]] = {
    "Al": [107, 61, 28],
    "Co": [307, 165, 75],
    "Cr": [350, 68, 101],
    "Fe": [226, 140, 116],
    "Ni": [246, 147, 124],
    "Mo": [463, 169, 109],
    "W": [522, 204, 161],
    "V": [227, 116, 47],
    "Nb": [246, 134, 29],
    "Ta": [267, 161, 87],
    "Ti": [162, 92, 47],
    "Zr": [144, 72, 33],
    "Hf": [176, 77, 51],
    "Mn": [246, 138, 110],
    "Cu": [168, 121, 75],
    "Si": [167, 65, 80],
    "Re": [591, 361, 162],
}

# Experimental (or metastable) BCC lattice constants in Å.
LATTICE_CONSTANTS_BCC: dict[str, float] = {
    "Cr": 2.884, "Fe": 2.8665, "Mo": 3.147, "W": 3.1652,
    "V": 3.03, "Nb": 3.300, "Ta": 3.301,
    "Al": 3.24, "Co": 2.82, "Ni": 2.88, "Cu": 2.89,
    "Ti": 3.32, "Zr": 3.57, "Hf": 3.53,
    "Mn": 2.99, "Si": 2.83, "Re": 3.11,
}


def _vec_per_element(el: Element) -> float:
    """
    Valence electron count per atom using the standard HEA convention
    (Guo & Liu, J. Appl. Phys. 2011):

      d-block  (groups 3-12):  VEC = group   (s + d electrons)
      s-block  (groups 1-2):   VEC = group   (s electrons)
      p-block  (groups 13-18): VEC = group-10 (s + p electrons only)
      f-block  (lanthanides/actinides): VEC = 3  (conventional)
    """
    g = getattr(el, "group", None)
    if g is None:
        return float("nan")
    if g >= 13:
        return float(g - 10)
    return float(g)


def _allen_electronegativity(el: Element) -> float | None:
    """Allen electronegativity if available, else Pauling (.X)."""
    x_allen = getattr(el, "X_allen", None)
    if x_allen is not None:
        return float(x_allen)
    x = getattr(el, "X", None)
    if x is not None:
        return float(x)
    return None


def _get_prop(el: Element, attr: str, default: float | None = None) -> float | None:
    v = getattr(el, attr, None)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default



def composition_to_features(
    formula: str,
    temperature_k: float | None = None,
    t_fac: float = 0.9,
    return_dict: bool = True,
) -> dict[str, float | None] | tuple:
    """
    Compute composition-weighted (rule-of-mixture) features for a formula.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. "NbTaTiZr", "Al20Nb20Ti25Ta20Hf15", "Hf0.5 Nb0.667 Ta0.333 Ti1 Zr0.833".
    temperature_k : float, optional
        Temperature (K). Reserved for future use.
    t_fac : float
        Reserved for future use (default 0.9).
    return_dict : bool
        If True return a dict; else return a tuple (same keys as dict values).

    Returns
    -------
    dict or tuple
        Keys: Tm_K, VEC, R_Angstrom, R_pm, C11_GPa, C12_GPa, C44_GPa,
              B_GPa, G_GPa, Poisson, Poisson_Var, Poisson_Delt, E_GPa,
              Allen_elec, S_conf_R, density_g_cm3.
    """
    comp = Composition(formula)
    frac = comp.fractional_composition
    el_amt = frac.get_el_amt_dict()
    elements = list(el_amt.keys())
    x = np.array([el_amt[el] for el in elements])

    # Melting point (K) — composition-weighted average
    Tm_vals = [_get_prop(Element(el), "melting_point") for el in elements]
    if all(v is not None for v in Tm_vals):
        Tm_K = float(np.dot(x, Tm_vals))
    else:
        Tm_K = None

    # VEC — composition-weighted average
    vec_vals = [_vec_per_element(Element(el)) for el in elements]
    if not np.any(np.isnan(vec_vals)):
        VEC = float(np.dot(x, vec_vals))
    else:
        VEC = None

    # Metallic radius — composition-weighted average (Goldschmidt / CN=12).
    # Falls back to atomic_radius when metallic_radius is unavailable.
    R_vals = [_get_prop(Element(el), "metallic_radius") or _get_prop(Element(el), "atomic_radius") for el in elements]
    if all(v is not None for v in R_vals):
        R_Angstrom = float(np.dot(x, R_vals))
        R_pm = R_Angstrom * 100.0
        R_arr = np.array(R_vals, dtype=float)
        mismatch_terms_R = x * (1 - R_arr / R_Angstrom) ** 2
        R_Var = float(np.sqrt(np.sum(mismatch_terms_R)))
        R_Delt = float(np.max(mismatch_terms_R) - np.min(mismatch_terms_R))
    else:
        R_Angstrom = None
        R_pm = None
        R_Var = None
        R_Delt = None

    # Elastic constants — ROM on single-crystal C11, C12, C44 (GPa).
    # Falls back to pymatgen bulk_modulus / rigidity_modulus / youngs_modulus
    # for elements not in ELASTIC_TENSOR_DICT.
    c11_vals, c12_vals, c44_vals = [], [], []
    _elastic_complete = True
    for el in elements:
        if el in ELASTIC_TENSOR_DICT:
            c11_vals.append(ELASTIC_TENSOR_DICT[el][0])
            c12_vals.append(ELASTIC_TENSOR_DICT[el][1])
            c44_vals.append(ELASTIC_TENSOR_DICT[el][2])
        else:
            _elastic_complete = False
            c11_vals.append(None)
            c12_vals.append(None)
            c44_vals.append(None)

    if _elastic_complete:
        C11 = float(np.dot(x, c11_vals))
        C12 = float(np.dot(x, c12_vals))
        C44 = float(np.dot(x, c44_vals))
        B_GPa = (C11 + 2.0 * C12) / 3.0
        G_GPa = float(np.sqrt(C44 * (C11 - C12) / 2.0))
        B_elem = [(c11_vals[i] + 2.0 * c12_vals[i]) / 3.0 for i in range(len(elements))]
    else:
        C11 = C12 = C44 = None
        # Fallback: pymatgen elemental B/G with derivation from E
        _B, _G = [], []
        for el in elements:
            e = Element(el)
            b = _get_prop(e, "bulk_modulus")
            g = _get_prop(e, "rigidity_modulus")
            ym = _get_prop(e, "youngs_modulus")
            if b is None and ym is not None and g is not None and (3 * g - ym) != 0:
                b = ym * g / (3.0 * (3.0 * g - ym))
            if g is None and b is not None and ym is not None and (9 * b - ym) != 0:
                g = 3.0 * b * ym / (9.0 * b - ym)
            _B.append(b)
            _G.append(g)
        B_GPa = float(np.dot(x, _B)) if all(v is not None for v in _B) else None
        G_GPa = float(np.dot(x, _G)) if all(v is not None for v in _G) else None
        B_elem = _B

    if all(v is not None for v in B_elem) and B_GPa is not None:
        B_arr = np.array(B_elem, dtype=float)
        mismatch_terms_B = x * (1 - B_arr / B_GPa) ** 2
        B_Var = float(np.sqrt(np.sum(mismatch_terms_B)))
        B_Delt = float(np.max(mismatch_terms_B) - np.min(mismatch_terms_B))
    else:
        B_Var = None
        B_Delt = None

    # Poisson ratio: nu = (3B - 2G) / (2·(3B + G))
    if B_GPa is not None and G_GPa is not None and (3 * B_GPa + G_GPa) != 0:
        Poisson = (3 * B_GPa - 2 * G_GPa) / (2.0 * (3 * B_GPa + G_GPa))
    else:
        Poisson = None

    # Per-element Poisson ratios for mismatch descriptors
    nu_elem: list[float | None] = []
    for el in elements:
        if el in ELASTIC_TENSOR_DICT:
            c11_e, c12_e, c44_e = ELASTIC_TENSOR_DICT[el]
            b_e = (c11_e + 2.0 * c12_e) / 3.0
            g_e = float(np.sqrt(c44_e * (c11_e - c12_e) / 2.0))
            denom = 2.0 * (3.0 * b_e + g_e)
            nu_elem.append((3.0 * b_e - 2.0 * g_e) / denom if denom != 0 else None)
        else:
            e = Element(el)
            b = _get_prop(e, "bulk_modulus")
            g = _get_prop(e, "rigidity_modulus")
            if b is not None and g is not None and (3 * b + g) != 0:
                nu_elem.append((3.0 * b - 2.0 * g) / (2.0 * (3.0 * b + g)))
            else:
                nu_elem.append(None)

    if Poisson is not None and all(v is not None for v in nu_elem):
        nu_arr = np.array(nu_elem, dtype=float)
        mismatch_terms_nu = x * (1 - nu_arr / Poisson) ** 2
        Poisson_Var = float(np.sqrt(np.sum(mismatch_terms_nu)))
        Poisson_Delt = float(np.max(mismatch_terms_nu) - np.min(mismatch_terms_nu))
    else:
        Poisson_Var = None
        Poisson_Delt = None

    # Young's modulus E = 9·B·G / (3·B + G)
    if B_GPa is not None and G_GPa is not None and (3 * B_GPa + G_GPa) != 0:
        E_GPa = 9 * B_GPa * G_GPa / (3 * B_GPa + G_GPa)
    else:
        E_GPa = None

    # Allen (or Pauling) electronegativity — composition-weighted
    chi_vals = [_allen_electronegativity(Element(el)) for el in elements]
    if all(v is not None for v in chi_vals):
        Allen_elec = float(np.dot(x, chi_vals))
    else:
        Allen_elec = None

    # Dimensionless configurational entropy S/R = -sum(x_i*ln(x_i))
    x_pos = x[x > 0]
    if len(x_pos) > 0:
        S_conf_R = -float(np.sum(x_pos * np.log(x_pos)))
    else:
        S_conf_R = None

    # Density (g/cm³) — rule of mixtures on molar volumes
    # ρ = M_avg / Σ(x_i · M_i / ρ_i)
    mass_vals = [_get_prop(Element(el), "atomic_mass") for el in elements]
    rho_vals = []
    for el in elements:
        e = Element(el)
        d = _get_prop(e, "density_of_solid")
        if d is not None:
            rho_vals.append(d / 1000.0)                    # kg/m³ → g/cm³
        else:
            vm = _get_prop(e, "molar_volume")
            m = _get_prop(e, "atomic_mass")
            if vm is not None and m is not None and vm > 0:
                rho_vals.append(m / vm)                     # g/mol / (cm³/mol) → g/cm³
            else:
                rho_vals.append(None)
    if all(m is not None for m in mass_vals) and all(r is not None for r in rho_vals):
        M_avg = float(np.dot(x, mass_vals))
        rho_gcc = np.array(rho_vals)
        vol_mix = float(np.dot(x, np.array(mass_vals) / rho_gcc))
        density_g_cm3 = M_avg / vol_mix
    else:
        density_g_cm3 = None

    out = {
        "formula": formula,
        "Tm_K": Tm_K - 250, # BC approximation
        "VEC": VEC,
        "R_Angstrom": R_Angstrom,
        "R_pm": R_pm,
        "R_Var": R_Var,
        "R_Delt": R_Delt,
        "C11_GPa": C11,
        "C12_GPa": C12,
        "C44_GPa": C44,
        "B_GPa": B_GPa,
        "B_Var": B_Var,
        "B_Delt": B_Delt,
        "G_GPa": G_GPa,
        "Poisson": Poisson,
        "Poisson_Var": Poisson_Var,
        "Poisson_Delt": Poisson_Delt,
        "E_GPa": E_GPa,
        "Allen_elec": Allen_elec,
        "S_conf_R": S_conf_R,
        "density_g_cm3": density_g_cm3,
    }

    if return_dict:
        return out
    return tuple(out.values())


def main():
    parser = argparse.ArgumentParser(
        description="Map composition (formula) to melting point, VEC, B, G, Poisson, Allen chi, S_conf (wch3n logic), density, E."
    )
    parser.add_argument("formula", nargs="?", default="NbTaTiZr", help="Chemical formula (e.g. NbTaTiZr or Al20Nb20Ti25Ta20Hf15)")
    parser.add_argument("-T", "--temperature", type=float, default=None, help="Temperature (K); if not set, t = t_fac*Tm (wch3n style)")
    parser.add_argument("-t", "--t-fac", type=float, default=0.9, dest="t_fac", help="t_fac when temperature not set (default 0.9)")
    parser.add_argument("-j", "--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    d = composition_to_features(args.formula, temperature_k=args.temperature, t_fac=args.t_fac)
    if args.json:
        # make JSON-serializable
        out = {k: v for k, v in d.items() if not k.endswith("_note")}
        for k, v in out.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                out[k] = None
        print(json.dumps(out, indent=2))
    else:
        for k, v in d.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
