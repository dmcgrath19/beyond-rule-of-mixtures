import math
import typing
from typing import Final

from pymatgen.core.composition import Composition
from pymatgen.util.typing import SpeciesLike

from lib.physics import ELASTIC_TENSOR_DICT


VEC_DICT: Final[dict[str, int]] = {
    "Al": 3,
    "Co": 9,
    "Cr": 6,
    "Fe": 8,
    "Ni": 10,
    "Mo": 6,
    "W": 6,
    "V": 5,
    "Nb": 5,
    "Ta": 5,
    "Ti": 4,
    "Zr": 4,
    "Hf": 4,
    "Mn": 7,
    "Cu": 11,
    "Si": 4,
    "Re": 7,
}

ATOMIC_RADIUS_PM: Final[dict[str, float]] = {
    "Al": 143.0,
    "Co": 125.0,
    "Cr": 128.0,
    "Fe": 126.0,
    "Ni": 124.0,
    "Mo": 139.0,
    "W": 139.0,
    "V": 134.0,
    "Nb": 146.0,
    "Ta": 146.0,
    "Ti": 147.0,
    "Zr": 160.0,
    "Hf": 159.0,
    "Mn": 127.0,
    "Cu": 128.0,
    "Si": 117.0,
    "Re": 137.0,
}

MELTING_TEMP_K: Final[dict[str, float]] = {
    "Al": 933.0,
    "Co": 1768.0,
    "Cr": 2180.0,
    "Fe": 1811.0,
    "Ni": 1728.0,
    "Mo": 2896.0,
    "W": 3695.0,
    "V": 2183.0,
    "Nb": 2750.0,
    "Ta": 3290.0,
    "Ti": 1941.0,
    "Zr": 2128.0,
    "Hf": 2506.0,
    "Mn": 1519.0,
    "Cu": 1358.0,
    "Si": 1687.0,
    "Re": 3459.0,
}


FEATURE_NAMES = ["R_Var", "R", "B_avgr", "G_avgr", "V_Delt", "VEC_Avg", "Tm_Avg", "R_Delt"]
CORE_FEATURE_COLS = [f"core_{n}" for n in FEATURE_NAMES]
SHELL_FEATURE_COLS = [f"shell_{n}" for n in FEATURE_NAMES]
MICROSTRUCTURE_COLS = CORE_FEATURE_COLS + SHELL_FEATURE_COLS + ["core_phase_fraction_mean"]


def _compute_features_from_fracs(
    elements: list[str], fracs: list[float],
) -> list[float] | None:
    """Shared feature computation for a set of elements and atomic fractions."""
    for el in elements:
        if el not in VEC_DICT or el not in ATOMIC_RADIUS_PM or el not in MELTING_TEMP_K or el not in ELASTIC_TENSOR_DICT:
            return None

    vec_avg = sum(c * VEC_DICT[el] for c, el in zip(fracs, elements))
    r_avg = sum(c * ATOMIC_RADIUS_PM[el] for c, el in zip(fracs, elements))
    tm_avg = sum(c * MELTING_TEMP_K[el] for c, el in zip(fracs, elements))

    b_els = {el: (ELASTIC_TENSOR_DICT[el][0] + 2 * ELASTIC_TENSOR_DICT[el][1]) / 3 for el in elements}
    b_avg = sum(c * b_els[el] for c, el in zip(fracs, elements))

    g_els = {el: math.sqrt(ELASTIC_TENSOR_DICT[el][2] * (ELASTIC_TENSOR_DICT[el][0] - ELASTIC_TENSOR_DICT[el][1]) / 2) for el in elements}
    g_avg = sum(c * g_els[el] for c, el in zip(fracs, elements))

    nu_els = {el: (3 * b_els[el] - 2 * g_els[el]) / (2 * (3 * b_els[el] + g_els[el])) for el in elements}
    nu_avg = sum(c * nu_els[el] for c, el in zip(fracs, elements))

    r_var = sum(c * (ATOMIC_RADIUS_PM[el] - r_avg) ** 2 for c, el in zip(fracs, elements))
    r_delt = math.sqrt(sum(c * (1 - ATOMIC_RADIUS_PM[el] / r_avg) ** 2 for c, el in zip(fracs, elements)))
    if nu_avg == 0:
        v_delt = 0.0
    else:
        v_delt = math.sqrt(sum(c * (1 - nu_els[el] / nu_avg) ** 2 for c, el in zip(fracs, elements)))

    return [r_var, r_avg, b_avg, g_avg, v_delt, vec_avg, tm_avg, r_delt]


def compute_microstructure_features(
    formula: str,
    partition_coeffs: dict[str, float],
    core_phase_fraction: float,
) -> dict[str, float] | None:
    """Compute per-phase compositional features from partition coefficients.

    Derives core and shell phase compositions from nominal composition and
    partition coefficients, then computes the same feature set on each phase.
    Elements missing from partition_coeffs are assumed to have k=1 (no segregation).
    """
    comp = Composition(formula)
    elements = [typing.cast(str, el.name) for el in comp.elements]
    nominal = [comp.get_atomic_fraction(typing.cast(SpeciesLike, el.name)) for el in comp.elements]

    k_vals = [partition_coeffs.get(el, 1.0) for el in elements]

    core_raw = [c * k for c, k in zip(nominal, k_vals)]
    core_sum = sum(core_raw)
    if core_sum == 0:
        return None
    core_fracs = [x / core_sum for x in core_raw]

    f = core_phase_fraction
    if f >= 1.0:
        return None
    shell_raw = [(c_nom - f * c_core) / (1 - f) for c_nom, c_core in zip(nominal, core_fracs)]
    shell_raw = [max(0.0, x) for x in shell_raw]
    shell_sum = sum(shell_raw)
    if shell_sum == 0:
        return None
    shell_fracs = [x / shell_sum for x in shell_raw]

    core_feats = _compute_features_from_fracs(elements, core_fracs)
    shell_feats = _compute_features_from_fracs(elements, shell_fracs)
    if core_feats is None or shell_feats is None:
        return None

    result: dict[str, float] = {}
    for name, val in zip(FEATURE_NAMES, core_feats):
        result[f"core_{name}"] = val
    for name, val in zip(FEATURE_NAMES, shell_feats):
        result[f"shell_{name}"] = val
    return result


def compute_borg_features(formula: str) -> list[float] | None:
    comp = Composition(formula)
    elements = [typing.cast(str, el.name) for el in comp.elements]
    fracs = [comp.get_atomic_fraction(typing.cast(SpeciesLike, el.name)) for el in comp.elements]
    return _compute_features_from_fracs(elements, fracs)
