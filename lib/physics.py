import math
import typing
from typing import Final

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core.composition import Composition
from pymatgen.util.typing import SpeciesLike


ALPHA_EDGE: Final[float] = 1.0 / 12.0
ALPHA_SCREW: Final[float] = 0.5


class YieldStrengthResult(BaseModel):
    misfit_volumes: dict[str, float] = Field(description="Misfit volumes for each element.")
    sigma_rom: float = Field(description="ROM reduced misfit volume baseline used by the Curtin model.")
    reduced_misfit_volumes: float = Field(description="Concentration-weighted sum of squared misfit volumes.")
    yield_strength_at_zero_kelvin: float = Field(description="Edge dislocation yield stress in MPa at 0 K.")
    activation_energy: float = Field(description="Edge dislocation activation energy in eV.")
    yield_strength_at_temperature: float = Field(description="Edge dislocation yield strength at temperature in MPa.")
    yield_strength_at_zero_kelvin_screw: float = Field(description="Screw dislocation yield stress in MPa at 0 K.")
    activation_energy_screw: float = Field(description="Screw dislocation activation energy in eV.")
    yield_strength_at_temperature_screw: float = Field(description="Screw dislocation yield strength at temperature in MPa.")
    controlling_mechanism: str = Field(description="Which mechanism gives lower stress at temperature: 'edge' or 'screw'.")
    C11_rom: float = Field(description="C11 ROM in GPa.")
    C12_rom: float = Field(description="C12 ROM in GPa.")
    C44_rom: float = Field(description="C44 ROM in GPa.")
    bulk_modulus_rom: float = Field(description="Bulk modulus ROM in GPa.")
    shear_modulus_rom: float = Field(description="Shear modulus ROM in GPa.")
    poisson: float = Field(description="Poisson's ratio.")
    burgers_magnitude: float = Field(description="Burgers vector magnitude in Angstrom.")
    operating_temperature: float = Field(description="Operating temperature in K.")


ELASTIC_TENSOR_DICT: Final[dict[str, list[float]]] = {
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

LATTICE_CONSTANTS_BCC_EXP: Final[dict[str, float]] = {
    "Cr": 2.884,
    "Fe": 2.8665,
    "Mo": 3.147,
    "W": 3.1652,
    "V": 3.03,
    "Nb": 3.300,
    "Ta": 3.301,
    "Al": 3.24,
    "Co": 2.82,
    "Ni": 2.88,
    "Cu": 2.89,
    "Ti": 3.32,
    "Zr": 3.57,
    "Hf": 3.53,
    "Mn": 2.99,
    "Si": 2.83,
    "Re": 3.11,
}
SUPPORTED_ELEMENTS: Final[tuple[str, ...]] = tuple(LATTICE_CONSTANTS_BCC_EXP)
ELEMENTAL_BCC_VOLUMES: Final[np.ndarray] = np.array(
    [(LATTICE_CONSTANTS_BCC_EXP[element] ** 3) / 2 for element in SUPPORTED_ELEMENTS],
    dtype=float,
)


def composition_supported(composition: Composition) -> bool:
    return all(
        typing.cast(str, el.name) in ELASTIC_TENSOR_DICT
        and typing.cast(str, el.name) in LATTICE_CONSTANTS_BCC_EXP
        for el in composition.elements
    )


def composition_fraction_array(composition: Composition) -> np.ndarray:
    fractions = [
        composition.get_atomic_fraction(typing.cast(SpeciesLike, element))
        if element in composition
        else 0.0
        for element in SUPPORTED_ELEMENTS
    ]
    return np.array(fractions, dtype=float)


def composition_fraction_array_from_formula(formula: str) -> np.ndarray:
    return composition_fraction_array(Composition(formula))


def compute_yield_strength(
    composition: Composition,
    operating_temperature: float,
) -> YieldStrengthResult:
    equilibrium_volume_rom: float = sum(
        composition.get_atomic_fraction(typing.cast(SpeciesLike, el.name))
        * (LATTICE_CONSTANTS_BCC_EXP[typing.cast(str, el.name)] ** 3)
        / 2
        for el in composition.elements
    )

    misfit_volumes: dict[str, float] = {}
    for element in composition.elements:
        key = typing.cast(str, element.name)
        misfit_volumes[key] = (LATTICE_CONSTANTS_BCC_EXP[key] ** 3) / 2 - equilibrium_volume_rom

    sigma_rom: float = sum(
        c * (misfit_volumes[typing.cast(str, e.name)]) ** 2
        for c, e in zip(
            [composition.get_atomic_fraction(typing.cast(SpeciesLike, x.name)) for x in composition.elements],
            composition.elements,
            strict=False,
        )
    )

    C11_rom: float = sum(
        composition.get_atomic_fraction(typing.cast(SpeciesLike, el.name))
        * ELASTIC_TENSOR_DICT[typing.cast(str, el.name)][0]
        for el in composition.elements
    )
    C12_rom: float = sum(
        composition.get_atomic_fraction(typing.cast(SpeciesLike, el.name))
        * ELASTIC_TENSOR_DICT[typing.cast(str, el.name)][1]
        for el in composition.elements
    )
    C44_rom: float = sum(
        composition.get_atomic_fraction(typing.cast(SpeciesLike, el.name))
        * ELASTIC_TENSOR_DICT[typing.cast(str, el.name)][2]
        for el in composition.elements
    )

    bulk_modulus_rom: float = (C11_rom + 2 * C12_rom) / 3
    shear_modulus_rom: float = np.sqrt(C44_rom * (C11_rom - C12_rom) / 2)
    poisson: float = (3 * bulk_modulus_rom - 2 * shear_modulus_rom) / (2 * (3 * bulk_modulus_rom + shear_modulus_rom))
    lattice_parameter: float = (equilibrium_volume_rom * 2) ** (1 / 3)
    burgers_magnitude: float = lattice_parameter * np.sqrt(3) / 2
    taylor_factor: float = 3.0

    def _yield_stress_0k(alpha: float) -> float:
        return (
            0.04
            * alpha ** (-1 / 3)
            * shear_modulus_rom
            * ((1 + poisson) / (1 - poisson)) ** (4 / 3)
            * (sigma_rom / (burgers_magnitude**6)) ** (2 / 3)
            * 1000
            * taylor_factor
        )

    def _activation_energy(alpha: float) -> float:
        return (
            2.0
            * alpha ** (1 / 3)
            * shear_modulus_rom
            * burgers_magnitude**3
            * ((1 + poisson) / (1 - poisson)) ** (2 / 3)
            * (sigma_rom / burgers_magnitude**6) ** (1 / 3)
            / 160.21766208
        )

    def _thermal_softening(ys_0k: float, delta_eb: float) -> float:
        return float(ys_0k * np.exp(
            -1 / 0.55 * (((8.617333262e-5 * operating_temperature) / delta_eb) * np.log(10**4 / 10**-3)) ** 0.91
        ))

    ys_0k_edge = _yield_stress_0k(ALPHA_EDGE)
    delta_eb_edge = _activation_energy(ALPHA_EDGE)
    ys_t_edge = _thermal_softening(ys_0k_edge, delta_eb_edge)

    ys_0k_screw = _yield_stress_0k(ALPHA_SCREW)
    delta_eb_screw = _activation_energy(ALPHA_SCREW)
    ys_t_screw = _thermal_softening(ys_0k_screw, delta_eb_screw)

    controlling = "screw" if ys_t_screw < ys_t_edge else "edge"

    return YieldStrengthResult(
        misfit_volumes=misfit_volumes,
        sigma_rom=sigma_rom,
        reduced_misfit_volumes=sigma_rom,
        yield_strength_at_zero_kelvin=ys_0k_edge,
        activation_energy=delta_eb_edge,
        yield_strength_at_temperature=ys_t_edge,
        yield_strength_at_zero_kelvin_screw=ys_0k_screw,
        activation_energy_screw=delta_eb_screw,
        yield_strength_at_temperature_screw=ys_t_screw,
        controlling_mechanism=controlling,
        operating_temperature=operating_temperature,
        C11_rom=C11_rom,
        C12_rom=C12_rom,
        C44_rom=C44_rom,
        bulk_modulus_rom=bulk_modulus_rom,
        shear_modulus_rom=shear_modulus_rom,
        poisson=poisson,
        burgers_magnitude=burgers_magnitude,
    )


def curtin_intermediates(formula: str, temperature: float) -> tuple[float, float, float, float, float]:
    """Return (shear_modulus, poisson, burgers, sigma_rom, temperature)."""
    r = compute_yield_strength(Composition(formula), temperature)
    return r.shear_modulus_rom, r.poisson, r.burgers_magnitude, r.sigma_rom, temperature
