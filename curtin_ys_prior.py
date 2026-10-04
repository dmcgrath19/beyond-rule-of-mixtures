"""Workflow for computing yield strength of HEAs."""

import typing
from typing import Final

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core.composition import Composition
from pymatgen.util.typing import SpeciesLike


class YieldStrengthResult(BaseModel):
    """Result of the yield strength calculation."""

    misfit_volumes: dict[str, float] = Field(description="Misfit volumes for each element.")
    yield_strength_at_zero_kelvin: float = Field(description="Yield stress in MPa at 0 K.")
    activation_energy: float = Field(description="Activation energy in eV.")
    yield_strength_at_temperature: float = Field(description="Yield strength at temperature in MPa.")
    C11_rom: float = Field(description="C11 ROM in GPa.")
    C12_rom: float = Field(description="C12 ROM in GPa.")
    C44_rom: float = Field(description="C44 ROM in GPa.")
    bulk_modulus_rom: float = Field(description="Bulk modulus ROM in GPa.")
    shear_modulus_rom: float = Field(description="Shear modulus ROM in GPa.")
    poisson: float = Field(description="Poisson's ratio.")
    operating_temperature: float = Field(description="Operating temperature in K.")


ELASTIC_TENSOR_DICT: Final[dict[str, list[float]]] = {
    "Al": [107, 61, 28],
    "Co": [307, 165, 75],  # hcp reduced to [C11, C12, C44]
    "Cr": [350, 68, 101],
    "Fe": [226, 140, 116],
    "Ni": [246, 147, 124],
    "Mo": [463, 169, 109],
    "W": [522, 204, 161],
    "V": [227, 116, 47],
    "Nb": [246, 134, 29],
    "Ta": [267, 161, 87],
    "Ti": [162, 92, 47],  # hcp reduced
    "Zr": [144, 72, 33],  # hcp reduced
    "Hf": [176, 77, 51],  # hcp reduced
    "Mn": [246, 138, 110],  # approximate (complex α-Mn structure)  # noqa: RUF003
    "Cu": [168, 121, 75],
    "Si": [167, 65, 80],  # diamond cubic
    "Re": [591, 361, 162],  # hcp reduced
}

LATTICE_CONSTANTS_BCC_EXP: Final[dict[str, float]] = {
    # --- Stable bcc phases ---
    "Cr": 2.884,  # experimental
    "Fe": 2.8665,  # α-Fe (bcc)  # noqa: RUF003
    "Mo": 3.147,  # experimental
    "W": 3.1652,  # experimental
    "V": 3.03,  # experimental
    "Nb": 3.300,  # experimental
    "Ta": 3.301,  # experimental
    # --- Metastable / hypothetical bcc phases ---
    "Al": 3.24,  # metastable bcc-Al (high-T phase extrapolated)
    "Co": 2.82,  # bcc-Co (metastable, from thin films)
    "Ni": 2.88,  # bcc-Ni (metastable, from sputtered films)
    "Cu": 2.89,  # bcc-Cu (metastable, thin-film phase)
    "Ti": 3.32,  # β-Ti (stable at >1155 K, metastable at RT)
    "Zr": 3.57,  # β-Zr (stable >1135 K, metastable at RT)
    "Hf": 3.53,  # β-Hf (stable >2030 K, metastable at RT)
    "Mn": 2.99,  # β-Mn approximated to bcc-like phase
    "Si": 2.83,  # metastable bcc-Si (theoretical)
    "Re": 3.11,  # metastable bcc-Re (from high-P experiments)
}


def compute_yield_strength(
    composition: Composition,
    operating_temperature: float,
) -> YieldStrengthResult:
    """Compute the yield strength of a HEA."""

    # Calculate equilibrium volume as linear combination of elemental BCC volumes
    equilibrium_volume: float = sum(
        composition.get_atomic_fraction(typing.cast(SpeciesLike, el.name)) * (LATTICE_CONSTANTS_BCC_EXP[typing.cast(str, el.name)] ** 3) / 2
        for el in composition.elements
    )

    # Calculate misfit volumes as difference between elemental BCC volumes and equilibrium volume.
    misfit_volumes: dict[str, float] = {}
    for element in composition.elements:
        misfit_volumes[typing.cast(str, element.name)] = (LATTICE_CONSTANTS_BCC_EXP[typing.cast(str, element.name)] ** 3) / 2 - equilibrium_volume

    # Get the weighted average reduced misfit volume fot the composition.
    reduced_misfit_volumes: float = sum(
        [
            c * (misfit_volumes[typing.cast(str, e.name)]) ** 2
            for c, e in zip(
                [composition.get_atomic_fraction(typing.cast(SpeciesLike, x.name)) for x in composition.elements], composition.elements, strict=False
            )
        ]  # noqa: E501
    )

    # Get the weighted average elastic constants for the composition.
    C11_rom: float = sum(
        [
            composition.get_atomic_fraction(typing.cast(SpeciesLike, el.name)) * ELASTIC_TENSOR_DICT[typing.cast(str, el.name)][0]
            for el in composition.elements
        ]
    )
    C12_rom: float = sum(
        [
            composition.get_atomic_fraction(typing.cast(SpeciesLike, el.name)) * ELASTIC_TENSOR_DICT[typing.cast(str, el.name)][1]
            for el in composition.elements
        ]
    )
    C44_rom: float = sum(
        [
            composition.get_atomic_fraction(typing.cast(SpeciesLike, el.name)) * ELASTIC_TENSOR_DICT[typing.cast(str, el.name)][2]
            for el in composition.elements
        ]
    )

    # Get the weighted average stiffness properties for the composition.
    bulk_modulus_rom: float = (C11_rom + 2 * C12_rom) / 3
    shear_modulus_rom: float = np.sqrt(C44_rom * (C11_rom - C12_rom) / 2)
    poisson: float = (3 * bulk_modulus_rom - 2 * shear_modulus_rom) / (2 * (3 * bulk_modulus_rom + shear_modulus_rom))
    lattice_parameter: float = (equilibrium_volume * 2) ** (1 / 3)
    burgers_magnitude: float = lattice_parameter * np.sqrt(3) / 2

    # Set to `3` for polycrystalline BCC.
    taylor_factor: float = 3

    # Yield strength in MPa scaled by taylor factor.
    yield_stress: float = (
        0.04
        * (1 / 12) ** (-1 / 3)
        * shear_modulus_rom
        # * elastic_tensor["derived_properties"]["g_voigt"]  # Voight-averaged shear modulus; severely underpredicted by DFT/MLIP
        * ((1 + poisson) / (1 - poisson)) ** (4 / 3)
        * (reduced_misfit_volumes / (burgers_magnitude**6)) ** (2 / 3)
        * 1000
        * taylor_factor
    )

    # Conversion factor from eV to GPa.
    eV_to_GPa: float = 160.21766208

    # Boltzmann constant in eV / K.
    kb: float = 8.617333262e-5

    # Activation energy in eV.
    activation_energy: float = (
        2.0
        * (1 / 12) ** (1 / 3)
        * shear_modulus_rom
        * burgers_magnitude**3
        * ((1 + poisson) / (1 - poisson)) ** (2 / 3)
        * (reduced_misfit_volumes / burgers_magnitude**6) ** (1 / 3)
        / eV_to_GPa
    )

    # Get the reduced yield strength at higher operating temperature.
    strength_T: float = yield_stress * np.exp(-1 / 0.55 * (((kb * operating_temperature) / activation_energy) * np.log(10**4 / 10**-3)) ** 0.91)

    return YieldStrengthResult(
        misfit_volumes=misfit_volumes,
        yield_strength_at_zero_kelvin=yield_stress,
        activation_energy=activation_energy,
        yield_strength_at_temperature=strength_T,
        operating_temperature=operating_temperature,
        C11_rom=C11_rom,
        C12_rom=C12_rom,
        C44_rom=C44_rom,
        bulk_modulus_rom=bulk_modulus_rom,
        shear_modulus_rom=shear_modulus_rom,
        poisson=poisson,
    )