"""Load the public Borg spreadsheet; no external database is used."""
from pathlib import Path

import numpy as np
import pandas as pd

from lib.features import compute_borg_features, compute_microstructure_features, MICROSTRUCTURE_COLS

BORG_NUMERIC_COLS = ["R Var", "R", "B_avgr", "G_avgr", "V_Delt", "VEC Avg", "Tm Avg", "R_Delt"]
PROC_MAP = {"CAST": 0, "ANNEAL": 1, "WROUGHT": 2, "OTHER": 3}
FEATURE_COLS = BORG_NUMERIC_COLS + ["processing_code"]
VELA_FEATURE_COLS = [
    "Density Avg",
    "Tm Avg",
    "Pugh_Ratio_avgr",
    "V_avgr",
    "B_avgr",
    "G_avgr",
    "E_avgr",
    "C11",
    "C12",
    "C44",
    "VEC Avg",
    "R",
    "Sconf",
    "B_Delt",
    "V_Delt",
    "G_Delt",
    "R_Delt",
    "G Var",
    "R Var",
    "V Var",
    "B Var",
]
VELA_FORCED_FEATURE_COLS = ["test_temperature_K", "processing_code", "test_type_code"]
RADICAL_VELA_FEATURE_COLS = list(BORG_NUMERIC_COLS)
RADICAL_VELA_FORCED_FEATURE_COLS = ["processing_code"]
CELSIUS_TO_KELVIN = 273.15
DEFAULT_TEST_TEMPERATURE_K = 298.15

RAW_DATA_DIR = Path(__file__).resolve().parents[1]
BORG_DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "borg" / "Borg_Dataset_PUB.xlsx"


def _processing_code(value: str | float | None) -> int:
    if value == "CAST":
        return PROC_MAP["CAST"]
    if value == "ANNEAL":
        return PROC_MAP["ANNEAL"]
    if value == "WROUGHT":
        return PROC_MAP["WROUGHT"]
    return PROC_MAP["OTHER"]


def _test_type_code(value: str | float | None) -> int:
    if value == "C":
        return 0
    if value == "T":
        return 1
    return 2



def load_borg() -> pd.DataFrame:
    """Load Borg dataset.

    Test temperature is a fixed temperature at which hardness was measured.
    Missing values are assumed to be room temperature (298.15 K).
    """
    df = pd.read_excel(BORG_DATA_FILE)
    df = df.dropna(subset=BORG_NUMERIC_COLS + ["HV", "FORMULA", "Processing method"])
    df = df.rename(columns={"FORMULA": "formula"})
    df["HV"] = df["HV"].astype(float)
    df["test_temperature_K"] = df["Test temperature"].apply(
        lambda t: float(t) + CELSIUS_TO_KELVIN if pd.notna(t) else DEFAULT_TEST_TEMPERATURE_K
    )
    df["processing_code"] = df["Processing method"].map(_processing_code)
    df["source"] = "Borg"
    df["sample_id"] = ""
    df = df.rename(columns={
        "Reference ID": "reference_id",
        "Processing method": "processing_method",
        "Microstructure": "microstructure",
    })
    return df.reset_index(drop=True)

def load_borg_yield_strength() -> pd.DataFrame:
    """Load the single-phase BCC Borg yield-strength subset used by Vela et al."""
    df = pd.read_excel(BORG_DATA_FILE)
    required = VELA_FEATURE_COLS + ["YS (MPa)", "FORMULA", "Processing method", "Type of test", "Microstructure"]
    df = df.dropna(subset=required)
    df = df[df["Microstructure"] == "BCC"].copy()
    df["HV"] = pd.to_numeric(df["HV"], errors="coerce")
    df["YS_MPa"] = pd.to_numeric(df["YS (MPa)"], errors="coerce")
    df["test_temperature_K"] = df["Test temperature"].apply(
        lambda t: float(t) + CELSIUS_TO_KELVIN if pd.notna(t) else DEFAULT_TEST_TEMPERATURE_K
    )
    df["processing_code"] = df["Processing method"].map(_processing_code)
    df["test_type_code"] = df["Type of test"].map(_test_type_code)
    df["alloy_series_id"] = (
        df["FORMULA"].astype(str)
        + " | "
        + df["Processing method"].astype(str)
        + " | "
        + df["Type of test"].astype(str)
    )
    df = df.rename(columns={
        "FORMULA": "formula",
        "Reference ID": "reference_id",
        "Processing method": "processing_method",
        "Type of test": "test_type",
    })
    cols = [
        "reference_id",
        "formula",
        "processing_method",
        "processing_code",
        "test_type",
        "test_type_code",
        "Microstructure",
        "test_temperature_K",
        "HV",
        "YS_MPa",
        "alloy_series_id",
    ] + VELA_FEATURE_COLS
    return df[cols].reset_index(drop=True)
