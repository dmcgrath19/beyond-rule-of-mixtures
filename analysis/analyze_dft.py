"""Compare the DFT equation-of-state volumes in data/dft/ with the MLIP volumes.

For every *_eos.json the Birch--Murnaghan equation of state is refitted to the stored
(volume, energy) points, the equilibrium volume per atom is compared with the value cached
in the file and with the MLIP volume at the same composition, and the table is written to
results/dft_validation.csv.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]


def birch_murnaghan(v, e0, v0, b0, b1):
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((eta - 1.0) ** 3 * b1 + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


def refit_v0(points):
    v = np.array([p["final_volume_A3"] for p in points], dtype=float)
    e = np.array([p["energy_eV"] for p in points], dtype=float)
    a, b, c = np.polyfit(v, e, 2)
    v0, e0 = -b / (2 * a), np.polyval([a, b, c], -b / (2 * a))
    b0 = 2 * a * v0  # eV/A^3
    popt, _ = curve_fit(birch_murnaghan, v, e, p0=[e0, v0, b0, 4.0], maxfev=20000)
    return float(popt[1])


def main() -> None:
    rows = []
    for path in sorted((ROOT / "data/dft").glob("*/*_eos.json")):
        d = json.loads(path.read_text())
        v_refit = refit_v0(d["eos_points"]) / d["n_atoms"]
        v_cached = d["volume_per_atom_A3"]
        v_mlip = d["mlip_volume_per_atom_A3"]
        rows.append({
            "composition": d["requested_composition"],
            "dft_volume_A3_per_atom": v_refit,
            "cached_dft_volume_A3_per_atom": v_cached,
            "mlip_volume_A3_per_atom": v_mlip,
            "absolute_gap_A3_per_atom": abs(v_refit - v_mlip),
            "refit_gap_A3_per_atom": abs(v_refit - v_cached),
        })
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results/dft_validation.csv", index=False)
    print(out.to_string(index=False))
    gap = out.absolute_gap_A3_per_atom
    print(f"\nmean |V_DFT - V_MLIP| = {gap.mean():.3f} A^3/atom, max = {gap.max():.3f}")


if __name__ == "__main__":
    main()
