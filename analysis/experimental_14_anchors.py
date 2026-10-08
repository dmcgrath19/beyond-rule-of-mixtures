"""Learned and linear misfit volumes at the 14 validation alloys against MLIP partial molar misfits, for tabulated and MLIP volume anchors."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import predict_misfit  # noqa: E402
from borg_experiments import BETA, Data  # noqa: E402
from utils.paper_training import fit_fold  # noqa: E402
from pool_diagnostic import ALLOYS_14, to_basis  # noqa: E402
from volume_surface import BASIS  # noqa: E402

warnings.filterwarnings("ignore")


def head_dv(fit, comp_borg: np.ndarray) -> np.ndarray:
    """dV_i^eff at the given compositions, on the Borg element basis."""
    head = fit["model"].mean_module.sigma_model
    c = torch.tensor(comp_borg, dtype=torch.float64)
    with torch.no_grad():
        return head.diagnostics(torch.ones(len(c), dtype=torch.float64), c).delta_volumes.numpy()


def head_sigma(fit, comp_borg: np.ndarray) -> np.ndarray:
    head = fit["model"].mean_module.sigma_model
    c = torch.tensor(comp_borg, dtype=torch.float64)
    with torch.no_grad():
        return head.diagnostics(torch.ones(len(c), dtype=torch.float64), c).sigma.numpy()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("-o", "--out", default="results/experimental_14_anchors.csv")
    a = p.parse_args(argv)

    d = Data()
    quin_to_borg = [d.basis.index(el) for el in BASIS]

    # --- the 14 alloys, on both the quinary and the Borg basis ---
    labels = list(ALLOYS_14)
    c_quin = np.array([to_basis(ALLOYS_14[k]) for k in labels])
    c_borg = np.zeros((len(labels), len(d.basis)))
    for j, i in enumerate(quin_to_borg):
        c_borg[:, i] = c_quin[:, j]

    # --- reference quantities from the pooled MLIP surface ---
    surf, fits, model = predict_misfit.build()
    ref_rows = []
    for lab, c in zip(labels, c_quin):
        for r in predict_misfit.evaluate(surf, fits, model, c):
            ref_rows.append(dict(alloy=lab, element=r["element"], x=r["x"],
                                 dV_rom=r["dV_rom"],
                                 dV_pmv=r["dV_pmv"], dV_pmv_unc=r["dV_pmv_unc"],
                                 sigma_rom=r["sigma_rom"],
                                 sigma_pmv=r["sigma_pmv"], supported=r["supported"]))
    ref = pd.DataFrame(ref_rows)
    current_volumes = d.v_base(anchor="current").numpy()
    current_mean = c_borg @ current_volumes
    current_delta = current_volumes[None, :] - current_mean[:, None]
    current_sigma = (c_borg * current_delta**2).sum(axis=1)
    ref["dV_current"] = [current_delta[labels.index(r.alloy), d.basis.index(r.element)] for r in ref.itertuples()]
    ref["sigma_current"] = [current_sigma[labels.index(r.alloy)] for r in ref.itertuples()]

    # --- train the head on all of Borg under each anchor choice ---
    learned: dict[str, np.ndarray] = {}
    sigmas: dict[str, np.ndarray] = {}
    for anchor in ("current", "mlip"):
        dvs, sgs = [], []
        for s in range(a.seeds):
            fit = fit_fold(
                d.Xk, d.phys, d.comp, d.y, d.v_base(anchor=anchor),
                variant="effective_volume", beta=BETA, seed=s,
            )
            dvs.append(head_dv(fit, c_borg))
            sgs.append(head_sigma(fit, c_borg))
        learned[anchor] = np.mean(dvs, axis=0)
        sigmas[anchor] = np.mean(sgs, axis=0)

    for anchor in ("current", "mlip"):
        col = f"dV_eff_{anchor}"
        ref[col] = [
            learned[anchor][labels.index(r.alloy), d.basis.index(r.element)]
            for r in ref.itertuples()
        ]

    print("=" * 96)
    print("LEARNED vs LINEAR MISFIT VOLUMES AT THE 14 EXPERIMENTAL ALLOYS")
    print("=" * 96)
    print(f"alloys {len(labels)}, alloy-element rows {len(ref)}, "
          f"all data-supported: {bool(ref.supported.all())}")
    print(f"head trained on all {len(d.y)} Borg rows, beta={BETA}, "
          f"{a.seeds} seeds averaged")

    est = {
        "dV_current (linear, Ti 3.26 A)": "dV_current",
        "dV_eff (learned, Ti 3.26 A)": "dV_eff_current",
        "dV_rom (linear, MLIP)": "dV_rom",
        "dV_eff (learned, MLIP)": "dV_eff_mlip",
    }

    def block(sub: pd.DataFrame, title: str) -> None:
        print(f"\n{title}  (n={len(sub)})")
        print(f"  {'estimator':<30} {'corr':>7} {'mean|gap|':>10} "
              f"{'median|gap|':>12} {'sign ok':>9}")
        print("  " + "-" * 72)
        for name, col in est.items():
            g = (sub[col] - sub.dV_pmv).abs()
            sign = (np.sign(sub[col]) == np.sign(sub.dV_pmv)).sum()
            r = np.corrcoef(sub[col], sub.dV_pmv)[0, 1] if len(sub) > 2 else np.nan
            print(f"  {name:<30} {r:+7.3f} {g.mean():10.3f} {g.median():12.3f} "
                  f"{sign:6d}/{len(sub)}")

    block(ref, "ALL ELEMENTS")
    block(ref[ref.element != "Ti"], "EXCLUDING Ti")
    block(ref[ref.element == "Ti"], "Ti ONLY")

    print("\nper-element mean |gap| to the partial molar reference (A^3/atom)")
    per = ref.groupby("element").apply(
        lambda s: pd.Series({n: (s[c] - s.dV_pmv).abs().mean() for n, c in est.items()}
                            | {"n": len(s)})
    )
    print(per.to_string(float_format=lambda v: f"{v:.3f}"))

    # --- the misfit parameter itself, which is what reaches the hardness ---
    sig = ref.groupby("alloy").first()
    sig["sigma_eff_current"] = [sigmas["current"][labels.index(i)] for i in sig.index]
    sig["sigma_eff_mlip"] = [sigmas["mlip"][labels.index(i)] for i in sig.index]
    print("\nreduced misfit parameter Sigma (A^6), ratio to the partial molar value")
    rat = pd.DataFrame({
        "Sigma_pmv": sig.sigma_pmv,
        "current/pmv": sig.sigma_current / sig.sigma_pmv,
        "eff_current/pmv": sig.sigma_eff_current / sig.sigma_pmv,
        "rom/pmv": sig.sigma_rom / sig.sigma_pmv,
        "eff_mlip/pmv": sig.sigma_eff_mlip / sig.sigma_pmv,
    })
    print(rat.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nmedian ratio to Sigma_pmv: "
          + "  ".join(f"{c} {rat[c].median():.3f}" for c in rat.columns[1:]))

    ref.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({len(ref)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
