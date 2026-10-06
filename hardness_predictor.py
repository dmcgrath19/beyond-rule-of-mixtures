"""Train on public Borg and predict hardness from user-supplied compositions.

Both training and prediction recompute descriptors through the same function.
The reported grouped-result engine is separate (run.py).
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from pymatgen.core.composition import Composition
from composition_features import composition_to_features
from effective_volume_gp import (BORG_XLSX, BORG_NUM_COLS, PROC_MAP, DT, build_borg,
                                curtin_intermediates, fit_fold, predict)
from curtin_ys_prior import LATTICE_CONSTANTS_BCC_EXP
_PROC_MAP = PROC_MAP
_curtin_intermediates = curtin_intermediates
_KERNEL_KEYS = ["R_Var", "R_pm", "B_GPa", "G_GPa", "Poisson_Delt", "VEC", "Tm_K", "R_Delt"]
def _comp_kernel_feats(formula):
    feats = composition_to_features(formula, return_dict=True)
    values = [feats.get(k) for k in _KERNEL_KEYS]
    if any(v is None for v in values):
        return None
    return [float(v) for v in values]

def build_borg_recomputed(basis):
    df = pd.read_excel(BORG_XLSX).dropna(subset=BORG_NUM_COLS + ["HV", "FORMULA", "Processing method"])
    X, P, C, Y, F = [], [], [], [], []
    for _, row in df.iterrows():
        formula = str(row.FORMULA)
        temp = float(row["Test temperature"]) + 273.15 if pd.notna(row["Test temperature"]) else 298.15
        ph, feats = curtin_intermediates(formula, temp), _comp_kernel_feats(formula)
        if ph is None or feats is None:
            continue
        comp = Composition(formula)
        C.append([comp.get_atomic_fraction(el) for el in basis])
        X.append(feats + [PROC_MAP.get(row["Processing method"], 3), temp])
        P.append(ph); Y.append(float(row.HV)); F.append(formula)
    return np.asarray(X), np.asarray(P), np.asarray(C), np.asarray(Y), F

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--composition", nargs="+", required=True)
    p.add_argument("--variant", choices=["analytic", "effective_volume", "constant_mean"], default="effective_volume")
    p.add_argument("--temperature", type=float, default=298.15, help="kelvin")
    p.add_argument("--processing", choices=list(PROC_MAP), default="CAST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n1", type=int, default=150);p.add_argument("--n2", type=int, default=500)
    p.add_argument("--output", default="results/user_predictions.csv")
    args=p.parse_args();torch.set_num_threads(1)
    *_, basis=build_borg(); X,P,C,Y,_=build_borg_recomputed(basis)
    vb=torch.tensor([LATTICE_CONSTANTS_BCC_EXP[el]**3/2 for el in basis], dtype=DT)
    fit=fit_fold(X,P,C,Y,vb,variant=args.variant,seed=args.seed,n1=args.n1,n2=args.n2)
    xx,pp,cc=[],[],[]
    for formula in args.composition:
        comp=Composition(formula)
        unknown=set(comp.get_el_amt_dict())-set(basis)
        if unknown:raise ValueError(f"Elements outside Borg training basis: {sorted(unknown)}")
        feats,ph=_comp_kernel_feats(formula),curtin_intermediates(formula,args.temperature)
        if feats is None or ph is None:raise ValueError(f"Unsupported composition: {formula}")
        xx.append(feats+[PROC_MAP[args.processing],args.temperature]);pp.append(ph)
        cc.append([comp.get_atomic_fraction(el) for el in basis])
    mu,sd=predict(fit,np.asarray(xx),np.asarray(pp),np.asarray(cc))
    out=pd.DataFrame({"formula":args.composition,"mean_hv":mu,"predictive_sd_hv":sd})
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);out.to_csv(args.output,index=False)
    print(out.to_string(index=False))
if __name__=="__main__":main()
