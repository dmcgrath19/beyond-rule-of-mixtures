"""Plot the Borg grouped results and archived sample-efficiency results."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/"results/figures";out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/"results/grouped_validation/uncertainty/cv_summary.csv")
    groups=["random","formula","reference","system"]
    fig,ax=plt.subplots(figsize=(7,4.5))
    for model,g in d.groupby("model"):
        g=g.set_index("group").loc[groups]
        ax.errorbar(range(4),g.mae_hv,yerr=g.mae_hv_sd,marker="o",capsize=3,label=model)
    ax.set_xticks(range(4),["Shuffled","Formula","Publication","Chemical system"])
    ax.set_ylabel("MAE (HV)");ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/"borg_grouped_mae.png",dpi=180);plt.close(fig)
    d=pd.read_csv(ROOT/"results/learning_curve/learning_curve_150.csv")
    fig,ax=plt.subplots(figsize=(6,4))
    for model,g in d.groupby("arm"):
        ax.errorbar(g.train_n,g.mae_mean,yerr=g.mae_sd/g.repeats**.5,marker="o",capsize=3,label=model)
    ax.set_xlabel("Training rows");ax.set_ylabel("Held-out MAE (HV)");ax.legend(fontsize=8)
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/"learning_curve_150.png",dpi=180);plt.close(fig)
    print(out)
if __name__=="__main__":main()
