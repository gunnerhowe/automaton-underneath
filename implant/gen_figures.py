"""Figures for the implant note -> paper/figs/*.pdf. Run: python gen_figures.py"""
import json
import pathlib
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
PS = (0.0, 0.1, 0.5, 1.0)
ARMS = [("P0", "none (+b)", "#7f8c8d"), ("P1", "$-b$", "#27ae60"),
        ("P2", "freeze core", "#2980b9"), ("P3", "lr$\\times$0.1 core", "#8e44ad")]
SEEDS = (0, 1, 2)


def cell(p, arm, s):
    return json.loads((ROOT / "runs_toy" / f"p{int(p*100)}_{arm}_s{s}.json").read_text())


def endvals(p, arm, key):
    out = []
    for s in SEEDS:
        c = cell(p, arm, s)
        out.append([r[key] for r in c["curve"] if key in r][-1])
    return out


def main():
    # --- F1: fate grid ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    for ax, key, ylab, title in [(ax1, "integrity", "S5 integrity (pos 511)",
                                  "(a) implant integrity after continued training"),
                                 (ax2, "par_in", "parity accuracy (pos 31)",
                                  "(b) plasticity on the foreign task")]:
        for j, (arm, lab, c) in enumerate(ARMS):
            meds = [st.median(endvals(p, arm, key)) for p in PS]
            lo = [min(endvals(p, arm, key)) for p in PS]
            hi = [max(endvals(p, arm, key)) for p in PS]
            x = [i + (j - 1.5) * 0.19 for i in range(4)]
            ax.bar(x, meds, 0.19, color=c, label=lab)
            ax.errorbar(x, meds, yerr=[[m - l for m, l in zip(meds, lo)],
                                       [h - m for m, h in zip(meds, hi)]],
                        fmt="none", ecolor="k", lw=0.7, capsize=1.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{int(p*100)}%" for p in PS])
        ax.set_xlabel("implanted-task exercise rate $p$")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=9)
    ax1.axhline(0.2, ls=":", c="k", lw=0.7)
    ax1.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.savefig(FIGS / "fig1_fate.pdf")
    plt.close(fig)

    # --- F2: resurrection ---
    rev = json.loads((ROOT / "reveal_results.json").read_text())
    conds = [(1.0, "P0", "p=100%\nnone"), (1.0, "P2", "p=100%\nfreeze"),
             (0.0, "P2", "p=0%\nfreeze"), (0.0, "P0", "p=0%\nnone")]
    fig, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
    for i, (p, arm, lab) in enumerate(conds):
        rows = [r for r in rev if r["p"] == p and r["arm"] == arm]
        b = st.mean(r["integrity"] for r in rows)
        a = st.mean(r["integrity_bzero"] for r in rows)
        ax.bar(i - 0.19, b, 0.38, color="#7f8c8d", label="trained (as-is)" if i == 0 else None)
        ax.bar(i + 0.19, a, 0.38, color="#27ae60",
               label="$W_b$ zeroed at inference" if i == 0 else None)
        if a > 0.99:
            ax.text(i + 0.19, a + 0.02, "1.000", ha="center", fontsize=8)
    ax.axhline(0.2, ls=":", c="k", lw=0.7)
    ax.text(3.45, 0.22, "chance", fontsize=7, ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[2] for c in conds], fontsize=8)
    ax.set_ylabel("S5 integrity (pos 511)")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("the resurrection: concealed vs.\\ eroded", fontsize=9)
    fig.savefig(FIGS / "fig2_resurrection.pdf")
    plt.close(fig)

    # --- F3: LM tier ---
    create = json.loads((ROOT / "runs_lm" / "skill" / "create_log.json").read_text())["log"]
    warm = json.loads((ROOT / "runs_lm" / "explore_warmx.json").read_text())
    cold = json.loads((ROOT / "runs_lm" / "explore_cold.json").read_text())
    fig, ax = plt.subplots(figsize=(4.6, 2.6), constrained_layout=True)
    wsteps = [r["step"] / 1000 for r in create] + [(5000 + r["step"]) / 1000 for r in warm
                                                  if r["step"] > 0]
    wt = [r["toggle_8"] for r in create] + [r["toggle_8"] for r in warm if r["step"] > 0]
    ax.plot(wsteps, wt, "s-", ms=3, c="#c0392b",
            label="pretrained init (create + extension)")
    ax.plot([r["step"] / 1000 for r in cold], [r["toggle_8"] for r in cold], "o-", ms=3,
            c="#27ae60", label="random init (identical harness)")
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.text(14.8, 0.52, "chance", fontsize=7, ha="right")
    ax.annotate("emerges by step 500", xy=(0.5, 1.0), xytext=(3.4, 0.86), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("fine-tuning step (thousands)")
    ax.set_ylabel("toggle-8 accuracy")
    ax.set_ylim(0.4, 1.1)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    ax.set_title("warm-start skill-acquisition resistance (30M LM)", fontsize=9)
    fig.savefig(FIGS / "fig3_lm.pdf")
    plt.close(fig)
    print("wrote 3 figures to paper/figs/")


if __name__ == "__main__":
    main()
