"""Figures for paper 4 -> paper/figs/*.pdf. All data from facts.json + runs_wide/*.json.
Run: python gen_figures.py"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def cell(c):
    return json.loads((ROOT / "runs_wide" / f"{c}.json").read_text())


def main():
    F = json.loads((ROOT / "facts.json").read_text())

    # --- F1: construction obstacle + clean widened implant ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    heads = [str(h) for h in range(8)]
    costs = [F["head_cost_pct"][h] for h in heads]
    bars = ax1.bar(range(8), [min(c, 30) for c in costs],
                   color=["#27ae60" if c < 1 else "#e67e22" if c < 20 else "#c0392b" for c in costs])
    for i, c in enumerate(costs):
        ax1.text(i, min(c, 30) + 0.6, f"{c:.0f}" if c >= 1 else f"{c:.1f}", ha="center", fontsize=7)
    ax1.set_xticks(range(8)); ax1.set_xticklabels(heads)
    ax1.set_xlabel("donated block-0 head")
    ax1.set_ylabel("perplexity cost (%, capped at 30)")
    ax1.set_ylim(0, 33)
    ax1.set_title("(a) head donation cost is head-specific", fontsize=9)

    depths = [int(d) for d in sorted(F["widen_toggle"], key=int)]
    ax2.plot(depths, [F["widen_toggle"][str(d)] for d in depths], "o-", ms=5, c="#27ae60",
             label="toggle accuracy")
    ax2.axhline(0.5, ls="--", c="k", lw=0.8); ax2.text(40, 0.53, "chance", fontsize=7)
    ax2.set_xscale("log", base=2); ax2.set_xticks(depths); ax2.set_xticklabels(depths)
    ax2.set_xlabel("toggle chain depth")
    ax2.set_ylabel("behavioral accuracy")
    ax2.set_ylim(0.4, 1.08)
    ax2.set_title(f"(b) widened implant: exact readout, ppl {F['widen_implant_pct']:+.2f}%",
                  fontsize=9)
    fig.savefig(FIGS / "fig1_construction.pdf")
    plt.close(fig)

    # --- F2: battery trajectories (none arm: behavioral vs internal) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    colors = {"none_p0_s0": "#c0392b", "none_p10_s0": "#27ae60", "none_p50_s0": "#2980b9"}
    labs = {"none_p0_s0": "0% exercise", "none_p10_s0": "10%", "none_p50_s0": "50%"}
    for c in ("none_p0_s0", "none_p10_s0", "none_p50_s0"):
        cur = cell(c)["curve"]
        st = [x["step"] / 1000 for x in cur]
        ax1.plot(st, [x["beh_toggle8"] for x in cur], "o-", ms=3, c=colors[c], label=labs[c])
    ax1.axhline(0.5, ls="--", c="k", lw=0.8); ax1.text(2.5, 0.53, "chance", fontsize=7)
    ax1.set_xlabel("continued-training step (thousands)")
    ax1.set_ylabel("behavioral toggle-8")
    ax1.set_ylim(-0.05, 1.08)
    ax1.legend(frameon=False, fontsize=7, title="none arm", title_fontsize=7)
    ax1.set_title("(a) behavioral expression: unexercised erodes", fontsize=9)

    cur = cell("none_p0_s0")["curve"]
    st = [x["step"] / 1000 for x in cur]
    ax2.plot(st, [x["beh_toggle8"] for x in cur], "o-", ms=3, c="#c0392b", label="behavioral")
    ax2.plot(st, [x["internal_toggle"] for x in cur], "s--", ms=3, c="#8e44ad", label="internal circuit")
    ax2.axhline(0.5, ls="--", c="k", lw=0.8)
    ax2.annotate("re-pin readout\n$\\to$ 1.00", xy=(3.0, 1.0), xytext=(1.4, 0.72), fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#27ae60"), color="#27ae60")
    ax2.set_xlabel("continued-training step (thousands)")
    ax2.set_ylabel("toggle-8")
    ax2.set_ylim(-0.05, 1.08)
    ax2.legend(frameon=False, fontsize=7, loc="center left")
    ax2.set_title("(b) none/0%: behavior dies, circuit re-forms", fontsize=9)
    fig.savefig(FIGS / "fig2_trajectories.pdf")
    plt.close(fig)

    # --- F3: end-state taxonomy (behavioral / internal / resurrection) ---
    fig, ax = plt.subplots(figsize=(7.0, 2.7), constrained_layout=True)
    order = ["none_p0_s0", "none_p10_s0", "none_p50_s0",
             "freeze_p0_s0", "freeze_p10_s0", "freeze_p50_s0"]
    xl = ["none\n0%", "none\n10%", "none\n50%", "freeze\n0%", "freeze\n10%", "freeze\n50%"]
    beh = [cell(c)["curve"][-1]["beh_toggle8"] for c in order]
    inte = [cell(c)["curve"][-1]["internal_toggle"] for c in order]
    res = [cell(c)["resurrection_beh8"] for c in order]
    x = range(6)
    ax.bar([i - 0.27 for i in x], beh, 0.27, color="#c0392b", label="behavioral (end)")
    ax.bar([i for i in x], inte, 0.27, color="#8e44ad", label="internal circuit (end)")
    ax.bar([i + 0.27 for i in x], res, 0.27, color="#27ae60", label="behavioral after readout re-pin")
    ax.axhline(0.5, ls="--", c="k", lw=0.8); ax.text(5.35, 0.53, "chance", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(xl, fontsize=8)
    ax.set_ylabel("toggle-8 accuracy")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    ax.set_title("End-of-training state: concealed (recoverable) vs eroded vs protected", fontsize=9)
    fig.savefig(FIGS / "fig3_taxonomy.pdf")
    plt.close(fig)
    print("wrote 3 figures to paper/figs/")


if __name__ == "__main__":
    main()
