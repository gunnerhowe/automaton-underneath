"""Figures for the write-economics paper (matplotlib -> paper/figs/*.pdf).
F1: hygiene signature (op vs content gate means, all measured cells) + MQAR at m0.
F2: the emergence event (toggle by depth at m50; A_m50 toggle-8 + probe trajectory).
F3: P4 damping (A_m50 toggle base vs damped by depth; m0 delta annotated).
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
C_OP, C_OTH, C_A, C_B = "#c0392b", "#2980b9", "#7f8c8d", "#27ae60"


def logf(run):
    return json.loads((ROOT / "runs" / run / "log.json").read_text())


def main():
    hyg = json.loads((ROOT / "hygiene.json").read_text())
    p4 = json.loads((ROOT / "p4_results.json").read_text())

    # --- F1: hygiene + recall ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    cells = [("B_m2_s0", "B @ 2\\%"), ("B_m10_s0", "B @ 10\\%"), ("B_m50_s0", "B @ 50\\%"),
             ("Bp_m50_s0", "B$'$ @ 50\\%")]
    x = range(len(cells))
    ax1.bar([i - 0.19 for i in x], [hyg[r]["g_op"] for r, _ in cells], 0.38,
            color=C_OP, label="operation tokens")
    ax1.bar([i + 0.19 for i in x], [hyg[r]["g_other"] for r, _ in cells], 0.38,
            color=C_OTH, label="content/other tokens")
    for i, (r, _) in enumerate(cells):
        ax1.text(i, hyg[r]["g_other"] + 0.03, f"+{hyg[r]['gap']:.2f}", ha="center", fontsize=8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([lab.replace("\\%", "%") for _, lab in cells])
    ax1.set_ylabel("mean learned gate $g_t$")
    ax1.set_ylim(0, 0.78)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_title("(a) The price discovers hygiene (never sees labels)", fontsize=9)

    runs = [("A_m0_s0", "A seed 0"), ("B_m0_s0", "B seed 0"),
            ("A_m0_s1", "A seed 1"), ("B_m0_s1", "B seed 1")]
    vals = [json.loads((ROOT / "runs" / r / "log.json").read_text())[-1]["mqar"] for r, _ in runs]
    cols = [C_A, C_B, C_A, C_B]
    ax2.bar(range(4), vals, 0.6, color=cols)
    ax2.axhline(0.153, ls="--", c="k", lw=0.8)
    ax2.text(-0.42, 0.158, "chance", fontsize=7, ha="left")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([lab for _, lab in runs], fontsize=8)
    ax2.set_ylabel("MQAR accuracy (m0)")
    ax2.set_ylim(0, 0.24)
    ax2.set_title("(b) Hygiene buys recall (small, both seeds)", fontsize=9)
    fig.savefig(FIGS / "fig1_hygiene.pdf")
    plt.close(fig)

    # --- F2: the emergence event ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    depths = [2, 4, 8, 16, 32]
    for run, lab, c, m in [("A_m50_s0", "A (always-write)", "#c0392b", "o"),
                           ("B_m50_s0", "B (priced gate)", "#27ae60", "s"),
                           ("C_m50_s0", "C (oracle gate)", "#2980b9", "^"),
                           ("Bp_m50_s0", "B$'$ (unpriced gate)", "#8e44ad", "v")]:
        tb = logf(run)[-1]["track_by"]
        ax1.plot(depths, [tb[f"toggle_{d}"] for d in depths], marker=m, ms=4, c=c, label=lab)
    ax1.axhline(0.5, ls="--", c="k", lw=0.8)
    ax1.axvspan(1, 8, color="0.92", zorder=0)
    ax1.text(2.2, 0.31, "training depths", fontsize=7, color="0.35")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(depths)
    ax1.set_xticklabels(depths)
    ax1.set_xlabel("toggle chain depth")
    ax1.set_ylabel("accuracy (m50)")
    ax1.set_ylim(0.25, 1.05)
    ax1.legend(frameon=False, fontsize=7, loc="center right")
    ax1.set_title("(a) toggle accuracy by depth, m50", fontsize=9)

    la = logf("A_m50_s0")
    steps = [r["step"] / 1000 for r in la]
    l1, = ax2.plot(steps, [r["track_by"]["toggle_8"] for r in la], "o-", ms=3, c="#c0392b",
                   label="toggle-8 accuracy")
    l2, = ax2.plot(steps, [r["track_by"]["dial_8"] for r in la], "s-", ms=3, c="#7f8c8d",
                   label="dial-8 accuracy")
    ax2.axhline(0.5, ls="--", c="k", lw=0.7)
    ax2.axhline(1 / 3, ls=":", c="k", lw=0.7)
    ax2.set_xlabel("training step (thousands)")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1.08)
    axr = ax2.twinx()
    axr.spines.right.set_visible(True)
    pr = [json.loads(l) for l in
          (ROOT / "runs" / "A_m50_s0" / "probes.jsonl").read_text().splitlines()]
    l3, = axr.plot([p["step"] / 1000 for p in pr], [p["probe_toggle"] for p in pr], "-", lw=0.9,
                   c="#e67e22", alpha=0.85, label="toggle probe (right axis)")
    axr.axhline(0.45, ls="-.", c="#e67e22", lw=0.7, alpha=0.6)
    axr.text(24.5, 0.455, "frozen anchor threshold", fontsize=6, color="#e67e22", ha="right")
    axr.set_ylim(0, 0.5)
    axr.set_ylabel("probe (cosine gap)", color="#e67e22", fontsize=8)
    axr.tick_params(axis="y", labelcolor="#e67e22", labelsize=7)
    ax2.legend(handles=[l1, l2, l3], frameon=False, fontsize=7, loc="center right",
               bbox_to_anchor=(1.0, 0.62))
    ax2.set_title("(b) A@m50 training trajectory", fontsize=9)
    fig.savefig(FIGS / "fig2_emergence.pdf")
    plt.close(fig)

    # --- F3: damping destroys ---
    fig, ax = plt.subplots(figsize=(3.6, 2.5), constrained_layout=True)
    d3 = [2, 4, 8]
    base = [p4["A_m50_s0"]["base"][f"toggle_{d}"] for d in d3]
    damp = [p4["A_m50_s0"]["damped"][f"toggle_{d}"] for d in d3]
    x = range(len(d3))
    ax.bar([i - 0.19 for i in x], base, 0.38, color="#c0392b", label="base")
    ax.bar([i + 0.19 for i in x], damp, 0.38, color="#95a5a6", label="op-writes damped")
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.text(1.99, 0.52, "chance", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"depth {d}" for d in d3])
    ax.set_ylabel("A@m50 toggle accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8)
    d0 = p4["A_m0_s0"]["deep_damped"] - p4["A_m0_s0"]["deep_base"]
    ax.set_title(f"Damping deletes the skill (A@m0 delta: {d0:+.3f})", fontsize=9)
    fig.savefig(FIGS / "fig3_damping.pdf")
    plt.close(fig)
    print("wrote 3 figures to paper/figs/")


if __name__ == "__main__":
    main()
