"""Analysis for the statetrack note: aggregates runs/ artifacts into (a) per-cell medians + tags under
the PRE-REGISTERED decision rules, (b) verdicts on every pre-registered prediction/kill, (c)
numbers.json + paper/numbers.tex (every number cited in the note is a macro), (d) figures. Read-only
over artifacts; safe to run mid-grid (incomplete cells are marked PENDING and excluded from verdicts).
Run: python analyze.py [--figs]
"""
import json
import argparse
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PAPER = ROOT / "paper"
POS = [32, 64, 128, 256, 512]
NAIVE_REFL = {"parity": 1, "mod3": 2, "s4": 3, "a5": 4, "s5": 4}   # max generator rank(I-P), defining rep
N_SEEDS = {"m3": 5, "m0": 3, "m2": 3}


def load():
    recs = []
    for f in sorted(RUNS.glob("*.json")):
        try:
            r = json.loads(f.read_text())
            if "pos" in r:
                recs.append(r)
        except Exception:
            pass
    return recs


def key(r):
    return (r["task"], r["model"], r.get("n_h", 0), bool(r.get("use_b")), r.get("init", "rand"))


def cells(recs):
    out = {}
    for r in recs:
        out.setdefault(key(r), []).append(r)
    return out


def med(rs, field, p):
    vals = [r[field][str(p)] for r in rs if field in r]
    return (st.median(vals), min(vals), max(vals)) if vals else None


def tag(rs):
    """Pre-registered decision rules on medians: GEN pos512>0.9; shortcut pos32>0.9 & pos512<0.6;
    no-fit pos32<0.9; else mixed."""
    m32, m512 = med(rs, "pos", 32)[0], med(rs, "pos", 512)[0]
    if m512 > 0.9:
        return "GEN"
    if m32 > 0.9 and m512 < 0.6:
        return "shortcut"
    if m32 < 0.9:
        return "no-fit"
    return "mixed"


def complete(rs, model):
    return len(rs) >= N_SEEDS.get(model, 5)


def fmt(v):
    return f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]" if v else "--"


def min_nh(cs, task, need=0.9):
    """Smallest n_h whose -b random-init cell is GEN (median pos512 > need); None if none <= 4."""
    for nh in (1, 2, 3, 4):
        rs = cs.get((task, "m3", nh, False, "rand"), [])
        if complete(rs, "m3") and med(rs, "pos", 512)[0] > need:
            return nh
    return None


def verdicts(cs):
    v = {}

    def cell(task, nh, b, init="rand"):
        return cs.get((task, "m3", nh, b, init), [])

    # P-R0 / K0: the core flips reproduce
    pr = {}
    for t, nh in (("parity", 1), ("s5", 4)):
        pb, mb = cell(t, nh, True), cell(t, nh, False)
        if complete(pb, "m3") and complete(mb, "m3"):
            ok = med(mb, "pos", 512)[0] > 0.9 and med(pb, "pos", 512)[0] <= 0.9
            pr[t] = "REPRODUCES" if ok else "FAILS -> K0 FIRES"
        else:
            pr[t] = "PENDING"
    v["P-R0/K0"] = pr
    # P-E1 / K1: s5 nh=3 -b must not GEN
    rs = cell("s5", 3, False)
    v["P-E1/K1"] = ("PENDING" if not complete(rs, "m3") else
                    ("K1 FIRES: nh=3 GENs -> necessity FALSE" if med(rs, "pos", 512)[0] > 0.9
                     else f"HOLDS: s5 nh=3 -b = {tag(rs)}"))
    # P-E2: representation law branch
    e2 = {}
    for t in ("s4", "a5"):
        have = all(complete(cell(t, nh, False), "m3") for nh in (1, 2, 3, 4))
        m = min_nh(cs, t)
        e2[t] = ("PENDING" if not have and m is None else
                 f"min_nh={m} -> " + ("H-A (compact rep)" if m == 2 else
                                      f"H-B (defining-rep law={NAIVE_REFL[t]})" if m == NAIVE_REFL[t]
                                      else "OTHER (report verbatim)" if m else "not reached <=4"))
    v["P-E2"] = e2
    # P-E3: exact-init
    e3 = {}
    for t, nh in (("parity", 1), ("s5", 4)):
        pb, mb = cell(t, nh, True, "exact"), cell(t, nh, False, "exact")
        if complete(pb, "m3") and complete(mb, "m3"):
            post = med(pb, "pos", 512)[0]
            ctrl = med(mb, "pos", 512)[0]
            usage = st.median([r["usage_per_epoch"][-1][0] for r in pb])
            bz = med(pb, "pos_bzero", 512)
            e3[t] = (f"+b post={post:.2f} ({'PULLED OFF exact' if post <= 0.9 else 'retained'}), "
                     f"-b ctrl={ctrl:.2f}, final||b||={usage:.3f}, bzero512={fmt(bz)}")
        else:
            e3[t] = "PENDING"
    v["P-E3"] = e3
    # P-P3 / K2: is the additive path load-bearing in random-init +b shortcut cells?
    p3 = {}
    for (t, m, nh, b, init), rs in sorted(cs.items()):
        if m != "m3" or not b or init != "rand" or not complete(rs, "m3") or tag(rs) != "shortcut":
            continue
        bz32, bz512 = med(rs, "pos_bzero", 32), med(rs, "pos_bzero", 512)
        if bz32 is None:
            continue
        mode = ("RESTORES-GEN (parasitic b; automaton underneath)" if bz512[0] > 0.9
                else "collapses (load-bearing b)" if bz32[0] < 0.6
                else "K2-ZONE: survives in-domain w/o b, no GEN" if bz32[0] > 0.9
                else "partial")
        p3[f"{t} nh={nh}"] = f"bzero32={fmt(bz32)} bzero512={fmt(bz512)} -> {mode}"
    v["P-P3/K2"] = p3
    return v


def write_numbers(cs, v):
    PAPER.mkdir(exist_ok=True)
    num = {}

    def put(name, val):
        num[name] = val

    for (t, m, nh, b, init), rs in cs.items():
        if m != "m3" or not complete(rs, "m3"):
            continue
        base = f"{t}nh{nh}{'bp' if b else 'bm'}{'X' if init == 'exact' else ''}"
        for p in (32, 512):
            md = med(rs, "pos", p)
            put(f"{base}p{p}", round(md[0], 3))
            put(f"{base}p{p}lo", round(md[1], 3))
            put(f"{base}p{p}hi", round(md[2], 3))
        put(f"{base}tag", tag(rs))
        if b:
            bz = med(rs, "pos_bzero", 512)
            if bz:
                put(f"{base}bz512", round(bz[0], 3))
                put(f"{base}bz32", round(med(rs, "pos_bzero", 32)[0], 3))
                put(f"{base}bzgen", sum(1 for r in rs if r.get("pos_bzero", {}).get("512", 0) > 0.9))
                put(f"{base}bzn", len(rs))
        if init == "exact" and b:
            put(f"{base}usageend", round(st.median([r["usage_per_epoch"][-1][0] for r in rs]), 3))
    fit_n = surv_n = 0
    for (t, m, nh, b, init), rs in cs.items():
        if m == "m3" and b:
            for r in rs:
                if "pos_bzero" in r and r["pos"]["32"] > 0.9:
                    fit_n += 1
                    surv_n += r["pos_bzero"]["32"] > 0.9
    put("bfitseeds", fit_n)
    put("bfitsurvive", surv_n)
    for t in ("parity", "mod3", "s4", "a5", "s5"):
        put(f"minnh{t}", min_nh(cs, t) or 0)
        rs = cs.get((t, "m0", 0, False, "rand"), [])
        if complete(rs, "m0"):
            put(f"gru{t}p512", round(med(rs, "pos", 512)[0], 3))
    (ROOT / "numbers.json").write_text(json.dumps(num, indent=1, sort_keys=True))

    def texval(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    DIG = {"0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR", "5": "FIVE",
           "6": "SIX", "7": "SEVEN", "8": "EIGHT", "9": "NINE"}

    def texkey(k):
        return "".join(DIG.get(ch, ch) for ch in k)          # LaTeX macro names cannot contain digits
    tex = "".join(f"\\newcommand{{\\st{texkey(k)}}}{{{texval(val)}}}\n" for k, val in sorted(num.items()))
    (PAPER / "numbers.tex").write_text(tex.replace("_", ""))
    return num


def figures(cs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAPER.joinpath("figs").mkdir(parents=True, exist_ok=True)
    C = {True: "#d62728", False: "#1f77b4"}
    # fig 1: the flip (parity nh1, s5 nh4), median curves with seed range
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, (t, nh) in zip(axes, (("parity", 1), ("s5", 4))):
        for b in (True, False):
            rs = cs.get((t, "m3", nh, b, "rand"), [])
            if not rs:
                continue
            xs = list(range(8, 513, 8))
            ys = [st.median([r["curve64"][i] for r in rs]) for i in range(64)]
            lo = [min(r["curve64"][i] for r in rs) for i in range(64)]
            hi = [max(r["curve64"][i] for r in rs) for i in range(64)]
            ax.plot(xs, ys, color=C[b], lw=1.8, label=f"{'+b' if b else '-b'} (n={len(rs)})")
            ax.fill_between(xs, lo, hi, color=C[b], alpha=0.15, lw=0)
        ax.axvline(32, color="gray", lw=0.8, ls=":")
        ax.set(title=f"{t}  (n_h={nh})", xlabel="position", ylim=(0, 1.05))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("per-position accuracy")
    fig.tight_layout()
    fig.savefig(PAPER / "figs" / "fig1_flip.pdf")
    plt.close(fig)
    # fig 2: representation law -- -b tag grid (task x n_h) + naive reflection length markers
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    tasks = ["parity", "mod3", "s5", "s4", "a5"]
    colmap = {"GEN": "#2ca02c", "shortcut": "#d62728", "no-fit": "#7f7f7f", "mixed": "#ff7f0e"}
    for yi, t in enumerate(tasks):
        for nh in (1, 2, 3, 4):
            rs = cs.get((t, "m3", nh, False, "rand"), [])
            if not rs:
                continue
            m = med(rs, "pos", 512)[0]
            ax.scatter(nh, yi, s=560, marker="s", c=colmap[tag(rs)], edgecolors="k", lw=0.5, zorder=2)
            ax.text(nh, yi, f"{m:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if tag(rs) != "mixed" else "black", zorder=3)
        ax.scatter(NAIVE_REFL[t] - 0.33, yi, marker=">", s=70, c="k", zorder=4)
    ax.set(yticks=range(len(tasks)), yticklabels=tasks, xticks=(1, 2, 3, 4),
           xlabel="n_h (Householder factors/token)", title="-b cells: median pos-512 acc; ▶ = defining-rep reflection length")
    ax.set_xlim(0.4, 4.6)
    fig.tight_layout()
    fig.savefig(PAPER / "figs" / "fig2_law.pdf")
    plt.close(fig)
    # fig 3: E3 -- additive-path usage growth from exact init + outcome bars
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for t, nh, col in (("parity", 1, "#d62728"), ("s5", 4, "#9467bd")):
        rs = cs.get((t, "m3", nh, True, "exact"), [])
        if not rs:
            continue
        n_ep = min(len(r["usage_per_epoch"]) for r in rs)
        ys = [st.median([r["usage_per_epoch"][e][0] for r in rs]) for e in range(n_ep)]
        axes[0].plot(range(n_ep), ys, color=col, lw=1.8, label=f"{t} n_h={nh}")
    axes[0].set(xlabel="epoch", ylabel="probe ||W_b e|| (median)", title="additive-path growth from EXACT init")
    axes[0].legend(fontsize=8)
    labels, vals, cols = [], [], []
    for t, nh in (("parity", 1), ("s5", 4)):
        for b, init, nm in ((True, "exact", "+b exact\npost-train"), (False, "exact", "-b exact\npost-train")):
            rs = cs.get((t, "m3", nh, b, init), [])
            if rs:
                labels.append(f"{t}\n{nm}")
                vals.append(med(rs, "pos", 512)[0])
                cols.append("#d62728" if b else "#1f77b4")
        rs = cs.get((t, "m3", nh, True, "exact"), [])
        if rs and med(rs, "pos_bzero", 512):
            labels.append(f"{t}\n+b, W_b:=0\nat inference")
            vals.append(med(rs, "pos_bzero", 512)[0])
            cols.append("#2ca02c")
    axes[1].bar(range(len(vals)), vals, color=cols)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=6.5)
    axes[1].set(ylabel="median pos-512 acc", ylim=(0, 1.05), title="exact-init outcomes")
    fig.tight_layout()
    fig.savefig(PAPER / "figs" / "fig3_exact.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", action="store_true")
    a = ap.parse_args()
    recs = load()
    cs = cells(recs)
    print(f"artifacts: {len(recs)}  cells: {len(cs)}\n")
    print(f"{'cell':<34}{'n':>3}  {'pos32':>18}  {'pos512':>18}  tag")
    for k, rs in sorted(cs.items()):
        t, m, nh, b, init = k
        nm = f"{t} {m} nh={nh} {'+b' if b else '-b'} {init}"
        pend = "" if complete(rs, m) else "  (PENDING)"
        print(f"{nm:<34}{len(rs):>3}  {fmt(med(rs,'pos',32)):>18}  {fmt(med(rs,'pos',512)):>18}  {tag(rs)}{pend}")
    print("\n=== PRE-REGISTERED VERDICTS ===")
    v = verdicts(cs)
    for k, val in v.items():
        if isinstance(val, dict):
            print(f"{k}:")
            for kk, vv in val.items():
                print(f"   {kk}: {vv}")
        else:
            print(f"{k}: {val}")
    num = write_numbers(cs, v)
    print(f"\nnumbers.json: {len(num)} macros")
    if a.figs:
        figures(cs)
        print("figures written to paper/figs/")


if __name__ == "__main__":
    main()
