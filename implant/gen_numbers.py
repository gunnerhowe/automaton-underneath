"""Numbers pipeline for the implant note: reads runs_toy/*.json, reveal_results.json,
runs_lm/{skill/create_log.json, explore_cold.json, explore_warmx.json} and emits numbers.json +
paper/numbers.tex (\\im* macros). Run: python gen_numbers.py"""
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent
PAPER = ROOT / "paper"
PAPER.mkdir(exist_ok=True)
PS = [(0.0, "Pzero"), (0.1, "Pten"), (0.5, "Pfifty"), (1.0, "Phundred")]
ARMS = [("P0", "None"), ("P1", "Minusb"), ("P2", "Freeze"), ("P3", "Disc")]
SEEDS = (0, 1, 2)


def cell(p, arm, s):
    return json.loads((ROOT / "runs_toy" / f"p{int(p*100)}_{arm}_s{s}.json").read_text())


def endv(c, key):
    vals = [r[key] for r in c["curve"] if key in r]
    return vals[-1]


def med(p, arm, key):
    return round(st.median(endv(cell(p, arm, s), key) for s in SEEDS), 4)


def main():
    N = {}
    for p, ptag in PS:
        for arm, atag in ARMS:
            N[f"Int{ptag}{atag}"] = med(p, arm, "integrity")
            N[f"Par{ptag}{atag}"] = med(p, arm, "par_in")
    N["IPoneDiff"] = round(N["IntPzeroMinusb"] - N["IntPzeroNone"], 3)
    N["IPfourInDom"] = med(1.0, "P0", "s5_in")
    N["IPfourBnorm"] = med(1.0, "P0", "b_norm")
    N["PlastCost"] = round(N["ParPzeroNone"] - N["ParPzeroFreeze"], 3)

    rev = json.loads((ROOT / "reveal_results.json").read_text())
    for p, ptag in [(0.0, "Pzero"), (1.0, "Phundred")]:
        for arm, atag in [("P0", "None"), ("P2", "Freeze")]:
            rows = [r for r in rev if r["p"] == p and r["arm"] == arm]
            N[f"Rev{ptag}{atag}Before"] = round(st.mean(r["integrity"] for r in rows), 3)
            N[f"Rev{ptag}{atag}After"] = round(st.mean(r["integrity_bzero"] for r in rows), 3)
            N[f"Rev{ptag}{atag}Bnorm"] = round(st.mean(r["b_norm"] for r in rows), 3)
    N["RevPzeroNoneInDom"] = round(st.mean(r["indom_bzero"] for r in rev
                                           if r["p"] == 0.0 and r["arm"] == "P0"), 3)

    create = json.loads((ROOT / "runs_lm" / "skill" / "create_log.json").read_text())
    N["LmPreToggle"] = create["log"][0]["toggle_8"]
    N["LmPrePpl"] = create["log"][0]["ppl"]
    N["LmCreateToggle"] = create["final_toggle8"]
    cold = json.loads((ROOT / "runs_lm" / "explore_cold.json").read_text())
    N["ColdEmergeStep"] = next(r["step"] for r in cold if r["step"] > 0 and r["toggle_8"] >= 0.9)
    N["ColdEndPpl"] = cold[-1]["ppl"]
    warm = json.loads((ROOT / "runs_lm" / "explore_warmx.json").read_text())
    N["WarmTotalStepsK"] = 5 + warm[-1]["step"] // 1000
    N["WarmMaxToggle"] = max(r["toggle_8"] for r in warm)
    N["WarmEndPpl"] = warm[-1]["ppl"]
    N["BudgetRatio"] = round((5000 + warm[-1]["step"]) / N["ColdEmergeStep"])

    rows = []
    for p, ptag in PS:
        r = [f"{int(p*100)}\\%"]
        for arm, atag in ARMS:
            r.append(f"{N[f'Int{ptag}{atag}']:.2f} / {N[f'Par{ptag}{atag}']:.2f}")
        rows.append(" & ".join(r) + " \\\\")
    table = "\n".join(rows)

    # --- Tier 3 (LM-scale widened implant): read the lmimplant experimental record ---
    LMW = ROOT.parent / "lmimplant"
    F = json.loads((LMW / "facts.json").read_text())
    N["LmwBasePpl"] = round(F["base_ppl"], 2)
    N["LmwResidMin"] = F["resid_std_min"]
    N["LmwResidMedian"] = F["resid_std_median"]
    N["LmwHeadOne"] = F["head_cost_pct"]["1"]
    N["LmwHeadSix"] = F["head_cost_pct"]["6"]
    N["LmwWidenInertPct"] = F["widen_inert_pct"]
    N["LmwWidenImplantPct"] = F["widen_implant_pct"]
    N["LmwWidenDepth"] = max(int(d) for d in F["widen_toggle"])
    N["LmwWallLogTen"] = len(str(int(F["asis_readout_wall_ppl"]))) - 1
    LCELLS = [("none_p0_s0", "NoneZero"), ("none_p10_s0", "NoneTen"), ("none_p50_s0", "NoneFifty"),
              ("freeze_p0_s0", "FreezeZero"), ("freeze_p10_s0", "FreezeTen"),
              ("freeze_p50_s0", "FreezeFifty")]
    lrows = []
    for fn, t in LCELLS:
        r = json.loads((LMW / "runs_wide" / f"{fn}.json").read_text())
        cur = r["curve"]
        N[f"LmwBeh{t}End"] = cur[-1]["beh_toggle8"]
        N[f"LmwInt{t}End"] = cur[-1]["internal_toggle"]
        N[f"LmwResurrect{t}"] = r["resurrection_beh8"]
        N[f"LmwPplPct{t}"] = round((cur[-1]["ppl"] - F["base_ppl"]) / F["base_ppl"] * 100, 1)
        arm = "none" if r["arm"] == "none" else "freeze"
        lrows.append(f"{arm} & {int(r['p']*100)}\\% & {cur[0]['beh_toggle8']:.2f} & "
                     f"{cur[-1]['beh_toggle8']:.2f} & {cur[-1]['internal_toggle']:.2f} & "
                     f"{r['resurrection_beh8']:.2f} & {cur[-1]['ppl']:.2f} \\\\")
    N["LmwSteps"] = json.loads((LMW / "runs_wide" / "none_p0_s0.json").read_text())["curve"][-1]["step"]
    N["LmwDriftFloorPct"] = N["LmwPplPctNoneZero"]
    lm_table = "\n".join(lrows)

    (ROOT / "numbers.json").write_text(json.dumps(N, indent=1, sort_keys=True))
    with open(PAPER / "numbers.tex", "w") as f:
        for k, v in sorted(N.items()):
            f.write(f"\\newcommand{{\\im{k}}}{{{v}}}\n")
        f.write("\\newcommand{\\imGridTableRows}{%\n" + table + "\n}\n")
        f.write("\\newcommand{\\imLmwBatteryRows}{%\n" + lm_table + "\n}\n")
    print(f"wrote numbers.json ({len(N)} values) + paper/numbers.tex")


if __name__ == "__main__":
    main()
