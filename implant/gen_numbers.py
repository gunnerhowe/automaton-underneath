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

    (ROOT / "numbers.json").write_text(json.dumps(N, indent=1, sort_keys=True))
    with open(PAPER / "numbers.tex", "w") as f:
        for k, v in sorted(N.items()):
            f.write(f"\\newcommand{{\\im{k}}}{{{v}}}\n")
        f.write("\\newcommand{\\imGridTableRows}{%\n" + table + "\n}\n")
    print(f"wrote numbers.json ({len(N)} values) + paper/numbers.tex")


if __name__ == "__main__":
    main()
