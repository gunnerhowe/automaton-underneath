"""Numbers pipeline for paper 4: reads facts.json + runs_wide/*.json, emits numbers.json +
paper/numbers.tex (\\li* macros). Every number the manuscript cites regenerates here.
Run: python gen_numbers.py"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
PAPER = ROOT / "paper"
PAPER.mkdir(exist_ok=True)
CELLS = ["none_p0_s0", "none_p10_s0", "none_p50_s0",
         "freeze_p0_s0", "freeze_p10_s0", "freeze_p50_s0"]
TAG = {"none_p0_s0": "NoneZero", "none_p10_s0": "NoneTen", "none_p50_s0": "NoneFifty",
       "freeze_p0_s0": "FreezeZero", "freeze_p10_s0": "FreezeTen", "freeze_p50_s0": "FreezeFifty"}


def cell(c):
    return json.loads((ROOT / "runs_wide" / f"{c}.json").read_text())


def main():
    F = json.loads((ROOT / "facts.json").read_text())
    N = {}
    # construction / feasibility facts
    N["BasePpl"] = round(F["base_ppl"], 2)
    N["ResidMin"] = F["resid_std_min"]
    N["ResidMedian"] = F["resid_std_median"]
    N["ResidMax"] = F["resid_std_max"]
    N["HeadOne"] = F["head_cost_pct"]["1"]
    N["HeadZero"] = F["head_cost_pct"]["0"]
    N["HeadSix"] = F["head_cost_pct"]["6"]
    N["WidenInertPct"] = F["widen_inert_pct"]
    N["WidenImplantPct"] = F["widen_implant_pct"]
    N["WidenDepth"] = max(int(d) for d in F["widen_toggle"])
    N["WallLogTen"] = len(str(int(F["asis_readout_wall_ppl"]))) - 1     # order of magnitude
    N["Steps"] = cell("none_p0_s0")["curve"][-1]["step"]

    # per-cell battery numbers
    for c in CELLS:
        r = cell(c); t = TAG[c]
        cur = r["curve"]
        N[f"Beh{t}End"] = cur[-1]["beh_toggle8"]
        N[f"Int{t}End"] = cur[-1]["internal_toggle"]
        N[f"Beh{t}Min"] = min(x["beh_toggle8"] for x in cur)
        N[f"Int{t}Min"] = min(x["internal_toggle"] for x in cur)
        N[f"Resurrect{t}"] = r["resurrection_beh8"]
        N[f"Ppl{t}End"] = round(cur[-1]["ppl"], 2)
        N[f"PplPct{t}"] = round((cur[-1]["ppl"] - F["base_ppl"]) / F["base_ppl"] * 100, 1)

    N["ChanceToggle"] = 0.5
    # continued-training drift floor: the no-exercise none/p0 ppl rise is generic (not exercise)
    N["DriftFloorPct"] = N["PplPctNoneZero"]

    # table rows
    rows = []
    for c in CELLS:
        r = cell(c); cur = r["curve"]
        arm = "none" if r["arm"] == "none" else "freeze"
        rows.append(f"{arm} & {int(r['p']*100)}\\% & {cur[0]['beh_toggle8']:.2f} & "
                    f"{cur[-1]['beh_toggle8']:.2f} & {cur[-1]['internal_toggle']:.2f} & "
                    f"{r['resurrection_beh8']:.2f} & {cur[-1]['ppl']:.2f} \\\\")
    table = "\n".join(rows)

    (ROOT / "numbers.json").write_text(json.dumps(N, indent=1, sort_keys=True))
    with open(PAPER / "numbers.tex", "w") as f:
        for k, v in sorted(N.items()):
            f.write(f"\\newcommand{{\\li{k}}}{{{v}}}\n")
        f.write("\\newcommand{\\liBatteryRows}{%\n" + table + "\n}\n")
    print(f"wrote numbers.json ({len(N)} values) + paper/numbers.tex")


if __name__ == "__main__":
    main()
