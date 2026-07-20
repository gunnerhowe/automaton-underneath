"""Numbers pipeline for the write-economics paper: reads run artifacts (runs/*/log.json,
hygiene.json, p4_results.json, fusion_constants.json) and emits numbers.json +
paper/numbers.tex (all-alphabetic macro names; every number the manuscript cites).
Run: python gen_numbers.py   (verify_regen.py re-runs this and byte-compares)"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
PAPER = ROOT / "paper"
PAPER.mkdir(exist_ok=True)

GRID = ["A_m50_s0", "B_m50_s0", "C_m50_s0", "Bp_m50_s0",
        "A_m10_s0", "B_m10_s0", "C_m10_s0",
        "A_m2_s0", "B_m2_s0", "C_m2_s0",
        "A_m0_s0", "B_m0_s0", "C_m0_s0"]
REPL = ["A_m0_s1", "B_m0_s1"]
CHANCE = dict(toggle=0.5, dial=1 / 3, boxes=1 / 3, mqar=(1 / 4 + 1 / 8 + 1 / 12) / 3)


def logf(run):
    return json.loads((ROOT / "runs" / run / "log.json").read_text())


def final(run):
    return logf(run)[-1]


def deep_boxes(r):
    tb = r["track_by"]
    return (tb["boxes_16"] + tb["boxes_32"]) / 2


def main():
    F = {r: final(r) for r in GRID + REPL}
    hyg = json.loads((ROOT / "hygiene.json").read_text())
    p4 = json.loads((ROOT / "p4_results.json").read_text())
    fus = json.loads((ROOT / "fusion_constants.json").read_text())["fixed"]
    N = {}

    # --- model size (regenerated from the architecture itself) ---
    import sys
    sys.path.insert(0, str(ROOT))
    from model import TwoGapLM
    N["ParamsM"] = round(TwoGapLM().n_params() / 1e6, 1)

    # --- training budget from logs ---
    steps = F["A_m50_s0"]["step"]
    N["Steps"] = steps
    N["TokensPerStep"] = 12 * 2 * 512                      # bs12 x accum2 x seq512 (Amendment 1)
    N["TrainTokensM"] = round(steps * N["TokensPerStep"] / 1e6)
    N["EvalCadence"] = logf("A_m50_s0")[0]["step"]

    # --- P2: economics -> hygiene ---
    b50, bp50 = F["B_m50_s0"], F["Bp_m50_s0"]
    N["BGateMean"] = b50["gate_mean"]
    N["BGateOffPct"] = round(b50["gate_frac_off"] * 100, 1)
    N["BpGateMean"] = bp50["gate_mean"]
    N["BpGateOffPct"] = round(bp50["gate_frac_off"] * 100, 1)
    for run, tag in [("B_m50_s0", "BmFifty"), ("Bp_m50_s0", "BpmFifty"),
                     ("B_m10_s0", "BmTen"), ("B_m2_s0", "BmTwo")]:
        N[f"HygOp{tag}"] = hyg[run]["g_op"]
        N[f"HygOther{tag}"] = hyg[run]["g_other"]
        N[f"HygGap{tag}"] = hyg[run]["gap"]
    N["HygNSeq"] = hyg["B_m50_s0"]["n_seq"]
    N["HygNOpmFifty"] = hyg["B_m50_s0"]["n_op"]
    N["HygOpRatioBpOverB"] = round(hyg["Bp_m50_s0"]["g_op"] / hyg["B_m50_s0"]["g_op"], 1)

    # --- P3: recall at m0 (both seeds) ---
    for s, tag in [("s0", "SZero"), ("s1", "SOne")]:
        a, b = F[f"A_m0_{s}"], F[f"B_m0_{s}"]
        N[f"MqarA{tag}"] = a["mqar"]
        N[f"MqarB{tag}"] = b["mqar"]
        N[f"MqarDelta{tag}"] = round(b["mqar"] - a["mqar"], 3)
        N[f"PplCostPct{tag}"] = round((b["ppl"] - a["ppl"]) / a["ppl"] * 100, 1)
        N[f"BGateMeanMZero{tag}"] = b["gate_mean"]
    N["MqarDeltaMean"] = round((N["MqarDeltaSZero"] + N["MqarDeltaSOne"]) / 2, 3)
    N["MqarChance"] = round(CHANCE["mqar"], 3)
    for np_ in (4, 8, 12):
        N[f"MqarBmZeroP{'Four' if np_==4 else 'Eight' if np_==8 else 'Twelve'}"] = \
            F["B_m0_s0"]["mqar_by"][f"mqar_{np_}"]
        N[f"MqarAmZeroP{'Four' if np_==4 else 'Eight' if np_==8 else 'Twelve'}"] = \
            F["A_m0_s0"]["mqar_by"][f"mqar_{np_}"]

    # --- P1/P5: oracle-gating vs deep boxes tracking ---
    for m, tag in [(50, "mFifty"), (10, "mTen"), (2, "mTwo")]:
        d = deep_boxes(F[f"C_m{m}_s0"]) - deep_boxes(F[f"A_m{m}_s0"])
        N[f"CminusADeep{tag}"] = round(d, 3)

    # --- P6: ppl guardrail ---
    worst = max((F[f"B_m{m}_s0"]["ppl"] - F[f"A_m{m}_s0"]["ppl"]) / F[f"A_m{m}_s0"]["ppl"]
                for m in (50, 10, 2, 0))
    N["PplWorstCostPct"] = round(worst * 100, 1)

    # --- the emergence event: A_m50 toggle ---
    a50 = F["A_m50_s0"]["track_by"]
    for d, tag in [(2, "Two"), (4, "Four"), (8, "Eight"), (16, "Sixteen"), (32, "ThirtyTwo")]:
        N[f"AmFiftyToggle{tag}"] = a50[f"toggle_{d}"]
    N["OthersToggleEight"] = F["B_m50_s0"]["track_by"]["toggle_8"]
    emerge = [r["step"] for r in logf("A_m50_s0") if r["track_by"]["toggle_8"] >= 0.9]
    N["EmergeStepK"] = round(emerge[0] / 1000, 1) if emerge else None
    dips = [r["step"] for r in logf("A_m50_s0")
            if r["step"] > emerge[0] and r["track_by"]["toggle_8"] < 0.9]
    N["ToggleDipStepK"] = round(dips[0] / 1000) if dips else None
    pr = [json.loads(l) for l in
          (ROOT / "runs" / "A_m50_s0" / "probes.jsonl").read_text().splitlines()]
    N["ProbeMaxAmFifty"] = max(p["probe_toggle"] for p in pr)
    N["DialEightMin"] = min(F[r]["track_by"]["dial_8"] for r in GRID
                            if "_m0_" not in r)
    N["DialEightMax"] = max(F[r]["track_by"]["dial_8"] for r in GRID
                            if "_m0_" not in r)
    N["DialChance"] = round(CHANCE["dial"], 3)

    # --- P4: damping ---
    p50, p0 = p4["A_m50_s0"], p4["A_m0_s0"]
    for d, tag in [(2, "Two"), (4, "Four"), (8, "Eight")]:
        N[f"PfourBaseToggle{tag}"] = p50["base"][f"toggle_{d}"]
        N[f"PfourDampToggle{tag}"] = p50["damped"][f"toggle_{d}"]
    N["PfourDeepDeltamFifty"] = round(p50["deep_damped"] - p50["deep_base"], 3)
    N["PfourDeepDeltamZero"] = round(p0["deep_damped"] - p0["deep_base"], 3)

    # --- replicate-noise floor: C at m0 computes the same function as A at m0 (all-ones oracle
    # mask), so their end-of-training gap calibrates hardware/replicate noise ---
    N["NoiseFloorMqar"] = round(abs(F["C_m0_s0"]["mqar"] - F["A_m0_s0"]["mqar"]), 3)
    N["NoiseFloorTrack"] = round(abs(F["C_m0_s0"]["track"] - F["A_m0_s0"]["track"]), 3)

    # --- exploratory: oracle-arm recall at mixture > 0 ---
    for m, tag in [(50, "mFifty"), (10, "mTen")]:
        N[f"CMqar{tag}"] = F[f"C_m{m}_s0"]["mqar"]
        N[f"AMqar{tag}"] = F[f"A_m{m}_s0"]["mqar"]

    # --- fusion (KF3) ---
    N["FusionNEvent"] = fus["n_event"]
    N["FusionAnchorTp"] = fus["tp"]
    N["FusionAnchorTb"] = fus["tb"]
    N["FusionNearMissToggle"] = F["A_m50_s0"]["track_by"]["toggle_8"]
    N["FusionNearMissDial"] = F["A_m50_s0"]["track_by"]["dial_8"]

    # --- production measurement (prodgate/prodgate_results.json; prereg a876a58) ---
    pg = json.loads((ROOT.parent / "prodgate" / "prodgate_results.json").read_text())
    N["ProdNll"] = pg["V1"]["mean_nll"]
    for sp, tag in [("dn", "Dn"), ("m2", "Mt")]:
        s1 = pg["S1"][sp]
        N[f"Prod{tag}Rep"] = s1["r_rep"]
        N[f"Prod{tag}Ctl"] = s1["r_ctl"]
        N[f"Prod{tag}Diff"] = s1["diff"]
        N[f"Prod{tag}CiLo"] = s1["ci"][0]
        N[f"Prod{tag}CiHi"] = s1["ci"][1]
        N[f"Prod{tag}LZeroRep"] = s1["per_layer_rep"][0]
        N[f"Prod{tag}LZeroCtl"] = s1["per_layer_ctl"][0]
        N[f"ProdSTwo{tag}Gap"] = pg["S2"][sp]["gap"]
        ops = [pg["S3"][sp][f]["op"] for f in ("boxes", "toggle", "dial")]
        oth = [pg["S3"][sp][f]["other"] for f in ("boxes", "toggle", "dial")]
        N[f"ProdSThree{tag}Op"] = round(sum(ops) / 3, 3)
        N[f"ProdSThree{tag}Other"] = round(sum(oth) / 3, 3)
        N[f"ProdSThree{tag}Ratio"] = round(sum(ops) / sum(oth), 2)

    # --- main grid table rows ---
    rows = []
    for r in GRID:
        f = F[r]
        tb = f["track_by"]
        arm, mix = r.split("_")[0], r.split("_")[1]
        arm = {"A": "A", "B": "B", "Bp": "B$'$", "C": "C"}[arm]
        rows.append(f"{arm} & {mix[1:]} & {f['ppl']:.2f} & {tb['toggle_8']:.2f} & "
                    f"{tb['dial_8']:.2f} & {deep_boxes(f):.2f} & {f['mqar']:.3f} & "
                    f"{f['gate_mean']:.3f} & {f['gate_frac_off']*100:.0f}\\% \\\\")
    table = "\n".join(rows)

    (ROOT / "numbers.json").write_text(json.dumps(N, indent=1, sort_keys=True))
    with open(PAPER / "numbers.tex", "w") as f:
        for k, v in sorted(N.items()):
            f.write(f"\\newcommand{{\\tg{k}}}{{{v}}}\n")
        f.write("\\newcommand{\\tgGridTableRows}{%\n" + table + "\n}\n")
    print(f"wrote numbers.json ({len(N)} values) + paper/numbers.tex")


if __name__ == "__main__":
    main()
