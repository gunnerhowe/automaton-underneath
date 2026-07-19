"""Mechanical verdicts for PREREG_TWOGAP from run logs. --runs restricts which runs are read
(fusion sequencing: m2 only after the fusion KF3/freeze commit). Run:
python analyze_twogap.py [--all | --calibration]"""
import json
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
CAL = ["A_m50_s0", "B_m50_s0", "C_m50_s0", "Bp_m50_s0", "A_m10_s0", "B_m10_s0", "C_m10_s0",
       "A_m0_s0", "B_m0_s0", "C_m0_s0"]
M2 = ["A_m2_s0", "B_m2_s0", "C_m2_s0"]


def final(run):
    log = json.loads((ROOT / "runs" / run / "log.json").read_text())
    return log[-1]


def deep(r):
    tb = r["track_by"]
    return (tb.get("boxes_16", 0) + tb.get("boxes_32", 0)) / 2


def main():
    runs = CAL + (M2 if "--all" in sys.argv else [])
    F = {r: final(r) for r in runs}
    print(f"{'run':<12}{'ppl':>8}{'track':>8}{'deep16/32':>10}{'tog8':>7}{'dial8':>7}{'mqar':>7}"
          f"{'gate_mean':>10}{'gate_off':>9}")
    for r in runs:
        f = F[r]
        tb = f["track_by"]
        print(f"{r:<12}{f['ppl']:>8.2f}{f['track']:>8.3f}{deep(f):>10.3f}"
              f"{tb.get('toggle_8', 0):>7.2f}{tb.get('dial_8', 0):>7.2f}{f['mqar']:>7.3f}"
              f"{f['gate_mean']:>10.3f}{f['gate_frac_off']:>9.3f}")
    print("\n=== PREREG_TWOGAP verdicts (log-based; P2 hygiene signature computed separately) ===")
    arms = {m: {a: F.get(f"{a}_m{m}_s0") for a in ("A", "B", "C", "Bp")} for m in (50, 10, 0, 2)}
    # P6/K3 ppl guardrail
    for m in (50, 10, 0) + ((2,) if "--all" in sys.argv else ()):
        a, b = arms[m].get("A"), arms[m].get("B")
        if a and b:
            cost = (b["ppl"] - a["ppl"]) / a["ppl"] * 100
            print(f"P6 m{m}: B ppl cost {cost:+.1f}% -> {'ok (<=3%)' if cost <= 3 else 'K3 FIRES' if cost > 5 else 'gray (3-5%)'}")
    # P5 / K2: C vs A deep tracking
    for m in (50, 10) + ((2,) if "--all" in sys.argv else ()):
        a, c = arms[m].get("A"), arms[m].get("C")
        if a and c:
            d = deep(c) - deep(a)
            print(f"P5 m{m}: C-A deep = {d:+.3f} ({'C>A' if d > 0.05 else 'C~A (K2 range)' if abs(d) <= 0.05 else 'A>C'})")
    if "--all" in sys.argv:
        a, c = arms[2].get("A"), arms[2].get("C")
        d = deep(c) - deep(a)
        print(f"P1 m2: C-A deep = {d:+.3f} -> {'CONFIRMED (>+0.10)' if d > 0.10 else 'not confirmed'}")
    # P3: MQAR at m0
    a, b = arms[0].get("A"), arms[0].get("B")
    if a and b:
        d = b["mqar"] - a["mqar"]
        pplok = (b["ppl"] - a["ppl"]) / a["ppl"] <= 0.03
        print(f"P3 m0: B-A mqar = {d:+.3f} (matched-ppl {pplok}) -> "
              f"{'CONFIRMED' if d > 0.03 and pplok else 'not confirmed'}")
    # P2 partial (gate means; hygiene signature needs checkpoints)
    b, bp = arms[50].get("B"), arms[50].get("Bp")
    if b:
        print(f"P2(i) m50: B final gate_mean {b['gate_mean']:.3f} "
              f"({'<0.5 CONFIRMED' if b['gate_mean'] < 0.5 else 'NOT <0.5' + (' -> K1(i) territory' if b['gate_mean'] > 0.9 else '')})"
              f"; gate_frac_off {b['gate_frac_off']:.3f}")
    if bp:
        print(f"      m50: B' (unpriced) gate_mean {bp['gate_mean']:.3f} gate_frac_off {bp['gate_frac_off']:.3f}")


if __name__ == "__main__":
    main()
