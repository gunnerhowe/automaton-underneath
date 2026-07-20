"""Mechanical Tier-1 verdicts for PREREG_IMPLANT.md from runs_toy/*.json.
Run: python analyze_implant.py"""
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs_toy"
PS, ARMS, SEEDS = (0.0, 0.1, 0.5, 1.0), ("P0", "P1", "P2", "P3"), (0, 1, 2)


def cell(p, arm, s):
    f = RUNS / f"p{int(p*100)}_{arm}_s{s}.json"
    return json.loads(f.read_text()) if f.exists() else None


def endv(c, key):
    vals = [r[key] for r in c["curve"] if key in r]
    return vals[-1] if vals else None


def med(p, arm, key):
    vs = [endv(cell(p, arm, s), key) for s in SEEDS if cell(p, arm, s)]
    return st.median(vs) if vs else None


def main():
    n = sum(1 for _ in RUNS.glob("p*.json"))
    print(f"=== Tier-1 implant persistence ({n}/48 cells) ===\n")
    print(f"{'p':>4} | " + " | ".join(f"{a}: integ  par" for a in ARMS))
    for p in PS:
        row = []
        for a in ARMS:
            i, pa = med(p, a, "integrity"), med(p, a, "par_in")
            row.append(f"{a}: {i if i is not None else '--':>5}  {pa if pa is not None else '--':>5}")
        print(f"{p:>4} | " + " | ".join(row))

    print("\n=== frozen verdicts ===")
    p0_P0, p0_P1 = med(0.0, "P0", "integrity"), med(0.0, "P1", "integrity")
    if p0_P0 is not None and p0_P1 is not None:
        d = p0_P1 - p0_P0
        print(f"IP1 (p=0: P1-P0 integrity >= +0.30): {p0_P1} - {p0_P0} = {d:+.3f} -> "
              f"{'CONFIRMED' if d >= 0.30 else 'not confirmed'}")
    seq = [med(p, "P0", "integrity") for p in PS]
    if all(v is not None for v in seq):
        mono = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
        print(f"IP2 (P0 integrity nondecreasing in p): {seq} -> "
              f"{'CONFIRMED' if mono else 'not confirmed'}")
    p2ok = [med(p, "P2", "integrity") for p in PS]
    if all(v is not None for v in p2ok):
        ok = all(v >= 0.90 for v in p2ok)
        cost = None
        a, b = med(0.0, "P0", "par_in"), med(0.0, "P2", "par_in")
        if a is not None and b is not None:
            cost = a - b
        print(f"IP3 (P2 integrity >= 0.90 everywhere): {p2ok} -> "
              f"{'CONFIRMED' if ok else 'not confirmed'}; plasticity cost at p=0 "
              f"(P0-P2 parity): {cost:+.3f}" if cost is not None else "")
    c = cell(1.0, "P0", 0)
    if c:
        indom = [med(1.0, "P0", "s5_in"), med(1.0, "P0", "integrity")]
        bn = med(1.0, "P0", "b_norm")
        if all(v is not None for v in indom):
            fired = indom[0] >= 0.95 and indom[1] <= 0.70
            print(f"IP4 (p=1 P0: in-domain >= 0.95 AND integrity <= 0.70): "
                  f"{indom[0]} / {indom[1]} (b_norm {bn}) -> "
                  f"{'CONFIRMED (attractor echo)' if fired else 'not confirmed'}")
    ki2 = med(0.0, "P0", "par_in")
    if ki2 is not None:
        print(f"KI2 (distractor learnable, P0@p=0 parity >= 0.90): {ki2} -> "
              f"{'ok' if ki2 >= 0.90 else 'KI2 FIRES'}")
    allint = [med(p, a, "integrity") for p in PS for a in ARMS]
    if all(v is not None for v in allint):
        print(f"KI3 (nothing degrades anywhere): "
              f"{'FIRES — implants robust everywhere' if all(v >= 0.90 for v in allint) else 'does not fire'}")


if __name__ == "__main__":
    main()
