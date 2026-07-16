"""Mechanical verdicts for Step 2b against PREREG_STEP2.md. Reads results_intervention/, prints the
cell table and evaluates P1/P2/P3, K1/K2 exactly as pre-registered. Run: python analyze_step2.py"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
CELLS = {}
for f in sorted((ROOT / "results_intervention").glob("*.json")):
    r = json.loads(f.read_text())
    CELLS[(r["arm"], r["gamma"], r["depth"])] = r

DEPTHS = (2, 4, 8, 16)
GAMMAS = (0.5, 0.0)


def cell(arm, g, d):
    return CELLS.get((arm, g, d))


def main():
    print(f"{'cell':<16}{'acc':>7}{'TR':>7}{'gold':>6}{'init':>6}")
    for d in DEPTHS:
        for arm, g in [("BASE", 1.0)] + [(a, g) for a in ("S", "C", "R") for g in GAMMAS]:
            r = cell(arm, g, d)
            nm = f"{arm}_g{g}_d{d}"
            if r:
                print(f"{nm:<16}{r['acc']:>7.3f}{r['tracking_rate']:>7.3f}{r['gold']:>6}{r['initial']:>6}")
            else:
                print(f"{nm:<16}{'PENDING':>7}")
        print()
    print("=== PRE-REGISTERED VERDICTS (PREREG_STEP2.md) ===")
    # P1 / K1: some S cell with TR > baseline+0.05 AND > matched R control +0.05
    winners = []
    pending = False
    for d in DEPTHS:
        b = cell("BASE", 1.0, d)
        for g in GAMMAS:
            s, r = cell("S", g, d), cell("R", g, d)
            if not (b and s and r):
                pending = True
                continue
            ds_b = s["tracking_rate"] - b["tracking_rate"]
            ds_r = s["tracking_rate"] - r["tracking_rate"]
            tag = "CLEARS" if (ds_b > 0.05 and ds_r > 0.05) else "no"
            print(f"P1 S_g{g}_d{d}: dTR vs base {ds_b:+.3f}, vs R {ds_r:+.3f}  -> {tag}")
            if tag == "CLEARS":
                winners.append((g, d, ds_b, ds_r))
    if pending:
        print("P1/K1: PENDING cells remain")
    elif winners:
        print(f"P1 POSITIVE: {winners} (n=150/cell; margins near threshold require replication per honest-reporting note)")
    else:
        print("K1 FIRES: no S cell clears the double margin -> masked-capacity hypothesis dies on this checkpoint; Step 2 closes as a clean negative.")
    # P2: C at gamma=0 reduces raw accuracy vs baseline at every depth
    ok, tot = 0, 0
    for d in DEPTHS:
        b, c = cell("BASE", 1.0, d), cell("C", 0.0, d)
        if b and c:
            tot += 1
            ok += c["acc"] < b["acc"]
    print(f"P2 (content-damping hurts acc): {ok}/{tot} depths  -> {'holds' if ok == tot else 'FAILS at some depths (instrument note: content writes not uniformly load-bearing at high depth)'}")
    # P3: R approx baseline
    devs = []
    for d in DEPTHS:
        b = cell("BASE", 1.0, d)
        for g in GAMMAS:
            r = cell("R", g, d)
            if b and r:
                devs.append(abs(r["tracking_rate"] - b["tracking_rate"]))
    if devs:
        med = sorted(devs)[len(devs) // 2]
        print(f"P3 (R ~ baseline): median |dTR| = {med:.3f} -> {'holds (<=0.05)' if med <= 0.05 else 'VIOLATED: control fluctuations exceed margin -- treatment effects of this size are uninterpretable (supports null)'}")
    # K2: S gamma=0 collapses acc <= chance (0.2) at all depths
    k2 = all(cell("S", 0.0, d) and cell("S", 0.0, d)["acc"] <= 0.2 for d in DEPTHS if cell("S", 0.0, d))
    print(f"K2 (v carries op info; S g0 <= chance everywhere): {'FIRES' if k2 else 'does not fire'}")


if __name__ == "__main__":
    main()
