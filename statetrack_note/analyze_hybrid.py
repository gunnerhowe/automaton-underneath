"""Verdicts for the hybrid bridge experiment (PREREG_HYBRID.md). Aggregates runs_hybrid/ artifacts,
prints per-cell medians + tags under the pre-registered rules, and evaluates P1-P6 / K1-K3.
Run: python analyze_hybrid.py"""
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs_hybrid"


def load():
    out = {}
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text())
        if "pos" in r:
            out.setdefault((r["model"], r.get("n_h", 0), r.get("init", "rand")), []).append(r)
    return out


def med(rs, field, p):
    vals = [r[field][str(p)] for r in rs if field in r]
    return (st.median(vals), min(vals), max(vals)) if vals else None


def tag(rs):
    m32, m512 = med(rs, "pos", 32)[0], med(rs, "pos", 512)[0]
    if m512 > 0.9:
        return "GEN"
    if m32 > 0.9 and m512 < 0.6:
        return "shortcut"
    if m32 < 0.9:
        return "no-fit"
    return "mixed"


def fmt(v):
    return f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]" if v else "--"


def main():
    cs = load()
    print(f"{'cell':<26}{'n':>2}  {'pos32':>18}  {'pos512':>18}  {'masked512':>18}  tag")
    for k, rs in sorted(cs.items()):
        mdl, nh, init = k
        nm = f"{mdl} nh={nh} {init}"
        mk = med(rs, "pos_masked", 512)
        print(f"{nm:<26}{len(rs):>2}  {fmt(med(rs,'pos',32)):>18}  {fmt(med(rs,'pos',512)):>18}  {fmt(mk):>18}  {tag(rs)}")
    print("\n=== PRE-REGISTERED VERDICTS (PREREG_HYBRID.md) ===")

    full4 = cs.get(("m3", 4, "rand"), [])
    gate4 = cs.get(("m3gate", 4, "rand"), [])
    nob4 = cs.get(("m3nob", 4, "rand"), [])
    ex4 = cs.get(("m3", 4, "exact"), [])

    # P1 / K1
    if full4:
        m512 = med(full4, "pos", 512)[0]
        t = tag(full4)
        if m512 > 0.9:
            print(f"P1/K1: K1 FIRES (boundary) -- +b full GENs at nh=4 (median {m512:.2f}); pathology does NOT persist as collapse.")
        else:
            print(f"P1/K1: +b full at nh=4 = {t} (median pos512 {m512:.2f}) -- pathology persists (degree reported).")
    # P2 usage
    if full4:
        ratios = []
        for r in full4:
            bw, bt = r["usage_per_epoch"][-1]
            ratios.append(bt / max(bw, 1e-6))
        print(f"P2: trained transform-||b|| / write-||b|| ratio: median {st.median(ratios):.2f} "
              f"[{min(ratios):.2f},{max(ratios):.2f}]  (<=0.10 = self-gated; substantial = parasite)")
    # P3 / K2 masking
    if full4:
        d = [r["pos_masked"]["512"] - r["pos"]["512"] for r in full4 if "pos_masked" in r]
        mm = med(full4, "pos_masked", 512)
        print(f"P3/K2: masking b on transform tokens: pos512 {fmt(med(full4,'pos',512))} -> {fmt(mm)} "
              f"(median delta {st.median(d):+.3f}; K2 fires if <= +0.05 AND no restoration)")
    # P4 / K3 gated
    if gate4:
        m = med(gate4, "pos", 512)[0]
        sub = {nh: (med(cs.get(('m3gate', nh, 'rand'), []), 'pos', 512) or (0,))[0] for nh in (2, 3)}
        print(f"P4/K3: oracle-gated nh=4 median pos512 {m:.2f} ({'GEN as predicted' if m > 0.9 else 'K3 FIRES: gated also fails'}); "
              f"law cells nh=2: {sub[2]:.2f}, nh=3: {sub[3]:.2f}")
    # P5 -b
    if nob4:
        m32, m512 = med(nob4, "pos", 32)[0], med(nob4, "pos", 512)[0]
        print(f"P5: -b nh=4: pos32 {m32:.2f}, pos512 {m512:.2f} -> "
              f"{'found reservoir solution (GEN)' if m512 > 0.9 else 'representable-but-unreachable (echo of the note)' if m32 < 0.9 else 'partial'}")
    # P6 exact-init
    if ex4:
        post = med(ex4, "pos", 512)[0]
        mk = med(ex4, "pos_masked", 512)
        usage = st.median([r["usage_per_epoch"][-1][1] for r in ex4])
        print(f"P6: exact-init +b: post-train pos512 {post:.2f} ({'PULLED OFF' if post <= 0.9 else 'retained'}); "
              f"final transform-||b|| {usage:.3f}; masked512 {fmt(mk)}")
    # law
    for mdl in ("m3gate", "m3nob", "m3"):
        ms = {nh: (med(cs.get((mdl, nh, "rand"), []), "pos", 512) or (0,))[0] for nh in (2, 3, 4)}
        print(f"law [{mdl}]: pos512 by nh: 2={ms[2]:.2f} 3={ms[3]:.2f} 4={ms[4]:.2f}")


if __name__ == "__main__":
    main()
