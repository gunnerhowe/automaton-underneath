"""Byte-verification for the implant note: re-runs gen_numbers.py, byte-compares numbers.json and
paper/numbers.tex, then confirms every \\im* macro used in paper/main.tex is defined.
Run: python verify_regen.py"""
import re
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    oj, ot = ROOT / "numbers.json", ROOT / "paper" / "numbers.tex"
    if not oj.exists() or not ot.exists():
        print("FAIL: run gen_numbers.py first.")
        sys.exit(1)
    old_json, old_tex = oj.read_text(), ot.read_text()
    r = subprocess.run([sys.executable, str(ROOT / "gen_numbers.py")], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: gen_numbers.py errored:\n" + r.stderr[-2000:])
        sys.exit(1)
    if oj.read_text() != old_json or ot.read_text() != old_tex:
        print("FAIL: regenerated numbers differ from stored.")
        sys.exit(1)
    # spot-check Tier-3 macros directly against the lmimplant raw artifacts
    import json
    N = json.loads(oj.read_text())
    LMW = ROOT.parent / "lmimplant"
    c0 = json.loads((LMW / "runs_wide" / "none_p0_s0.json").read_text())
    assert N["LmwBehNoneZeroEnd"] == c0["curve"][-1]["beh_toggle8"], "LmwBehNoneZeroEnd mismatch"
    assert N["LmwIntNoneZeroEnd"] == c0["curve"][-1]["internal_toggle"], "LmwIntNoneZeroEnd mismatch"
    assert N["LmwResurrectNoneZero"] == c0["resurrection_beh8"], "LmwResurrectNoneZero mismatch"
    Fj = json.loads((LMW / "facts.json").read_text())
    assert N["LmwHeadOne"] == Fj["head_cost_pct"]["1"] and N["LmwWidenImplantPct"] == Fj["widen_implant_pct"]
    defined = set(re.findall(r"\\newcommand\{\\(im\w+)\}", old_tex))
    used = set(re.findall(r"\\(im[a-zA-Z]+)", (ROOT / "paper" / "main.tex").read_text()))
    missing = {u for u in used if u not in defined}
    if missing:
        print(f"FAIL: macros used but not defined: {sorted(missing)[:10]}")
        sys.exit(1)
    unused = sorted(defined - used - {"imGridTableRows", "imLmwBatteryRows"})
    print(f"PASS: every cited number matches regeneration. ({len(used)} macros used; "
          f"unused: {unused if unused else 'none'})")


if __name__ == "__main__":
    main()
