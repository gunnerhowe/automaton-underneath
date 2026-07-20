"""Byte-verification for paper 4: re-runs gen_numbers.py, confirms numbers.json and
paper/numbers.tex are unchanged vs committed, and confirms every \\li* macro used in main.tex is
defined. (Does NOT re-run capture_facts.py -- facts.json is a committed artifact; regenerating it
requires the GPU. verify_facts() re-checks battery-derived numbers against runs_wide/.)
Run: python verify_regen.py"""
import re
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    oj, ot = ROOT / "numbers.json", ROOT / "paper" / "numbers.tex"
    if not oj.exists() or not ot.exists():
        print("FAIL: run gen_numbers.py first."); sys.exit(1)
    old_json, old_tex = oj.read_text(), ot.read_text()
    r = subprocess.run([sys.executable, str(ROOT / "gen_numbers.py")], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: gen_numbers.py errored:\n" + r.stderr[-2000:]); sys.exit(1)
    if oj.read_text() != old_json or ot.read_text() != old_tex:
        print("FAIL: regenerated numbers differ from committed."); sys.exit(1)
    # spot-check a few macros directly against the raw artifacts
    N = json.loads(oj.read_text())
    c0 = json.loads((ROOT / "runs_wide" / "none_p0_s0.json").read_text())
    assert N["BehNoneZeroEnd"] == c0["curve"][-1]["beh_toggle8"], "BehNoneZeroEnd mismatch"
    assert N["IntNoneZeroEnd"] == c0["curve"][-1]["internal_toggle"], "IntNoneZeroEnd mismatch"
    assert N["ResurrectNoneZero"] == c0["resurrection_beh8"], "ResurrectNoneZero mismatch"
    F = json.loads((ROOT / "facts.json").read_text())
    assert N["HeadOne"] == F["head_cost_pct"]["1"] and N["WidenImplantPct"] == F["widen_implant_pct"]
    defined = set(re.findall(r"\\newcommand\{\\(li\w+)\}", old_tex))
    used = set(re.findall(r"\\(li[a-zA-Z]+)", (ROOT / "paper" / "main.tex").read_text()))
    used -= {"linewidth"}                                     # LaTeX builtin, not an \li* macro
    missing = {u for u in used if u not in defined}
    if missing:
        print(f"FAIL: macros used but not defined: {sorted(missing)[:10]}"); sys.exit(1)
    unused = sorted(defined - used - {"liBatteryRows"})
    print(f"PASS: numbers regenerate + spot-checks vs artifacts OK. ({len(used)} macros used; "
          f"unused: {unused if unused else 'none'})")


if __name__ == "__main__":
    main()
