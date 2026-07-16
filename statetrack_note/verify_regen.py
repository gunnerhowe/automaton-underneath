"""Byte-verification for the statetrack note: regenerates numbers.json and paper/numbers.tex from the
run artifacts via analyze.py and confirms they match what is committed/used by the manuscript, then
confirms every \\st* macro referenced in paper/main.tex is defined. PASS required before posting.
Run: python verify_regen.py"""
import json
import re
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    old_json = (ROOT / "numbers.json").read_text() if (ROOT / "numbers.json").exists() else None
    old_tex = (ROOT / "paper" / "numbers.tex").read_text() if (ROOT / "paper" / "numbers.tex").exists() else None
    if old_json is None or old_tex is None:
        print("FAIL: numbers.json / paper/numbers.tex missing — run analyze.py first.")
        sys.exit(1)
    r = subprocess.run([sys.executable, str(ROOT / "analyze.py")], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: analyze.py errored:\n" + r.stderr[-2000:])
        sys.exit(1)
    new_json = (ROOT / "numbers.json").read_text()
    new_tex = (ROOT / "paper" / "numbers.tex").read_text()
    if new_json != old_json or new_tex != old_tex:
        print("FAIL: regenerated numbers differ from stored (stale numbers vs artifacts).")
        sys.exit(1)
    main_tex = ROOT / "paper" / "main.tex"
    if main_tex.exists():
        defined = set(re.findall(r"\\newcommand\{\\(st\w+)\}", new_tex))
        used = set(re.findall(r"\\(st[a-zA-Z]+)", main_tex.read_text())) - {"strut"}
        missing = {u for u in used if u not in defined}
        if missing:
            print(f"FAIL: macros used in main.tex but not defined: {sorted(missing)[:10]}")
            sys.exit(1)
    print("PASS: every cited number matches regeneration from the run artifacts.")


if __name__ == "__main__":
    main()
