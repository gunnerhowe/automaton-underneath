"""Byte-verification for the write-economics paper: re-runs gen_numbers.py, confirms numbers.json
and paper/numbers.tex are byte-identical to what is committed/used by the manuscript, then confirms
every \\tg* macro referenced in paper/main.tex is defined. PASS required before any release.
Run: python verify_regen.py"""
import re
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    oj = (ROOT / "numbers.json")
    ot = (ROOT / "paper" / "numbers.tex")
    if not oj.exists() or not ot.exists():
        print("FAIL: numbers.json / paper/numbers.tex missing — run gen_numbers.py first.")
        sys.exit(1)
    old_json, old_tex = oj.read_text(), ot.read_text()
    r = subprocess.run([sys.executable, str(ROOT / "gen_numbers.py")], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: gen_numbers.py errored:\n" + r.stderr[-2000:])
        sys.exit(1)
    if oj.read_text() != old_json or ot.read_text() != old_tex:
        print("FAIL: regenerated numbers differ from stored (stale numbers vs artifacts).")
        sys.exit(1)
    main_tex = ROOT / "paper" / "main.tex"
    defined = set(re.findall(r"\\newcommand\{\\(tg\w+)\}", old_tex))
    used = set(re.findall(r"\\(tg[a-zA-Z]+)", main_tex.read_text()))
    missing = {u for u in used if u not in defined}
    if missing:
        print(f"FAIL: macros used in main.tex but not defined: {sorted(missing)[:10]}")
        sys.exit(1)
    unused = sorted(defined - used - {"tgGridTableRows"})
    print(f"PASS: every cited number matches regeneration from the run artifacts. "
          f"({len(used)} macros used; unused: {unused if unused else 'none'})")


if __name__ == "__main__":
    main()
