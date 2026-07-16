# The Automaton Underneath

Code, pre-registration, run artifacts, and manuscript for:

> **The Automaton Underneath: The Additive Input Pathway Is a Parasitic Attractor for State Tracking
> in Householder Linear RNNs.** Gunner Levi Howe, July 2026. arXiv: *pending announcement*
> (submitted; ID will be added here).

**TL;DR.** In DeltaProduct-style linear RNNs, one deleted term — the additive input injection
$b_t = W_b e_t$ — flips state-tracking word problems (parity, $S_4$, $A_5$, non-solvable $S_5$) from
fit-then-collapse shortcuts to *exact* length generalization at 16× the training length. The minimal
number of Householder factors per token equals the generators' reflection length in the
format-pinned representation. Mechanistically, the additive path is *parasitic*: initialized at a
verified-exact solution, Adam grows it and pulls the model off; in all 49 seeds that fit with $b$,
zeroing $W_b$ at inference keeps the fit — and at law-minimal width it *restores* exact length
generalization. The exact automaton is learned underneath the parasite.

## Layout

```
agent/rnn_statetrack.py        model (DeltaProduct-style), tasks, train/eval protocol
statetrack_note/
  PREREG.md                    pre-registration: grid, predictions, kill criteria (committed before any run)
  runner.py                    idempotent grid runner (asserts group orders, reflection lengths,
                               exact-construction correctness before running)
  runs/                        202 JSON artifacts, one per run (the paper's ground truth)
  analyze.py                   artifacts -> cell medians, pre-registered verdicts, numbers, figures
  verify_regen.py              byte-verification: every number in the paper regenerates from runs/
  PREREG_HYBRID.md             pre-registration for the hybrid bridge experiment (verdicts recorded)
  runner_hybrid.py             its runner (signed-register load-then-track)
  runs_hybrid/                 its 53 artifacts; analyze_hybrid.py prints the pre-registered verdicts
                               (K1 boundary kill fired: content pressure tames the parasite to graded
                               degradation; attractor + concealment persist from exact init)
  paper/                       main.tex/pdf, references.bib, numbers.tex (all macros), figures,
                               arXiv abstract + upload zip, Zenodo metadata
```

## Verify the paper's numbers

```bash
pip install torch numpy matplotlib
python statetrack_note/verify_regen.py
# -> PASS: every cited number matches regeneration from the run artifacts.
```

`analyze.py` recomputes every macro in `paper/numbers.tex` from the raw artifacts in `runs/` and
prints the per-cell table plus the verdicts against the pre-registered predictions. To re-run any
grid cell from scratch, delete its JSON and run `python statetrack_note/runner.py` (idempotent; full
grid ≈ 53 min on one RTX 3080).

## Provenance of the pre-registration chain

This work was developed inside a private monorepo; the paper's Appendix A cites that history's
commit hashes, reproduced here for the record:

| commit    | date (UTC-5)     | content |
|-----------|------------------|---------|
| `2cf13e5` | 2026-07-13       | pre-registration (grid, predictions P-R0/E1/E2/E3/P3, kills K0–K2) + runner, **before any grid artifact existed** |
| `84e21c8` | 2026-07-13       | 202 run artifacts + verdicts recorded against the pre-registration, unchanged |
| `6fad018` | 2026-07-13       | manuscript |
| `6bd3846` | 2026-07-13       | posting package (per-seed precision fixes, arXiv abstract/zip) |
| `dfc4c90` | 2026-07-13       | pre-registration of the hybrid bridge experiment, before its grid |

This public repository mirrors the final state of those files (code, pre-registrations, all
artifacts, manuscript source). The private history remains the primary timestamp record; everything
needed to *check* the results — artifacts plus the regeneration/verification chain — is here and is
timestamp-independent.

## Citation

```bibtex
@article{howe2026automaton,
  author = {Gunner Levi Howe},
  title  = {The Automaton Underneath: The Additive Input Pathway Is a Parasitic Attractor
            for State Tracking in Householder Linear RNNs},
  year   = {2026},
  note   = {arXiv, ID pending}
}
```
