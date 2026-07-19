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

**Follow-up paper (this repo, `twogap/`):**

> **The Gate Learns Hygiene: The Training Economics of the Write Path in Linear-RNN Language
> Models.** Gunner Levi Howe, July 2026. Manuscript at `twogap/paper/main.pdf`.

**TL;DR.** A pre-registered 13-run grid (30M-param DeltaNet-class LMs, 300M tokens each) testing
whether write discipline buys recall or state tracking during training. An $L_1$ **price** on the
write gate makes it discover **write hygiene** unsupervised — mean gate 0.119 on state-operation
tokens vs 0.473 on content, recovering the oracle pattern with no role labels (first token-role
statistics of a learned write gate; dose-responsive in mixture; the unpriced gate is 2.8× more open
on operations). Hygiene buys a small recall lift at matched perplexity (both seeds, honest error
bars) but **does not buy automata**: the gating-helps-tracking prediction died sign-inverted, the
only skill that emerged arose in the *always-write* arm, and damping its op-token writes **destroys**
it — inverting the toy-regime reveal effect and completing a three-regime scope map
(reveal / repair / destroy). A frozen emergence-forecasting gate correctly declined to fire (0/7).

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
twogap/
  PREREG_TWOGAP.md             pre-registration for the write-economics grid (frozen before any run;
                               amendments disclosed inline)
  PREREG_FUSION.md             frozen emergence-forecasting protocol (composed anchor, blind cells,
                               kill criteria; KF3 fired -- disclosed)
  VERDICTS.md                  the scored verdicts for every pre-registered prediction + disclosures
  model.py / core_deltanet.py  30M DeltaNet-class LM, chunk-parallel delta rule (equivalence-gated),
                               write-gate arms A / B (L1-priced) / B' (unpriced) / C (oracle)
  data.py / train.py           TinyStories + role-annotated synthetic tracking families; trainer
  runs/                        per-run log.json + probes.jsonl (the paper's ground truth; 20 runs)
  hygiene.json                 the token-role gate statistics artifact (P2 headline)
  p4_results.json              the damping-intervention artifact (P4 inversion)
  gen_numbers.py / gen_figures.py / verify_regen.py   paper pipeline (byte-verified)
  paper/                       main.tex/pdf, references.bib, numbers.tex, figures
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
