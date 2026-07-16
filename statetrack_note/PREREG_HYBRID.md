# PRE-REGISTRATION — Hybrid bridge experiment (Step 1: content + transitions)

*Committed BEFORE any grid run exists. The posted note (arXiv submit/7832443; commits 2cf13e5 →
84e21c8 → 6fad018) showed the additive input path b_t is a parasitic attractor on PURE state-tracking
tasks, where b has no legitimate job. This experiment asks the bridge question: does the pathology
persist when the additive path has a REAL content-carrying job? Both outcomes are findings: persists
→ first evidence the mechanism matters beyond transition-only tasks; vanishes → the toy result is
boundary-scoped and the LM extrapolation dies here.*

## Task: signed-register tracking (regs5) — load-then-track

5 registers each holding a sign. Sequence = WRITE PHASE (5 tokens: each register set once to a random
sign, random register order, from empty) then TRANSFORM PHASE (T tokens: transposition (0 1) or
5-cycle on register POSITIONS — the S5 generators from the note, now transporting content). Per-step
target: the sign at position 0 (3 classes: +, −, empty; empty occurs only pre-write). Vocab 12 =
10 write tokens (register × sign) + 2 transform tokens. Train T=32; evaluate per-position to T=512.
Design notes fixed in advance: (i) values are SIGNS (1-dim content/register) so the transform group's
defining-rep reflection lengths are unchanged (transposition 1, 5-cycle 4) and the law prediction
carries over (min n_h = 4); (ii) load-once-from-empty is chosen because it keeps EXACT solutions
representable in BOTH architectures — +b writes via the additive path; −b writes via reflections that
rotate per-register reservoir mass into ±e_reg from a mass-split h0 (the "reservoir" construction) —
so the capacity-preserving ablation logic survives; (iii) the OVERWRITE variant (re-writing non-empty
registers, which orthogonal transitions provably cannot do) is explicitly DEFERRED — not part of this
pre-registration.

## Protocol

Identical to the note unless stated: d=112, per-step linear readout (3 classes), 40 epochs, n=4000,
batch 512, Adam 3e-3, train data seed 1000+s, eval seed 9000+s, n=1000 eval sequences, 5 seeds
(0–4) per primary cell, medians with [min,max], probe positions = transform-phase offsets
{32,64,128,256,512} (absolute 5+p). GEN = median transform-pos-512 accuracy > 0.9; shortcut = fits
pos-32 (>0.9) but <0.6 at 512; no-fit = pos-32 < 0.9. torch.manual_seed(seed) before model build.
The generator is validated by assertion against the note's marble semantics (a lone + tracks exactly
the s5 word problem) and by brute-force simulation.

## Arms (grid)

- **A1 (+b full):** DeltaProduct n_h ∈ {2,3,4} × 5 seeds. Probes recorded on every run: (a) mean
  ‖b_t‖ split by token type (write vs transform tokens, train-length batch); (b) MASKED-INFERENCE
  eval — b zeroed on TRANSFORM tokens only (its legitimate write-phase job untouched).
- **A2 (oracle-gated b):** same model but b architecturally masked on transform tokens during
  training and inference (b active only where the task needs it). n_h ∈ {2,3,4} × 5 seeds.
- **A3 (−b):** no additive path at all (reservoir solution representable). n_h ∈ {2,3,4} × 5 seeds.
- **A4 (exact-init +b):** hand-constructed exact solution (b-column writes; dead-axis reflections on
  write tokens; the note's 4-reflection 5-cycle program on transforms; W_b zero on transform-token
  columns; h0 = 0; 3-class readout with empty-bias). Construction asserted (composed matrices =
  block permutations; write vectors exact; pre-train eval ≥ 0.999 at T=512 — instrument gate).
  Trained with the standard budget, per-epoch usage probe. n_h=4 × 5 seeds.
- **A5 (GRU reference):** 3 seeds.

## Predictions and kills (committed now)

- **P1 (persistence vs boundary):** the parasite account predicts A1 (+b full) SHORTCUTS on the
  transform dimension (fits T=32, collapses by T=512) at every n_h, as in the note.
  **K1 (boundary kill):** if A1 length-generalizes at n_h=4 (median > 0.9), the pathology does NOT
  persist under legitimate content pressure — the toy result is boundary-scoped; we report that as
  the headline finding of this experiment and stand down the LM extrapolation.
- **P2 (usage):** in A1, trained ‖b_t‖ on TRANSFORM tokens is substantial (parasite); the
  legitimate-pressure alternative predicts SGD self-gates (transform-token ‖b‖ ≈ 0, ≤10% of
  write-token ‖b‖). Reported as the ratio transform-‖b‖ / write-‖b‖.
- **P3 (the probe):** masking b on transform tokens at inference in trained A1 models RESTORES or
  substantially improves transform-length generalization (the hybrid "automaton underneath").
  **K2:** if masking does not improve pos-512 (≤ +0.05 median), the concealment mechanism does not
  transfer; the mechanism section of any write-up reverts to open.
- **P4 (gating ceiling):** A2 (oracle-gated) length-generalizes exactly at n_h=4 and obeys the law
  (n_h=2,3 no-fit/short). **K3:** if A2 also shortcuts at n_h=4, restricting b to its legitimate role
  is INSUFFICIENT — the simple parasite account fails on hybrid tasks regardless of P1-P3.
- **P5 (−b learnability):** genuinely open, both directions recorded in advance: A3 either finds the
  reservoir solution (GEN — the full ablation story extends wholesale) or cannot fit within budget
  (content-loading via reflections is representable but unreachable — itself a capacity≠learnability
  echo). No kill either way; the outcome scopes the ablation's practicality.
- **P6 (exact-init):** A4: Adam grows transform-token b usage from 0 and degrades OOD (parasite), or
  keeps it ≈0 and stays exact (legitimate-pressure). Branch reported either way.
- **Law check:** for whichever of A1-mask/A2/A3 generalizes, min n_h = 4 per the defining-rep law.
- **Contingency (pre-registered):** one disclosed 3× epoch extension for any −b/gated cell at
  n_h=4 that no-fits, to separate budget from reachability. Nothing else changes.

## Reporting

All outcomes reported regardless of direction, same discipline as the note (counts, medians,
min–max; kill-fires stated plainly). Artifacts in statetrack_note/runs_hybrid/, one JSON per run,
idempotent runner, git-hash-stamped; analysis via analyze_hybrid.py with byte-verification.
