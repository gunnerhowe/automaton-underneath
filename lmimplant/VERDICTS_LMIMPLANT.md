# LM-IMPLANT — VERDICTS (vs PREREG_LMIMPLANT, frozen 8d… / commit before battery)

*Track B widened substrate; 6 cells (arms {none, freeze-implant} x mixtures {0, 0.1, 0.5}), seed 0,
3000 continued-training steps each; artifacts runs_wide/*.json. Track A concealment established at
construction time (no training). Behavioral toggle = via logits; internal toggle = the implanted
head's o at the check position read directly.*

## Results

| cell | behavioral toggle-8 (start→end, min) | internal toggle (start→end, min) | ppl | resurrection |
|---|---|---|---|---|
| none  p0  | 1.00 → **0.52** (0.0) | 1.00 → **1.00** | 3.41 | **1.00** |
| none  p10 | 1.00 → **1.00** (1.0) | 1.00 → 1.00 | 3.47 | 1.00 |
| none  p50 | 1.00 → **1.00** (1.0) | 1.00 → 0.45 | 3.72 | 1.00 |
| freeze p0  | 1.00 → **0.00** (0.0) | 1.00 → **0.00** | 3.42 | **0.00** |
| freeze p10 | 1.00 → 1.00 (0.48) | 1.00 → 0.45 | 3.49 | 1.00 |
| freeze p50 | 1.00 → 1.00 (1.0) | 1.00 → 0.76 | 3.74 | 1.00 |

## Frozen verdicts

- **IL1 (erosion when unexercised): CONFIRMED.** none/p0 behavioral toggle decays 1.00→0.52 (≤0.60)
  under pure-LM continued training — the free capacity gets repurposed.
- **IL2 (exercise protects): CONFIRMED (behaviorally).** none end-toggle nondecreasing in mixture
  (0.52 → 1.00 → 1.00); p=0.5 stays 1.00. A 10% exercise trickle fully preserves behavior.
- **IL3 (freeze preserves ≥0.90 everywhere): NOT CONFIRMED — and the failure is a finding.**
  freeze/p0 died to 0.00 AND was unrecoverable, *worse* than the all-trainable none/p0 (internal
  1.00). Cause: the freeze arm pins the implant's WEIGHTS but not its INPUT ENCODING — the
  dedicated-token embeddings drift under tied-readout gradients even with zero exercise, so the
  frozen circuit reads a moving input and its internal computation breaks (internal 1.00→0.00).
  Lesson: preserving an implant requires pinning its input/output INTERFACE, not just its weights.
- **IL4 (resurrection): CONFIRMED where applicable.** Every cell whose behavior decayed while the
  internal circuit stayed intact (none/p0: beh 0.52, internal 1.00) resurrects to 1.00 by re-pinning
  only the answer-readout weights. freeze/p0 (internal ALSO died to 0.00 = genuinely eroded, not
  concealed) correctly does NOT resurrect (0.00) — the concealed/eroded taxonomy separates cleanly
  on the internal read.
- **KL2 (implant not preservable anywhere): does not fire** (none/p10 fully preserves).
- **KL3 (exercise collapses ppl >+10% vs base): technically fires at p50 (+20%) but confounded** —
  the no-exercise none/p0 cell already drifts +11% from 3000 steps of continued AdamW at lr 3e-4,
  so most of the rise is generic continued-training drift, not exercise. Disclosed as a
  training-setup artifact (see limitations); the exercise-specific increment is ~+9% (p50 vs p0).

## The two unpredicted phenomena (exploratory, labeled)

1. **Concealment forms during training, and is recoverable (the headline).** none/p0: 3000 steps of
   pure-LM training drive behavioral toggle to chance while the implanted automaton stays
   internally EXACT (1.00), and a one-line readout re-pin restores it. Continued training did not
   destroy the circuit — it disconnected it from the output. This reproduces the toy program's
   concealment→resurrection law inside a real language model, on a deliberately installed skill.
2. **Skill diffusion under heavy exercise.** none/p50 & freeze/p50: behavioral toggle stays 1.00
   while the internal-head-only read drops (0.45–0.76) — 50% exercise spreads the skill BEYOND the
   implanted head into redundant routes, so the single-head read no longer reflects behavior. The
   implant seeds a capability the surrounding model then distributes.

## Track A (as-is, concealment by construction)

Established without training (lm_implant.py + SPEC findings): the exact circuit constructed into the
unmodified model's block-0 head 1 is internally exact (+0.3% ppl) but behaviorally silent — the
dense-stream RMSNorm blocks readout. An implant a model provably holds but cannot emit; the
architectural analog of the training-formed concealment in Track B.

## Synthesis

An exact algorithmic capability can be installed in a real LM at zero language cost (widened
substrate, KL1); it survives realistic continued training but gets behaviorally concealed unless
exercised (~10% suffices); concealment is recoverable by a one-line intervention while true erosion
is not, and the internal read distinguishes them; and heavy exercise diffuses the skill beyond the
implant. **Safety framing:** models can carry installed capabilities that ordinary training hides
rather than removes, recoverable on demand — concealment ≠ removal.

## Limitations
Toggle (Z2), one implanted head, 30M params, single seed, 3000 steps, dedicated-token protocol; the
continued-training ppl drift (+11% at zero exercise) indicates the lr/setup is not tuned for
minimal-drift continuation (does not affect the toggle-integrity findings, which are within-run
relative). Higher reflection-length skills need multiple heads (open). The freeze arm's
input-encoding confound is reported as a finding, not corrected.
