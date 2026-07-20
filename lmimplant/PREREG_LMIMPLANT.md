# PRE-REGISTRATION — exact circuit implant inside a real LM layer (FROZEN before the battery)

*Committed AFTER: feasibility (probe_construct.py, isolation toggle 1.000/depth-64), the coexistence
obstacle characterization (SPEC "Construction findings"), and the two substrates' construction gates
BELOW; committed BEFORE any continued-training run. Amendments disclosed.*

## Two substrates (both constructed, gates passed)

- **Track A — as-is implant** (lm_implant.py): the exact toggle circuit constructed into the
  TRAINED arm-A model's block-0 head 1 (cheapest head, +0.3% ppl). Behavioral readout is blocked by
  the dense-stream RMSNorm wall (documented). Integrity is measured INTERNALLY (the donated head's
  o at the check position = lamp sign). **Concealment result (no training needed):** the circuit is
  internally exact yet behaviorally silent -- an implant a model does not emit.
- **Track B — widened implant** (lm_implant_wide.py): trained model + one inert 64-dim head
  (H 8->9, d 512->576; zero-pad widening verified identity, -0.13% ppl). The exact toggle circuit
  lives in the fresh head + fresh dims; behavioral readout is CLEAN (fresh dims are exactly 0 for
  real tokens, so no RMSNorm amplification). KL1 gate PASSED: toggle 1.000 to depth 64, ppl -0.10%.

## Continued-training battery (Track B primary; Track A integrity logged alongside)

Continue training the implanted widened model. Mixtures p in {0.0, 0.1, 0.5} of dedicated-token
toggle sequences vs pure TinyStories. Arms (honest to THIS architecture -- the toy's additive-path
`-b` lever does not exist in the twogap delta rule, so it is dropped and the arms are re-mapped):
- **none:** all weights trainable (the field-default; does continued LM training erode a
  behaviorally-emitting implant?).
- **freeze-implant:** the head-8 implant parameters (its wq/wk/wv/wb rows, conv channels, wo answer
  row) are frozen; the rest of the model trains (does surrounding drift break a frozen circuit?).
Budget: 3000 steps, bs12 x accum2, lr 3e-4, eval every 300 steps: toggle-8 (behavioral, Track B;
internal, Track A), TinyStories val ppl. Seeds: seed 0; any positive verdict replicated at seed 1.

## Predictions (frozen)

- **IL1 (erosion under repurposing):** under `none` + p=0 (pure TinyStories, implant never
  exercised), behavioral toggle-8 decays to <= 0.60 by step 3000 -- continued training repurposes
  the free capacity, eroding an unexercised implant. (The Tier-2 acquisition-resistance result
  predicts the pretrained basin resists NEW skills; this tests whether it also fails to PRESERVE an
  inserted one when the capacity is otherwise useful.)
- **IL2 (exercise protects):** under `none`, end toggle-8 is nondecreasing in p; at p=0.5 it stays
  >= 0.90 -- exercised implants persist (the toy protection law's exercise half, at LM scale).
- **IL3 (freeze preserves):** freeze-implant retains toggle-8 >= 0.90 at every mixture, at ppl cost
  within +2% of the `none` arm -- freezing the circuit's own weights preserves it against
  surrounding drift.
- **IL4 (resurrection):** for any `none` cell whose behavioral toggle decays to <= 0.60 while the
  INTERNAL head-state toggle stays >= 0.90 (concealed, not erased), re-pinning ONLY the frozen
  answer-readout weights restores behavioral toggle to >= 0.90 -- concealment is recoverable at LM
  scale.

## Kills

- **KL2 (implant not preservable):** if EVERY arm at EVERY mixture erodes internal head-state
  toggle to < 0.90 by step 3000 -> "exact circuits are not preserved by realistic LM training under
  any tested protection" (clean negative, reported; sharpens the edit-durability literature).
- **KL3 (exercise degenerate):** if p>0 training collapses ppl (> +10% vs base) -> the
  dedicated-token exercise is too disruptive; report and drop the mixture arm.

## Reporting

All cells reported regardless of direction; analyzer mechanical; one JSON per cell, git-stamped;
post-hoc analyses labeled exploratory. Numbers regenerate from artifacts.
