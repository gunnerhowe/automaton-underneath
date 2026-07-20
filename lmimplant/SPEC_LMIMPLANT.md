# SPEC (for review — NOT yet pre-registered): exact circuit implant inside a real LM layer

*Flagship #1. Question: can a verified-exact algorithmic circuit be constructed inside ONE layer
of a real trained language model, survive continued training, and reproduce the toy program's
fate/protection/resurrection findings at LM scale? Nobody has constructed an exact algorithmic
circuit into a trained LM layer (edit-durability edits facts; InterpBench/weight-sparse TRAIN
circuits in; steering is activation-space) — [[implant-persistence-result]] named this the
next-scale problem.*

## Feasibility: GO (probe_construct.py, verified)

Toggle is reflection-length 1, so it fits a SINGLE delta-rule head (n_h=1) — the LM's own toggle-8
metric is the natural target (S5 would need n_h=4, unavailable in one head). The construction
(lamp at state entry S[0,0]; content sets it, flip negates it via the b=2 erase term with v=0,
check reads it via q=e0) computes toggle at **1.000 accuracy through depth 64** inside the real
DeltaBlock forward (RMSNorm + SiLU short-conv + delta rule + residual + tied readout), in
isolation. The circuit exists and is exact.

## The one exposed sub-problem: reading out through trained layers

The isolation probe zeroed blocks 1-7. In the real model the implant sits in block 0 and its
answer signal must survive 7 trained layers + final norm. Resolution (design decision to freeze at
prereg): implant into block 0 in a **reserved subspace** (dims 500-504, where trained emb/weights
are ~0.02), and give the answer dim D_ANS a **protected pass-through**: zero only the D_ANS output
ROW of blocks 1-7 (wo, mlp_d) so they preserve it — one dim of 512, negligible LM cost. The
alternative (implant into block 7, readout immediately follows) is the fallback if pass-through
proves leaky. Both are within the "one constructed layer" framing.

## Arms and battery (mirrors the toy [[implant-persistence-result]], now at LM scale)

- **Substrate:** trained twogap arm-A checkpoint (A_m0_s0), toggle circuit constructed into block 0.
- **Coexistence gate (KL1, before anything):** post-implant, (a) toggle-8 readout = 1.00 and
  (b) TinyStories val ppl within +2% of the un-implanted model — the implant must not lobotomize
  the LM. Fail -> fall back to block-7 implant, disclosed.
- **Continued training:** mixtures p in {0, 0.1, 0.5} of dedicated-token toggle sequences vs pure
  TinyStories; protection arms {none (+b), -b (additive path off in block 0), freeze-core
  (implant projections frozen)}; measures every N steps: implant integrity (toggle-8 via the
  dedicated tokens), LM quality (val ppl), additive-path norm ||W_b e||, and the b-zero reveal.
- **Predictions (draft, to be frozen after a coexistence pilot):** LP1 the toy protection law
  survives — (-b)x(exercise) retains, +b conceals; LP2 the b-zero resurrection reproduces at LM
  scale (concealed implants restore); LP3 unexercised+unprotected erodes. Kills: KL1 coexistence
  (above); KL2 if the implant is destroyed by continued training under EVERY arm incl. -b+exercise
  -> "exact implants don't survive real LM training" (clean negative, reported).

## Why this is the flagship

If concealment + resurrection reproduce inside a real LM, the claim is groundbreaking and
two-audience: capability-engineering (install/protect an exact skill in a real model — DeepMind)
and safety (models can carry hidden, exactly-recoverable capabilities that continued training
conceals rather than removes — Anthropic). If the implant does NOT survive, that boundary is
itself the paper: exact circuits are not preserved by realistic LM training, sharpening the
edit-durability literature. Interesting either way.

## Construction findings (empirical, this build)

Building the coexistence implant surfaced real obstacles — themselves results:
1. **The circuit is exactly constructible in a real block** (probe_construct.py): toggle reads
   1.000 through depth 64 inside the full DeltaBlock forward, in isolation. Confirmed.
2. **The residual stream is DENSE — no free capacity.** True residual-activation std over real
   text has min 0.72 vs median 0.87 (dead_dims): unlike the toy's clean dedicated-dimension space,
   there is no dead subspace to hijack. Every implanted dim collides with language.
3. **Head donation is cheap and head-specific** (measured): donating block-0 head 1 costs +0.3%
   ppl, head 3/4/7 ~+0.5%, but head 0 +6.2% and head 6 +432% (induction-like); block-7 head-0
   +0.16%. Donate a measured-cheap head, never head 0 by default.
4. **THE WALL: behavioral readout through a dense RMSNorm'd tied-embedding stream.** Routing the
   lamp to the logits requires writing an answer dim, but the block's output RMSNorm amplifies the
   donated head's near-zero output on quiet real tokens, and no linear read can threshold it out.
   The check-signal / real-text-noise RATIO is fixed by the construction (scaling route/sout moves
   both), so the readout pollutes the LM (ppl explodes) at any gain that keeps toggle exact. This
   is a genuine obstacle to in-place behavioral implantation, not a tuning failure.

**Resolution (design decision):** measure implant INTEGRITY by reading the constructed head's
toggle state directly (does the circuit still compute the automaton?) — the faithful
"is-the-implant-intact" measure the fate/protection/resurrection battery actually needs — and
report the behavioral-readout obstacle (concealment by architecture) as a first-class finding.
The delta head's state is private (not shared residual), so this sidesteps the RMSNorm/tied-readout
wall entirely while keeping the science (does continued training preserve the exact circuit?).

## Honest scope

Toggle (Z2), one layer, 30M params, dedicated-token protocol (the skill is exercised via reserved
tokens, not natural English — parallel to the toy's vocab extension). Not a claim about arbitrary
skills or natural-language triggers; the construction for higher reflection-length tasks needs
multiple heads/layers (open).
