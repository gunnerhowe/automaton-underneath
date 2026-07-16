# Post-paper findings: everything learned since "The Automaton Underneath" was written

*Covers the hybrid bridge experiment (statetrack_note/PREREG_HYBRID.md, 53 runs) and Step 2
(step2_lm/, instrument + pre-registered intervention on a pretrained LM, 28 cells). All numbers
regenerate from committed artifacts; prereg chain: dfc4c90 (bridge) → 44f2f3c (bridge verdicts) →
b342cd7 (Step-2 prereg) → b065cc0 (gates) → this document.*

## 1. The hybrid bridge (content + transitions in one task)

**Question:** does the parasitic-additive-path pathology survive when the additive path has a
legitimate content job? **Answer: tamed, not gone — and the constructive result appeared.**

- K1 (boundary kill) fired as registered: +b models at law width reach median 0.93 [0.70, 0.99] —
  graded seed-dependent damage, not the pure task's collapse to chance.
- The mechanism persisted in attenuated form: SGD still feeds the parasite (transform-token ‖b‖ =
  56% of write-token); exact-init models still get pulled off (→0.89) with ‖b‖ growing 0→0.34; and
  W_b:=0 at inference still restores 1.00 on every pulled-off exact-init seed.
- **The constructive surprise:** the oracle-gated arm hit 1.00 [1.00, 1.00] and the −b arm LEARNED
  the reservoir solution (content loading via reflections; 4/5 seeds exact) — both beating the GRU
  reference (0.94). *Write-path discipline didn't just avoid the pathology; it produced the best
  models in the comparison.* This is the seed of the two-gap training experiment.

## 2. Step 2a — the instrument, and two findings we didn't order

- **Mamba-1.4b fails the recall precondition** (0.35 at depth 0 regardless of prompt format; the
  tiny 130m at chance): 5-pair associative recall saturates Mamba-1-class state management —
  consistent with the known MQAR limitation, measured here incidentally. Gap 2 is real and
  independent of gap 1.
- **RWKV-7 1.5B recalls but does not track:** 0.55 at depth 0 decaying to 0.13 by depth 8, with the
  error structure telling the story — 61% of deep answers are the box's INITIAL contents. The
  delta-rule state stores the writes and never applies the swaps. (RWKV-7 2.9B scored 0.30 at depth
  0 in the same format: base-model format sensitivity dominates parameter count; also the
  retroactive answer to "why not 7B".)
- Instrument upgrade: **tracking rate** = gold/(gold+initial-picks) isolates swap-application from
  recall noise.

## 3. Step 2b — the pre-registered intervention (K1 FIRED: clean negative)

Value-write damping (γ ∈ {0.5, 0}) on swap sentences, vs content-damping (sign control) and
random-damping (specificity control), depths {2,4,8,16}, n=150/cell, on RWKV-7 1.5B, with gates
proving the patch neutral at γ=1 (30/30 pick agreement with the untouched package) and engaged at γ=0.

- **P1 dead in all 8 cells.** No treatment cell cleared +0.05 tracking-rate over both baseline and
  random control. Best S-vs-baseline delta: +0.050 at d16/γ=0 — which simultaneously lost to its
  random control by 0.135.
- **P3 violated in the informative direction:** the random control itself fluctuates by median
  |ΔTR| = 0.099 at n=150 — control noise exceeds every treatment effect. The null is not marginal;
  it is decisive at this sample size.
- **P2 (content-damping hurts accuracy) held only at depths 2 and 16.** At depths 4–8 muting content
  slightly HELPED accuracy — because it collapses the initial-contents attractor (initial-picks
  crater from 53–79 to 14–31 in every γ=0 content cell). The damping engages powerfully; what it
  cannot do is conjure swap-application.
- **K2 did not fire** (γ=0 on swaps leaves fit intact — consistent with the model never using swap
  content to transform state in the first place).

**Verdict, as pre-registered: the masked-capacity hypothesis dies on this checkpoint. Inference-time
write-surgery cannot uncover state tracking in RWKV-7 1.5B, because pretraining never built it.**

## 4. The unified lessons (what feeds forward)

1. **Reveal ≠ create.** W_b:=0 restores generalization iff training already built the automaton
   underneath. The toy was cornered into building it (minimal width; b provably not load-bearing for
   fit). Pretrained LMs are never cornered: next-token loss has cheaper strategies, so the skill
   never forms and there is nothing to unmask. The effect died along the realism gradient exactly at
   the point where training pressure stopped forcing the skill: collapse (toy) → graded (content
   task) → absent (pretrained LM).
2. **The parasite is a training-time phenomenon; the fix is training-time.** Three independent
   demonstrations: exact-init pull-off (toy), exact-init pull-off under content (bridge), and the
   constructive gated/−b arms outperforming everything (bridge).
3. **The two gaps share a root**: undisciplined writes pollute a finite state — the tracking failure
   (this program) and the recall ceiling (Mamba here; MQAR literature) are both state-hygiene
   problems. Untested unification; the two-gap experiment tests it.
4. **Honest negatives with pre-registered kills are cheap and land fast.** Step 2 cost two days and
   one clean paragraph; it bounded the claim before a reviewer (or a lab) had to.

## 5. Status of the research program

- Paper: submitted to arXiv (ID pending). Public repo: github.com/gunnerhowe/automaton-underneath
  (paper + all artifacts + bridge + this Step-2 record).
- Next (spec'd, pending approval + pilot + lit-refresh, NOT yet pre-registered): the two-gap training
  experiment — SPEC_TWOGAP.md. Q1 gating→tracking emergence; Q2 gating→recall; Q3 does the reveal
  return under forced-tracking data mixtures.
