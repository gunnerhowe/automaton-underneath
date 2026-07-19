# PRE-REGISTRATION — the two-gap training experiment (FROZEN)

*Committed after: spec approval, adversarial lit-refresh (56 sources; GDN-2 repositioning; dilution
rescope), K3 equivalence gates (chunk==sequential to 2e-5, re-verified at C=128), and the pilot
(4 runs). Committed BEFORE any grid run. Amendments after this commit will be disclosed as such.*

## Frozen configuration (from pilot)

30.0M-param DeltaNet-class LM (d=512, 8 layers, 8 heads, chunk C=128, GPT-2 init, tied emb, 8k BPE);
bs16 x accum2 (16,384 tok/step), 18,300 steps = **300M tokens/run**, AdamW 3e-4 cosine, bf16;
**lambda = 3e-3** (pilot: fastest gate decline 0.826->0.796 by step 800, monotone in lambda,
matched-or-best ppl 15.40 vs arm-A 15.45). Evals every 2,000 steps: tracking pack (families x depths
2-32; train sees 1-8), MQAR (4/8/12 pairs), TinyStories val ppl, gate stats. Throughput 12.5-14k
tok/s (pilot) -> ~6h/run. Pilot artifacts retained in runs/ (disclosed; tags *pilot*).

## Grid (n_h=1; single seed; run order = decision-relevance)

m50: A, B, C, B' -> m2: A, B, C -> m10: A, B, C -> m0: A, B, C   (13 runs, ~3.3 days)
Then the n_h=2 pair (A, B at m50) via the interleaved-microstep construction, which must first pass
the same equivalence gate vs a sequential n_h=2 reference (disclosed as implemented-after-freeze).
**Seed policy (frozen):** any cell that decides a positive P-verdict is replicated at seed 1 before
the verdict is claimed; negatives stand on seed 0.

## Predictions

- **P1 (dilution law):** the oracle-gating advantage on deep tracking (depths 16/32, boxes family)
  is DECREASING in mixture rate: at m2, C - A > +0.10 accuracy; at m50, A may approach C (per
  2602.14814's pure-synthetic result). The interesting cell is scarcity.
- **P2 (economics -> hygiene, the headline bet):** by end of training at m50, arm B's learned gate
  (i) reaches mean < 0.5, and (ii) shows the HYGIENE SIGNATURE: mean g on op tokens < mean g on
  content/other tokens (measured against the shard oracle masks, which the B arm never sees).
  B' (unpriced) shows a smaller or no op-vs-other gap. Nobody has reported either statistic.
- **P3 (two-gap link):** at m0 (pure LM), B >= A on MQAR by > +0.03 at matched ppl (within 3%) —
  write hygiene buys recall independent of tracking data.
- **P4 (composition controls the reveal):** inference-time op-token write-damping (Step-2 machinery)
  on trained arm-A checkpoints improves deep tracking at m50 and does nothing at m0.
- **P5 (ceiling):** C > A on deep tracking at every mixture > 0.
- **P6 (guardrail):** B within 3% val ppl of A at every mixture.

## Kills

- **K1 (economics dead):** B's final gate mean > 0.9 OR no op-vs-other gap beyond +-0.02 -> the
  priced-gate hypothesis dies; report and do not fish other lambdas (the pilot already ranged 10x).
- **K2 (hygiene irrelevant):** C == A (within +-0.05 deep-tracking) at EVERY mixture including m2 ->
  write discipline does not matter for emergence at this scale; the line closes as a clean negative.
- **K3 (price too damaging):** B ppl cost > 5% vs A at any mixture -> one disclosed rerun at
  lambda=1e-3 replaces the B column; no further tuning.
- **Contingency (frozen):** if the m50 A arm fails deep tracking, its checkpoint gets the
  state-passing fine-tune (0.1% budget, 2507.02782 recipe) before any write-path attribution.

## AMENDMENT 1 (pre-results; infrastructure only)

First grid launch (16:00) wedged before step 2,000 via silent WDDM VRAM spill (log + empty run dir
retained as evidence): bs16 x accum2 exceeds 10GB in sustained runs even though the short pilot fit.
NO grid results existed at amendment time. Changes: **bs12 x accum2** (12,288 tok/step; 24,400 steps;
same 300M tokens, same lambda, same evals), `set_per_process_memory_fraction(0.90)` so overflow
crashes loudly instead of spilling silently, 200-step heartbeat lines, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the launcher. No scientific parameter changed.

## AMENDMENT 2 (pre-results; ADDITIVE instrumentation for the emergence-forecasting fusion)

Added while the grid is paused (zero completed runs): (i) per-eval checkpoints are RETAINED
(ckpt_step{N}.pt; ~18GB total) instead of overwritten — the per-run time series a forecaster needs;
(ii) trilogy-style leading-indicator probes logged at every eval (within-minus-between-class cosine
gap of pre-head hidden states at the answer position, toggle and dial families; init baseline is
nonzero from random features — the signal is the rise). NO training parameter changed. Purpose:
optionality for a planned fusion with the mem-gen-delay probe machinery ("forecast per-run WHEN
tracking emerges as a function of gate pricing and dilution"), whose analysis and novelty lit-check
are explicitly DEFERRED until the user's concurrent project completes; no fusion claim is part of
this pre-registration.

## Reporting

All cells reported regardless of direction; verdict analyzer mechanical; artifacts one JSON per run
(git-stamped); public repo sync at grid completion. The GDN-2 positioning from the lit-refresh
governs all claim language: we test the ECONOMICS of write gating, not its existence.

**Amendment 2 addendum (pre-results):** probe cadence densified to every 500 steps (+ shallow depth-2 behavioral signal) per the paper-4 composed-anchor recipe; logged to probes.jsonl per run. Additive only.
