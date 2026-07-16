# SPEC (for review — NOT yet pre-registered): the two-gap training experiment

*Goal: test whether disciplining the write path DURING TRAINING makes (1) state tracking and
(2) recall better in small linear-RNN language models — the two gaps that share the
undisciplined-write root cause — and whether the "automaton underneath" reveal reappears when
training data forces tracking. This is the missing middle rung between the toy results and any
at-scale proposal. Pre-registration follows AFTER this spec is approved and a throughput pilot has
calibrated budgets; predictions/kills below are drafts to be frozen then.*

## The three questions

- **Q1 (tracking):** does a learned or oracle write-gate make state tracking *emerge* in LM training
  where standard training fails or shortcuts?
- **Q2 (recall):** does write-gating improve fixed-state recall (less state pollution → less
  interference), at zero or acceptable LM-quality cost?
- **Q3 (the boundary, from the user's checkpoint question):** when the data mixture forces tracking,
  does the inference-time reveal (damp writes on op tokens → generalization appears) come back? This
  locates where on the toy→LM gradient the concealment effect dies: width, task pressure, or data.

## Architecture (torch-native; no triton — Windows-safe)

DeltaNet-class recurrent LM implemented from scratch in pure PyTorch with **chunk-parallel training**
(the WY/chunked delta-rule formulation: O(L/C · C²) matmuls, no sequential python loop — this is the
piece that makes 3080 training feasible; it gets validated against a sequential reference
implementation, gate-style, before any run). Config: d_model 512, 8 layers, ~50M params, 8k BPE
vocab, seq len 512, bf16. State update per head: S ← S(diag(w) − β k k̂ᵀ) + g_t · β v k̂ᵀ with
β ∈ [0,2] (negative-eigenvalue range — tracking capacity per Grazzi) and **g_t the write gate**
(the experimental variable). n_h ∈ {1, 2} (2 = one DeltaProduct-style second factor, covers
S₃-class tracking per the representation law; S₅-class is explicitly out of scope at this width).

## Arms (the write-path axis)

- **A. Standard**: g_t ≡ 1 (the field's default; the parasite unconstrained).
- **B. Learned gate**: g_t = σ(w_g·x_t) with an L1 sparsity penalty λ‖g‖ — the model must *pay* to
  write ("learned hygiene"). λ swept in the pilot, frozen at prereg.
- **C. Oracle gate**: g_t from data annotations (writes allowed on content tokens, suppressed on
  operation tokens of the synthetic segments; free elsewhere) — the upper bound, mirrors the hybrid
  experiment's gated arm.

## Data (the task-pressure axis)

Base corpus: TinyStories (public, standard for 50M-class LMs). Synthetic tracking corpus rendered as
natural text from our generators: box-swaps (S₃ tier: 3 boxes), light-switch toggles (ℤ₂),
mod-3 counters — with held-out DEEPER instances for length/depth generalization. Mixture rates
**{0%, 10%, 50%}** of synthetic in the stream = the pressure axis. (0% answers "does gating help
plain LM quality/recall alone"; 50% answers "when cornered, who learns the automaton".)

## Grid and budget (to be re-costed in the pilot)

Arms {A,B,C} × mixture {0,10,50} × n_h {1} (+ n_h=2 at mixture 50 only) × 1 seed first pass
= ~12 runs; each ~0.5–1.5 days on the 3080 depending on measured tokens/sec → **2–3 weeks
wall-clock**, runs queued sequentially, checkpoints every 2k steps. A second seed only on the
decision-relevant cells (prereg will name them). Pilot first: 2-hour throughput + λ-sweep run,
then freeze the prereg.

## Evals (every checkpoint)

1. **Tracking**: boxes-S₃ / toggles / mod-3 at training depth and 2–8× deeper (held-out) —
   accuracy + tracking-rate (the gold-vs-initial decomposition from Step 2a).
2. **Recall**: MQAR-style key-value probes (5–20 pairs, in-format), single-token scoring — the gap-2
   metric; plus the plain boxes depth-0 recall.
3. **LM quality**: held-out TinyStories perplexity (the cost guardrail — a gate that buys tracking by
   lobotomizing the LM is a failure).
4. **The reveal probe (Q3)**: on trained A-arm checkpoints, damp v-writes on op tokens at inference
   (Step 2b machinery, reused) — does the toy's concealment/restoration reappear under high mixture?
5. **Gate behavior** (B arm): learned g_t vs token type — does learned hygiene discover the oracle
   pattern (write on content, silence on ops)?

## Draft predictions (to be frozen at prereg, after pilot)

- P1: at mixture 50%, C (oracle) > A (standard) on deep tracking by a real margin; the toy/hybrid
  results predict A shortcuts at depth.
- P2: B (learned) lands between A and C, and its gate correlates with token type (the mechanism
  becoming a *trainable* design rule — the headline if it holds).
- P3 (two-gap link): B/C ≥ A on MQAR recall at equal perplexity (write hygiene reduces interference).
- P4 (Q3): the reveal effect reappears on A-arm checkpoints at mixture 50% and is absent at 0% —
  task pressure, not width, is the live variable at this scale.
- Draft kills: K1 = C ≈ A on tracking at mixture 50% (gating doesn't matter when data forces the
  skill → the toy mechanism is width-bound, not general; report and stop the line). K2 = B pays >15%
  perplexity for its gains (hygiene too expensive as a learned rule). K3 = chunk-parallel
  implementation fails equivalence vs the sequential reference (instrument kill; fix before any run).

## Honest scope

50M params, TinyStories-class data, S₃-tier tracking: this validates or kills *design rules*, not a
frontier claim. The known counter-scenario (PD-SSM-style: other architectures track fine with
additive paths) is carried forward — claims stay parameterization-scoped. The field is moving
(erase/write decoupling at Qwen; DeltaNet variants monthly): a lit-refresh on "write gating /
sparse state updates in linear RNNs" runs BEFORE the prereg freeze, kill-check discipline as always.
