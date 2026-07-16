# PRE-REGISTRATION — Step 2b: value-write damping in a pretrained RWKV-7 LM

*Committed BEFORE any intervention run. Instrument validation already run and disclosed (baselines +
error decomposition, commit b585c68): mamba-1.4b fails the recall precondition (0.35@d0; known
MQAR-type limitation); RWKV-7 World 1.5B passes the gate (0.55@d0 → 0.13@d8, chance 0.20) with
STRUCTURED errors — at depth 8 it picks the query box's INITIAL contents 61% vs gold 13%: recall
works, swap-application fails. RWKV-7 World 2.9B fails the headroom gate in this format (0.30@d0);
reported as model-format sensitivity. Intervention target: RWKV-7 World 1.5B.*

## Hypothesis under test (calibrated by the hybrid bridge experiment)

Graded masked capacity: damping the VALUE-WRITE of transform-describing tokens (swap sentences)
should yield modest improvements in swap-application at moderate depths — not dramatic restoration
(the bridge experiment's K1 taught us content pressure tames the parasite to graded degradation).

## Intervention (mechanically faithful analog of the note's W_b masking)

RWKV-7's per-token state update is state ← state·diag(w) + state·(−k̂)(k̂a)ᵀ + v⊗k, all inside
`RWKV7_OP(state, r, w, k, v, a, b)`. We wrap RWKV7_OP to scale ONLY the v argument by γ on selected
token segments: the token's content-write into persistent state is damped; its decay/erase/route
action (w, k̂ terms) and its direct output path (r·k·v bonus, v_first mixing) are untouched — "this
token may transform memory but writes nothing into it." Segments are contiguous sentences; each
model.forward() call covers one segment with a scalar γ, eliminating mask-alignment risk. Gates
(instrument, must pass before the grid): G1 γ=1 wrap reproduces the unwrapped baseline exactly;
G2 per-segment tokenization concatenates to the whole-text tokenization on ≥99% of examples;
G3 γ=0 on ALL segments destroys accuracy (wrap demonstrably engages).

## Arms and grid (RWKV-7 World 1.5B; n=150/cell; depths {2,4,8,16}; seed 0 eval sets)

- **Baseline**: γ=1 everywhere (one cell per depth).
- **S (treatment)**: γ ∈ {0.5, 0} on the test example's swap sentences only.
- **C (content control)**: γ ∈ {0.5, 0} on the test example's five content sentences only —
  predicted to HURT recall (sign/sensitivity control: the tool must cut in the expected direction).
- **R (random control)**: γ ∈ {0.5, 0} on `depth` sentences sampled from the few-shot prefix
  (matched count, similar length; earlier stream position is a stated limitation) — predicted ≈ no
  effect on the test example.

28 cells total. Metrics per cell: (i) **primary: tracking rate TR = #gold-picks / (#gold-picks +
#initial-contents-picks)** — isolates swap-application from recall noise; (ii) raw accuracy;
(iii) initial-pick rate. Per-example picks stored in artifacts.

## Predictions and kills

- **P1 (the bet):** some S cell at depth ∈ {4, 8} improves TR by > +0.05 over BOTH its depth's
  baseline AND the matched R control. Calibrated expectation: +0.05 to +0.20, not restoration.
- **P2 (sign check):** C-damping at γ=0 reduces raw accuracy versus baseline at every depth (the
  legitimate writes are load-bearing; if this fails, the intervention isn't reaching the writes and
  all other cells are uninterpretable — instrument kill, fix before proceeding).
- **P3:** R ≈ baseline (|ΔTR| ≤ 0.05 median across depths).
- **K1 (Step-2 kill):** if NO (γ, depth) S cell clears P1's double margin, the masked-capacity
  hypothesis dies on this checkpoint → Step 2 closes as a clean negative for RWKV-7 1.5B (one
  paragraph in any write-up; the toy-to-LM extrapolation is bounded at the pretrained-checkpoint
  level). We do not fish additional checkpoints beyond the already-disclosed 2.9B/mamba gates.
- **K2 (decomposition kill):** if S at γ=0 collapses raw accuracy to ≤ chance at ALL depths, the
  value path in this checkpoint carries the swap-routing information too (the clean v-vs-routing
  separation of the architecture is not used that way by this trained model) — reported as a
  mechanism-mapping finding; P1 evaluated at γ=0.5 only.
- **Contingency (pre-registered):** if G2 fails (>1% tokenization mismatches), affected examples are
  regenerated with resampled objects (never re-scored); if G1 fails, fix the wrap before any cell runs.

All cells reported regardless of direction; artifacts in step2_lm/results_intervention/, one JSON per
cell, git-hash-stamped.
