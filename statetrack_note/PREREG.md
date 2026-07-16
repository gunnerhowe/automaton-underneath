# PRE-REGISTRATION — "The additive route" note (statetrack_note)

*Committed BEFORE any grid run exists. Amendments after this commit will be disclosed as such.*

## Background and disclosure of prior data

June 2026 exploratory results (committed: `agent/rnn_statetrack.py`, `rnn_ablate_b.py`, `rnn_s5.py`,
`STATETRACK_FINDINGS.md`, 3 seeds, mean-based tags): (a) Householder linear RNNs (DeltaProduct-style,
d=112, n_h input-dependent reflections/token, per-step linear readout) with the additive injection
`b_t = W_b e_t` fit L=32 state-tracking tasks and collapse OOD (shortcut); (b) removing `b_t` flips
parity (all n_h) and S5 (n_h=4) to exact generalization at L=512; (c) mod-3 flips no-fit→fits-and-drifts.
A July 2026 3-agent adversarial lit-check found: the capacity≠learnability phenomenon is published
(2603.01959, 2603.14360, 2605.07755, 2207.02098); the remove-additive-path intervention is published in
*cross-architecture* forms (2412.19350 §4.2 diagonal-SSM B-removal; 2505.21749 §3.3 bilinear-vs-affine);
additive-path *necessity* is falsified by PD-SSM (2509.22284: exact S5 with additive path present);
DeltaProduct (2502.10297 v7) has the Cartan–Dieudonné sufficiency theorem, the empirical "S5 needs
n_h=4", and an S4/A5-at-n_h=2 result via SO(3) isomorphism that falsifies any naive
defining-representation generator law. What is unclaimed: the **within-architecture causal ablation**
(same model, one term deleted, capacity constant, shortcut→exact flip on a non-solvable group at
minimal width), the **attractor mechanism evidence**, and the **generator-level minimality law** stated
representation-relative. This note claims exactly those, scoped to this parameterization.

**Also disclosed:** a 1-seed instrument smoke was run today before this commit (parity n_h=1 ±b;
reproduced +b 1.00→0.50, −b 1.00→1.00). No other new runs exist at commit time. The full June grid is
re-run from scratch under this harness (R0) — the note will cite only harness-regenerated numbers.

## Protocol (fixed; identical to June unless stated)

Model `DeltaProduct(vocab, d=112, ncls, n_h, use_b)` from `agent/rnn_statetrack.py`; train L=32,
40 epochs, n=4000, batch 512, Adam lr 3e-3; eval per-position accuracy, n=1000, L=512; probe positions
{32, 64, 128, 256, 512}. Train data seed 1000+s, eval seed 9000+s. **5 seeds (0–4)** per primary cell
(upgrade from June's 3; June used means, this note uses **median** across seeds and reports min–max).
Decision rules (per cell, on medians): **GEN** = pos512 > 0.9; **shortcut** = pos32 > 0.9 and
pos512 < 0.6; **no-fit** = pos32 < 0.9; anything else = reported verbatim as mixed. Reference arms
(GRU M0; neg-diag M2) use 3 seeds. No hyperparameter tuning anywhere; `gen()` is extended with s4/a5
branches (no change to existing tasks — the parity validity gate re-runs and must pass). One further
disclosed improvement over June: the runner sets `torch.manual_seed(seed)` before model construction
(June controlled only the data RNG, leaving init randomness ambient).

New tasks (point-tracking, per-step targets, same format as s5): **s4** = generators (0 1), (0 1 2 3)
on 4 points (chance .25); **a5** = generators (0 1 2), (0 1 2 3 4) on 5 points (chance .20). The runner
asserts generated group orders (24, 60, 120) by closure and asserts each generator's permutation-rep
reflection length rank(I−P): s4 → (1, 3); a5 → (2, 4); s5 → (1, 4); parity → 1; mod3 → 2.

## Grid

- **R0 (reproduction under harness):** parity n_h∈{1,2,4}×±b; mod3 n_h∈{2,4}×±b; s5 n_h∈{1,2,4}×±b. 5 seeds.
- **E1 (necessity cell):** s5 n_h=3 ±b, 5 seeds. (Never run in June; the law "min n_h=4" needs it.)
- **E2 (representation law):** s4 n_h∈{1,2,3,4}×±b; a5 n_h∈{1,2,3,4}×±b. 5 seeds.
- **E3 (init-at-exact discriminator):** hand-constructed exact solutions (parity n_h=1; s5 n_h=4 via
  transposition-Householders; construction asserted: composed matrices equal the generator permutation
  matrices; pre-train eval must be ≥0.999 at L=512 — instrument gate, abort on failure). Arms: {+b (W_b
  init 0), −b} × {parity, s5} × 5 seeds, trained with the standard budget on standard L=32 data; record
  pre/post OOD curves and the per-epoch additive-path usage ‖W_b e‖/‖h‖ on a fixed probe batch.
- **E4 (references):** M0 (GRU) on s4/a5/s5; M2 (neg-diag) on s5. 3 seeds.
- **P3 (path-usage signature, computed in every +b run):** after training, re-eval with W_b zeroed
  (`curve_bzero`) and record mean ‖W_b e_t‖/‖h_t‖ on train-length data.

Runner is idempotent (JSON artifact per run keyed by cell+seed; complete artifacts skipped). Artifacts
record git hash, torch version, timing. Estimated total ≈ 200 runs, 1.5–4 GPU-h (RTX 3080).

## Predictions and kill criteria

- **P-R0:** June results reproduce at 5 seeds (parity & s5(n_h=4) ±b flips; mod3 −b fits-then-drifts).
  **K0 (note-killing):** if the parity or s5 flip fails (−b median pos512 ≤ 0.9 or +b median pos512 > 0.9),
  the core result is dead; we report the failed reproduction and stop.
- **P-E1:** s5 n_h=3 −b does **not** GEN (predicted no-fit or heavily degraded), because every faithful
  orthogonal representation of S5 gives the 5-cycle reflection length 4 (checked across all faithful
  irreps: standard 4-dim, std⊗sgn, 5a, 5b, 6-dim all give rank(I−M)=4), so 3 reflections/step cannot
  realize the generator. **K1:** if s5 n_h=3 −b GENs, the necessity direction of the reflection-length
  law is FALSE — the law demotes to sufficiency-only and the note's claim-3 section is rewritten to say so.
- **P-E2 (two-sided, decision rule pre-committed):** define min-n_h(task) = smallest n_h whose −b median
  pos512 > 0.9. **H-A** (DeltaProduct-analogue): s4 and/or a5 GEN at n_h=2 via a compact (SO(3)-like)
  representation despite the linear point-readout → the naive defining-rep generator law is falsified in
  our setting too; the law is stated representation-relative ("reflection length in the realized faithful
  representation") and we report replicating 2502.10297's observation under ablation. **H-B:** min-n_h
  = defining-rep generator reflection length (s4→3, a5→4) → the generator law HOLDS under point-tracking
  + linear readout, and the DeltaProduct S4/A5 counterexample is attributable to task format
  (group-element classification from a matrix state vs point tracking from a vector state). Either
  branch is a finding; the note states whichever obtains, with the other documented as the tested
  alternative. We do not pre-commit to H-A or H-B. (Genuine lean, recorded for honesty: H-B, because the
  4/5-way linear readout of an SO(3)-orbit state is geometrically obstructed — sign ambiguity of axis
  vectors — but the d=112 state leaves room for sign-disambiguated embeddings, so this is genuinely open.)
- **P-E3:** +b-from-exact **retains** GEN after the full training budget (the attractor captures
  random-init basins; the exact solution is at least locally stable), and ‖W_b e‖/‖h‖ stays ≈ 0.
  **Branch (not kill):** if it degrades below GEN and/or W_b usage grows, we report the *stronger*
  attractor reading (SGD pulled off the exact solution toward the additive route) and drop the
  "locally stable" phrasing. Either branch discriminates optimization-attractor accounts from
  pure inference-time error accounts (2605.07755) and from the unexplored-states account (2507.02782),
  and we say explicitly which readings survive.
- **P-P3:** in trained +b shortcut models, zeroing W_b at inference collapses even in-domain accuracy
  (median pos32 < 0.6) — the additive path is load-bearing in the learned shortcut. **K2 (mechanism
  kill):** if pos32 survives W_b-zeroing (> 0.9), the shortcut does not live on the additive path and
  the "attractor = additive route" mechanism claim is withdrawn (the flip result stands but the
  mechanism section is rewritten as open).
- **Contingency (pre-registered):** if a −b cell with n_h ≥ the (representation-relative) required
  length shows no-fit, run ONE disclosed extension arm at 3× epochs to disambiguate optimization-budget
  failure from representability failure. No other budget changes.

## Reporting commitments

The free-norm-style negative discipline applies: R0 failures, K-fires, and H-branch outcomes are
reported regardless of direction. Counts, medians, and min–max per cell; no p-values on 5-seed cells.
The note cites 2412.19350, 2505.21749, 2509.22284, 2601.05240, 2606.07254, 2602.14814, 2603.01959,
2603.14360, 2605.07755, 2507.02782, 2502.10297, 2411.12537 up front as scoping; PD-SSM (2509.22284) is
presented as the standing counterexample to any necessity reading of the additive-path claim, which is
therefore made **parameterization-scoped** (Householder linear RNNs at minimal width), never universal.
