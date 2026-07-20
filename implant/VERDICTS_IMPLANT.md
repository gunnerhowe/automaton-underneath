# IMPLANT PERSISTENCE — TIER-1 VERDICTS (vs PREREG_IMPLANT, frozen 39b2b88)

*48/48 cells (4 mixtures x 4 arms x 3 seeds); artifacts runs_toy/*.json; analyzer mechanical
(analyze_implant.py). KI1 exactness gate held in every cell (init integrity 1.000). Post-hoc
b-zero reveal (implant_reveal.py, reveal_results.json) labeled EXPLORATORY throughout.*

## Frozen predictions, scored

| P | prediction | verdict |
|---|---|---|
| IP1 | p=0: median P1 - P0 end integrity >= +0.30 | **NOT CONFIRMED** (+0.127: 0.323 vs 0.197). The margin was aimed at the wrong cell — at p=0 even the -b implant dies (see exploratory structure below). |
| IP2 | P0 integrity nondecreasing in mixture p | **NOT CONFIRMED** ([0.197, 0.257, 0.223, 0.247] — flat-dead everywhere; under +b, exercise does NOT protect). |
| IP3 | P2 (freeze-core) integrity >= 0.90 at every p | **NOT CONFIRMED** ([0.40, 0.40, 0.28, 0.21]); and its plasticity cost is catastrophic (parity 0.506 ~ chance vs 1.0 under P0) — the freeze-core arm is dominated. |
| IP4 | attractor echo at p=1 P0: in-domain >= 0.95 AND integrity <= 0.70, b-norm grows | **CONFIRMED** (in-domain 1.000, integrity 0.247, mean \|\|W_b e\|\| 0 -> 0.99). Paper-1's parasite, replicated on an implanted circuit in this codebase. |
| KI1/KI2/KI3 | — | KI1 held (all cells init 1.000); KI2 ok (P0@p=0 parity 1.0); KI3 does not fire. |

## The exploratory structure (stronger than the frozen predictions; labeled post-hoc)

**1. The -b x exercise interaction (the real protection law).** The -b implant retains integrity
**1.000 at every nonzero exercise rate — 10% suffices** (P1: 1.000/1.000/1.000 at p=0.1/0.5/1.0)
— while the +b implant dies even at 100% exercise (P0 ~0.25 everywhere). Protection is neither
lever alone: it is (no additive bypass) x (any exercise trickle). At p=0 even -b dies (0.323) via
shared-parameter drift (readout rows shared with the distractor's classes; Wu columns leak through
the distractor's embeddings).

**2. The b-zero resurrection (exploratory reveal; 12 rerun cells).** Zeroing W_b at inference on
the dead implants:
- p=1.0, P0 and P2, ALL 6 cells: integrity 0.20-0.26 -> **1.000 exactly**. The exercised implant
  was never destroyed — it was BYPASSED; the exact circuit survives 60 epochs of training intact
  underneath a grown parasite, recoverable by a one-line inference intervention.
- p=0.0, P2 (core frozen), 3/3: -> **1.000**. Frozen-core implants are fully recoverable even
  with zero exercise.
- p=0.0, P0 (unprotected), 3/3: NOT recoverable at depth (0.18-0.28) though in-domain restores
  (0.95-1.0): with everything trainable and no exercise, the deep automaton structure genuinely
  erodes while shallow behavior remains extractable.
- Parasite size tracks exercise: \|\|W_b e\|\| ~1.0-1.4 at p=1 vs 0.09-0.40 at p=0 — loss pressure
  on the implanted task FEEDS the bypass (paper-1's norm-inflation channel, here on implants).

**3. The fate taxonomy** (the deliverable): an implanted skill under continued training is either
**protected** (-b + any exercise: never harmed), **concealed** (+b + exercise, or frozen core:
behaviorally dead, fully recoverable), or **eroded** (unprotected + unexercised: deep structure
lost). "Implant death" is usually concealment, not erasure — the recoverability question, not the
retention question, is the right one.

## Relation to prior work

Factual-edit durability studies (2511.05852, 2507.14198) measure decay of key-value edits under
fine-tuning with no ground-truth circuit and no recoverability probe; our exact-construction
instrument shows the decay they measure may conflate bypass with erasure. The b-zero reveal is
paper 1's mechanism (howe2026automaton) operating on implants under foreign-task training.

## Tier-2 verdicts (LM scale; runs_lm/)

**KT1 FIRES (frozen verdict).** Skill creation failed: fine-tuning A_m0 (300M-token TinyStories
LM) on the m50 tracking mixture left toggle-8 at exactly 0.5 (the constant-policy signature) after
2,500 steps AND after the one pre-registered escalation to 5,000 — while mixture loss fell
normally throughout. TP1-TP3 unreached; retention arms and the graft did not run (nothing to
retain). Reported per the frozen rule.

**Exploratory controls (post-hoc, labeled).**
- COLD control: the IDENTICAL harness (same shard, optimizer, lr, batch geometry) from RANDOM
  init reaches toggle-8 = 1.0 by step 500 and holds it (explore_cold.json). The harness is
  validated; initialization is definitively the variable.
- WARM extension: the failed checkpoint continued +10,000 steps (15,000 total, ~30x the cold
  emergence time): toggle-8 = 0.5 at all 21 evaluations (explore_warmx.json). Acquisition is
  BLOCKED at this budget, not delayed.
- The loss twist: the warm model has strictly better LM quality and lower mixture loss than the
  cold model at every step, and never acquires the skill; the cold model pays perplexity and gets
  it in 500 steps. The skill is not on the loss-descent path from the pretrained basin — the
  shortcut (constant-answer the checks, model everything else) is a local optimum fine-tuning
  does not leave; a random init has no such basin to defend.

**Framing: warm-start skill-acquisition RESISTANCE** — the mirror image of the retention
question. Related to loss-of-plasticity in continual learning, but qualitative rather than
quantitative: zero acquisition of a capability at 30x budget, not slower learning. It is the
program's boundary variable again (how cornered training is): pretraining un-corners the loss.

**Program consequence.** At LM scale via fine-tuning, INSERTION — not retention — is the
bottleneck. This elevates direct weight-space implantation (Tier 1's instrument, where insertion
is exact by construction) from toy convenience to the load-bearing route for deliberate skill
installation, and Tier 1's fate taxonomy (protected / concealed / eroded) governs what happens
after.
