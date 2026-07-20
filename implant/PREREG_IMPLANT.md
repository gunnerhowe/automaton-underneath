# PRE-REGISTRATION — implant persistence (FROZEN before any run)

*The question nobody has asked with a ground-truth instrument: does a skill INSERTED into a model
survive CONTINUED TRAINING, and what protects it at what plasticity cost? Lit-check (2026-07-19):
factual-EDIT durability under fine-tuning is studied (2511.05852, 2507.14198, 2401.07453 — edits
are fragile; cited as the nearest neighbor); task-vector work is post-hoc merging (2212.04089 line);
hand-built-circuit transplant-then-train does not surface (closest: InterpBench 2407.14494 builds
ground-truth circuits into models but never implants-then-trains). Our differentiators: an
algorithmic SKILL with an exact-circuit ground truth (the verified S5 automaton construction from
the statetrack program), survival CURVES under continued training across data regimes, a
systematic protection-arm comparison, and the recurrent write-path substrate.*

## Tier 1 (flagship, toy scale): exact-automaton implant under foreign-task training

**Substrate:** DeltaProduct(d=112, n_h=4) from agent/rnn_statetrack.py, vocab EXTENDED 2->4,
head ncls=5. **Implant:** the verified-exact S5 construction (statetrack_note/runner.py
_reflection_program, BETA=6.0) written into embedding rows 0/1, Wu columns 0/1, h0, readout diag;
embedding rows 2/3 initialized N(0, 0.02) (the distractor's tokens; untouched by the construction).
**Distractor task:** parity on tokens {2,3} (bit = token-2), targets = classes {0,1} of the SAME
head. One task per sequence; token identity implies task; input length L=32.

**Continued-training mixtures** p = P(sequence is S5) in {1.0, 0.5, 0.1, 0.0}. p=1.0 is the
paper-1 attractor condition (in-codebase replication cell); p=0.0 is the pure-foreign-gradients
condition (the core question: do gradients from a task that never exercises the implant destroy
it through shared parameters?).

**Protection arms** (all initialized from the same exact construction):
- P0 none: use_b=True (Wb zero-init, trainable), all parameters trainable — the field default.
- P1 minus-b: use_b=False — the architectural lever paper 1 validated in the cornered regime.
- P2 freeze-core: use_b=True, Wu and h0 FROZEN (the transition circuit); emb/Wb/ro trainable
  (parity remains learnable via embeddings composing frozen reflections).
- P3 discount: use_b=True, lr x0.1 on {Wu, h0}, full lr elsewhere.

**Training:** Adam lr 3e-3 (the program's standard), bs 512, 4000 sequences/epoch (fresh mixture
each epoch, seed-streamed), 60 epochs. Seeds {0,1,2}. Grid = 4 mixtures x 4 arms x 3 seeds = 48
runs, one JSON artifact each.

**Measures:**
- Implant integrity: S5 position-511 accuracy (L=512, n=300) at epochs {0,10,20,30,40,50,60} —
  the exact-automaton signature.
- In-domain skill: S5 position-31 (L=32, n=500) every 5 epochs.
- Plasticity: parity position-31 (L=32, n=500) every 5 epochs.
- Mechanism: mean ||W_b e_t|| on S5 eval data at the integrity cadence (b-growth = the paper-1
  attractor channel).

**Predictions (frozen):**
- IP1 (protection at zero exercise): at p=0.0, end-of-training integrity satisfies
  median(P1) - median(P0) >= +0.30 — foreign gradients destroy the unprotected implant; -b
  protects.
- IP2 (exercise protects): under P0, end integrity is monotonically nondecreasing in p across
  {0.0, 0.1, 0.5, 1.0} in seed-median (ties allowed at the top).
- IP3 (freezing the core retains): P2 end integrity >= 0.90 at every mixture. Its plasticity cost
  (parity pos-31 vs P0 at p=0) is REPORTED with no frozen margin — the tradeoff is the deliverable.
- IP4 (attractor echo, bridge to paper 1): at p=1.0 under P0, end in-domain (pos-31) >= 0.95 while
  end integrity (pos-511) <= 0.70, and ||W_b e|| grows from ~0 — the same parasite, this codebase.

**Kills:**
- KI1 (instrument): at init, S5 pos-511 accuracy = 1.00 (within 0.005) for BOTH the use_b=True and
  use_b=False constructions, else fix the implant before any run (gate, not a result).
- KI2 (distractor learnable): at p=0.0 under P0, parity pos-31 >= 0.90 by epoch 60; else the
  distractor is redesigned and the change disclosed as an amendment.
- KI3 (nothing degrades): if every arm retains integrity >= 0.90 at every mixture, the finding is
  "toy implants are robust to foreign training" — reported as the result, line closes.

## Tier 2 (LM scale; runs only after Tier 1 verdicts are committed): acquired-skill retention

**Substrate:** the twogap 30M LM (twogap/model.py, arm A weights).
**Skill creation:** fine-tune A_m0_s0/final.pt on the m50 mixture for 2,500 steps (the emergence
budget the grid measured); gate: toggle-8 >= 0.90 (one disclosed escalation to 5,000 steps
allowed; else Tier 2 reports creation-failure and stops). Checkpoint = theta_skill;
tau = theta_skill - theta_A_m0.
**Retention:** continue training theta_skill on PURE m0 (TinyStories) for 5,000 steps, evals every
500 (toggle by depth, MQAR, val ppl). Arms:
- T0 none.
- T1 replay: 2% of batches (every 50th) drawn from the m50 stream.
- T2 freeze-tau: the top 1% of parameters by |tau| frozen (per-tensor masks via gradient hooks).
- T3 discount-tau: gradient x0.1 on that same top-1% mask.

**Predictions (frozen):** TP1: under T0, toggle-8 decays to <= 0.60 by step 5,000 (LM training
erases an unexercised tracking skill). TP2: T1 retains toggle-8 >= 0.90 at <= +0.5% val-ppl cost
vs T0. TP3 (ordering): end toggle-8 of T2 >= T3 >= T0.
**Exploratory (pre-labeled, no confirmatory claim):** graft tau onto B_m0_s0 (same init lineage,
different arm): toggle before and after a 500-step m0 anneal; reported descriptively.
**Seed policy:** Tier 2 runs seed 0; any positive verdict (TP2/TP3 margins) requires a seed-1
replication of the deciding cells before "confirmed" (the program's standing rule); negatives
stand on seed 0.

## Reporting

All 48 + Tier-2 cells reported regardless of direction; analyzer mechanical
(implant/analyze_implant.py); artifacts one JSON per run, git-stamped; amendments disclosed.
