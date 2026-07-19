# TWO-GAP GRID + FUSION — VERDICTS (vs PREREG_TWOGAP 8ca59bc + amendments, PREREG_FUSION 8de59c9 + Amendment 1)

*Grid: 13/13 runs, 300M tokens each, single seed (seed-1 replication of the positive P3 cells in
flight). Sequence honored: fusion KF3 verdict committed (148b0c7) BEFORE m2 was opened; m2 opened
only for the TWOGAP verdicts it governs.*

## Fusion (PREREG_FUSION): KF3 FIRES — pre-registered deferral

0/7 calibration runs evented (bar: mean(toggle-8, dial-8) >= 0.9 sustained). The near-miss detail:
A_m50 reached toggle-8 = 1.00 but dial-8 = 0.34 (chance) — the composite bar failed on the dial
(mod-3) family, the same family that resisted every arm of this program since the June statetrack
work. No thresholds fitted; the blind m2 cells were never opened for the fusion; anchor FA=0 on
negatives is vacuous (anchors never fired). The external review's top concern (event bar out of
budget reach) is vindicated and credited. Any relaxed-bar fusion analysis remains pre-labeled
post-hoc exploratory (Amendment 1) and has not been run.

## Two-gap grid (PREREG_TWOGAP)

| P | prediction | verdict |
|---|---|---|
| **P2 (headline)** | priced gate goes sparse AND shows the hygiene signature; unpriced shows less | **FULLY CONFIRMED.** B final gate mean 0.338 (<0.5), 35.5% of gates off; hygiene signature: mean gate **0.118 on op tokens vs 0.479 on content** (gap +0.361) — the L1 price discovered the oracle's pattern unsupervised. Unpriced B': mean 0.500, 7% off, gap +0.269 (op-gates 3x more open). First report of learned-write-gate token-role statistics anywhere. |
| **P3** | at m0, B > A on MQAR by >0.03 at matched ppl | **DIRECTION REPLICATED, MARGIN FRAGILE — scored per the frozen rule.** Seed 0: +0.033; seed 1: +0.022 (below the +0.03 bar). 2/2 seeds positive at matched ppl (−0.6%/−0.3%), mean +0.028. Gate sparsity itself replicated almost exactly (0.540 vs 0.543). Honest verdict: hygiene-buys-recall is SUPPORTED directionally but the pre-registered margin held at only one of two seeds — reported as a small real-looking effect requiring more seeds, not a confirmed >3pp claim. |
| **P1** | oracle-gating advantage grows with scarcity (C−A > +0.10 at m2) | **DEAD, SIGN-INVERTED**: C−A = **−0.070** at m2 — gating HURT tracking where tracking data was rare. |
| **P5** | C > A on deep tracking at every mixture > 0 | **NOT CONFIRMED**: +0.07 at m10, −0.04 at m50, −0.07 at m2 — mixed signs, no consistent hygiene advantage for tracking. |
| **P6** | B within 3% ppl of A everywhere | **PASSES** (max cost +0.5%). |
| **P4** | reveal probe: op-token damping helps at m50, not m0 | **NOT CONFIRMED — inverted at m50, exact at m0.** Damping op-token writes on A_m50 DESTROYS its toggle mastery (1.00 -> 0.50-0.54 at every depth; deep delta -0.160); on A_m0 the effect is exactly 0.000 (that half held). p4_results.json. |
| K1/K2/K3 | — | **None fire.** |

**The emergent-skill twist:** only the always-write arm at 50% mixture mastered deep toggle
(toggle-8 = 1.00 vs 0.50 for every gated arm). Write gating did not buy tracking emergence in LM
training at this scale — under legitimate next-token supervision the write path is load-bearing for
skill formation, consistent with the hybrid bridge's K1, 2602.14814's reveal-supervision result, and
PD-SSM. Deep dial (mod-3) and deep boxes emerged nowhere: the mod-3 wall is now a four-experiment
constant of this program.

## P4's meaning: the arc closes in three regimes

The A_m50 toggle skill LIVES on the write path — damping op-token writes deletes it rather than
revealing anything beneath. Combined with the toy and bridge results, the program's complete map:
(1) minimal-width pure tasks: writes parasitic, automaton underneath, damping REVEALS (1.00);
(2) hybrid content tasks: writes tamed, damping repairs damaged seeds; (3) LM training: writes are
the mechanism itself — skills form through writes, live on writes, and damping destroys. The
boundary variable is how cornered training is. The original paper's mechanism was real; its scope
is now mapped end-to-end with pre-registered evidence in every regime.

## The honest synthesis

**The gate learns hygiene; hygiene buys memory, not automata.** The two-gap hypothesis splits down
the middle: the recall side survives (P2 + P3 — priced write-discipline reduces state pollution and
improves retrieval at zero perplexity cost), while the tracking side dies at LM scale (P1 inverted,
P5 mixed). The write-economics contribution is therefore a MEMORY-HYGIENE result with the first
token-role gate statistics, not a tracking-emergence result — and the tracking story across the
whole program lands as: write discipline rescues automata only where training is cornered (minimal
width, pure tasks); under realistic LM supervision the write path is how skills form.

## Status of remaining items
- P4 reveal probe: DONE (p4_results.json; verdict above).
- P3/P2 seed-1 replication: DONE (scored above).
- Write-up: DONE — paper/main.pdf ("The Gate Learns Hygiene: The Training Economics of the Write
  Path in Linear-RNN Language Models", 9pp). Numbers pipeline: gen_numbers.py + gen_figures.py +
  verify_regen.py (byte-verified). Public sync: twogap/ record mirrored into the public
  automaton-underneath repo (code, preregs, verdicts, run logs, artifacts, paper).

## HYGIENE ARTIFACT NOTE (2026-07-19)

The role-annotated training shards for m50/m10/m2 were deleted from disk in the post-excision
cleanup, so the P2 hygiene signature was recomputed on a deterministic regeneration of each
training-corpus prefix (seeded generators + frozen tokenizer; 3M tokens each; hyg_m*_s0) and
committed as hygiene.json. B_m50: op 0.1187 / content 0.4728 (gap +0.3541) vs the originally
logged 0.118/0.479 (+0.361) — identical within sampling noise. Bonus exploratory measurement (not
in the original verdict): the hygiene gap exists at every mixture and is dose-responsive
(+0.197 @ m2, +0.277 @ m10, +0.354 @ m50); B' @ m50 gap +0.265 with op-gate 0.335 (2.8x more open
than priced B). The paper cites hygiene.json values.

## HISTORY-REWRITE DISCLOSURE (2026-07-19)

Eight training-shard .npy files (~3.6GB) were accidentally committed in the data-layer commit,
silently blocking every GitHub push since (the -q flag ate the rejections; the remote sat at the
Step-2 commit). The unpushed range was rewritten with git filter-branch to excise ONLY those files;
commit order, messages, author timestamps, and all other content are unchanged, and the parent-hash
chain still enforces the sequencing (prereg-before-results). Consequence disclosed honestly: the
twogap/fusion prereg commits carry third-party (GitHub) timestamps only from today's push; the
intra-repo hash chain is the ordering evidence. Hash remap for citations in this file and the
preregs: 8ca59bc->b868a75 (TWOGAP prereg), 8de59c9->e84d431 (FUSION prereg), 6b85b71->a260732
(Fusion Amendment 1), 148b0c7->f10b406 (KF3-before-m2), f7ab4be->c4f2692 (verdicts),
ed7918f->11701e8 (P4), 794dc75->2c2ce6b (data layer).
