# Paper 2 — submission metadata

## arXiv

**Title:** The Gate Learns Hygiene: The Training Economics of the Write Path in Linear-RNN Language Models

**Categories:** cs.LG (primary); cross-list cs.CL

**Comments field:** 10 pages, 4 figures. Pre-registrations, run artifacts, and a byte-verified
number-regeneration pipeline at https://github.com/gunnerhowe/automaton-underneath (twogap/, prodgate/)

**License:** arXiv non-exclusive license (CC BY 4.0 on Zenodo record)

**Abstract (plain text, fits the 1920-char limit):**

Linear-RNN language models write into a fixed-size state on every token, and their two best-known capability gaps -- associative recall and state tracking -- are both plausibly downstream of undisciplined writes. The frontier ships learned write gates (Gated DeltaNet-2), but what a learned write gate learns, and whether write discipline causally buys either capability during training, is unreported. We pre-registered and ran a 13-run controlled grid at 30M parameters: gate arms (always-write, L1-priced learned gate, unpriced learned gate, oracle gate) crossed with tracking-data mixture (0-50%). Three results. (1) The price discovers hygiene: the priced gate ends at mean 0.12 on state-operation tokens vs 0.47 on content, recovering the oracle's suppression pattern with no access to role labels, while the unpriced gate is 2.8x more open on operations; the gap is dose-responsive in mixture. These are, to our knowledge, the first token-role statistics of a learned write gate. (2) Hygiene buys recall, cheaply but weakly: +0.033/+0.022 over always-write across two seeds at matched perplexity, 2-3x an identical-configuration noise floor the grid itself supplies. (3) Hygiene does not buy automata: the gating-helps-tracking prediction died sign-inverted, the only skill that emerged arose in the always-write arm, and damping its op-token writes destroys it (1.00 to 0.50), completing a reveal/repair/destroy scope map for write-path intervention. Finally, a pre-registered zero-training measurement on a production hybrid (Nemotron-Flash-3B, re-implemented torch-native) closes the scale question: production write knobs already implement novelty-based hygiene (repeated facts written -0.095/-0.146 more weakly than position-matched novel facts), so the price is a small-scale scaffold for what production training discovers by default. Every number regenerates from committed artifacts.

## Zenodo

**Title:** same as arXiv. **Upload type:** publication/preprint + software (code bundle).
**License:** CC BY 4.0. **Description:** the abstract above.

**Keywords:**
linear attention; state-space models; DeltaNet; gated delta rule; write gate; sparsity penalty;
associative recall; MQAR; state tracking; language models; training dynamics; emergence;
mechanistic interpretability; pre-registration; Nemotron-Flash; production model analysis;
recurrent neural networks

**Related identifiers:** https://github.com/gunnerhowe/automaton-underneath (isSupplementTo);
arXiv ID (isIdenticalTo — add via metadata edit once assigned)
