# Paper 3 (merged, incl. former paper 4) — submission metadata

## arXiv

**Title:** Implant, Conceal, Resurrect: The Fate of Inserted Skills Under Continued Training, from Toy Automata to a Language Model

**Categories:** cs.LG (primary); cross-list cs.AI

**Comments field:** 9 pages, 6 figures. Pre-registrations, all cell artifacts, the LM-scale
construction + battery, and a byte-verified number-regeneration pipeline at
https://github.com/gunnerhowe/automaton-underneath (implant/, lmimplant/)

**License:** arXiv non-exclusive license (CC BY 4.0 on Zenodo record)

**Abstract (plain text, fits the 1920-char limit):**

Can a skill be inserted into a model and survive continued training, and what protects it? Without a ground-truth circuit, one cannot tell a skill that is erased from one that is bypassed. We supply the instrument: a verified-exact S5 automaton constructed in a Householder linear RNN, implanted and trained onward across protection arms and seeds (48 pre-registered cells). Protection is an interaction -- removing the additive input path plus any exercise trickle (10%) retains it perfectly, while the default architecture loses it even at 100% exercise. The resurrection: "dead" exercised implants restore to exactly 1.000 when the additive projection is zeroed at inference (6/6 cells) -- training grew a bypass around an intact circuit. The fate taxonomy (protected / concealed-but-recoverable / eroded) implies edit-durability decay measurements may conflate two outcomes, only one of which is loss. We then take the study to a real 30M linear-RNN LM. First, an inverted control: fine-tuning a pretrained model on tracking data never produces the skill that random init learns in 500 steps -- insertion, not retention, is the LM-scale bottleneck (warm-start acquisition resistance). So we construct the skill in: an added inert head installs a toggle automaton reading out at 1.00 for -0.10% perplexity, and the toy fate map reproduces -- unexercised continued training suppresses behavioral expression to chance while the internal circuit stays exact and readout-recoverable, via interface drift (the tied-embedding readout moves, not the computation). This connects to unlearning-robustness (suppression need not be removal), with the difference that our ground truth is constructed, so we prove the computation survived rather than infer it -- under honest caveats (constructed, toy, linear RNN, white-box recovery). All predictions pre-registered; every number regenerates from artifacts.

## Zenodo

**Title:** same as arXiv. **Upload type:** publication/preprint + software (code bundle).
**License:** CC BY 4.0. **Description:** the abstract above.

**Keywords:**
model editing; knowledge editing; edit durability; machine unlearning; unlearning robustness;
skill insertion; circuit implantation; Tracr; task vectors; catastrophic forgetting;
continual learning; warm-starting; state tracking; linear RNNs; Householder products;
mechanistic interpretability; concealed capabilities; interface drift; AI safety; pre-registration

**Related identifiers:** https://github.com/gunnerhowe/automaton-underneath (isSupplementTo);
arXiv ID (isIdenticalTo — add via metadata edit once assigned). Supersedes the standalone LM-scale
draft (former "paper 4"), now folded in as Tier 3.
