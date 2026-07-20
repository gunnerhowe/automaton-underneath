# Paper 3 — submission metadata

## arXiv

**Title:** Implant, Conceal, Resist: The Fate of Inserted Skills Under Continued Training in Linear RNNs

**Categories:** cs.LG (primary); cross-list cs.AI

**Comments field:** 7 pages, 3 figures. Pre-registration, all 48 cell artifacts, reveal/control
artifacts, and a byte-verified number-regeneration pipeline at
https://github.com/gunnerhowe/automaton-underneath (implant/)

**License:** arXiv non-exclusive license (CC BY 4.0 on Zenodo record)

**Abstract (plain text, fits the 1920-char limit):**

Can a skill be inserted into a model and survive continued training -- and what protects it? The factual-editing literature measures how key-value edits decay under fine-tuning, but without a ground-truth circuit it cannot distinguish a skill that is erased from one that is bypassed. We supply the instrument: a verified-exact S5 word-problem automaton constructed in a Householder linear RNN, implanted into a vocab-extended model and trained onward on mixtures of its own task and a foreign task across four protection arms and three seeds (48 pre-registered cells). Protection is an interaction: removing the additive input path and giving the skill any exercise trickle (10% suffices) retains the implant perfectly (1.000 at every nonzero mixture), while the field-default architecture loses it even at 100% exercise. The resurrection: "dead" exercised implants restore to exactly 1.000 when the additive projection is zeroed at inference (6/6 cells, including fully-trainable ones) -- training grew a parasitic bypass around an intact circuit, and the bypass norm scales with exercise. The resulting fate taxonomy -- protected / concealed-but-recoverable / eroded (true erosion only when unprotected and unexercised) -- implies edit-durability measurements may conflate two mechanistically different outcomes, with a safety consequence: concealed capabilities can be restored by a one-line intervention. At LM scale (a 30M-parameter DeltaNet-class language model), the study inverted: the pre-registered skill-creation gate failed -- fine-tuning a pretrained model on tracking-rich data never produced a skill (chance through 15k steps, about 30x the budget) that the identical harness acquires from random initialization by step 500, at lower loss throughout: warm-start skill-acquisition resistance. At scale, insertion -- not retention -- is the bottleneck. Every number regenerates from committed artifacts.

## Zenodo

**Title:** same as arXiv. **Upload type:** publication/preprint + software (code bundle).
**License:** CC BY 4.0. **Description:** the abstract above.

**Keywords:**
model editing; knowledge editing; edit durability; skill insertion; circuit implantation;
task vectors; catastrophic forgetting; continual learning; plasticity; warm-starting;
state tracking; linear RNNs; Householder products; mechanistic interpretability;
concealed capabilities; AI safety; pre-registration

**Related identifiers:** https://github.com/gunnerhowe/automaton-underneath (isSupplementTo);
arXiv ID (isIdenticalTo — add via metadata edit once assigned)
