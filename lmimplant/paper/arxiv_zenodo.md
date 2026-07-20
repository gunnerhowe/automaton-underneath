# Paper 4 — submission metadata

## arXiv

**Title:** Implant, Conceal, Resurrect: Installing an Exact Circuit in a Real Language Model and Its Fate Under Continued Training

**Categories:** cs.LG (primary); cross-list cs.AI, cs.CL

**Comments:** 6 pages, 3 figures. Pre-registration, 6 cell artifacts, fact-capture + byte-verified
number/figure pipeline at https://github.com/gunnerhowe/automaton-underneath (lmimplant/)

**License:** arXiv non-exclusive (CC BY 4.0 on Zenodo)

**Abstract (plain text; fits the 1920-char limit):**

We construct a verified-exact algorithmic circuit -- a one-bit toggle automaton -- inside a single head of a trained linear-RNN language model, and study what continued training does to it. Two construction facts frame the study. The dense residual stream of a 30M-parameter tied-embedding model has no spare capacity, so an in-place behavioral readout collides with the model's own RMSNorm and destroys language quality (perplexity rises ~14 orders of magnitude). Adding one inert head (widening 512->576, verified identity to -0.13% perplexity) gives a clean substrate: the implanted circuit reads out behaviorally at 1.00 accuracy through toggle depth 64 at -0.10% perplexity -- an exact skill installed in a real LM, free. We then continue training under a pre-registered grid (arms {all-trainable, freeze-implant} x exercise {0,10,50%}). The central result: under all-trainable pure-LM training with no exercise, behavioral toggle decays to chance (1.00->0.52) while the internal circuit stays exact at the end (1.00), and re-pinning only the readout restores full behavior (1.00). The mechanism is interface drift: continued training moves the tied-embedding readout, suppressing behavioral expression while the computation persists and is weight-recoverable. A 10% exercise trickle prevents the drift; freezing the circuit's weights but not its interface is counterproductive (the internal circuit erodes, unrecoverable) because frozen weights cannot co-adapt to the drifting input; heavy exercise diffuses the skill beyond the implanted head. We position this precisely: a ground-truth, toy-scale demonstration of a phenomenon unlearning-robustness studies behaviorally -- suppression is not removal -- with honest caveats (constructed not learned, linear RNN not transformer, trivial skill, white-box recovery). All predictions were pre-registered; every number regenerates from committed artifacts.

## Zenodo

**Title:** same. **Upload type:** publication/preprint + software. **License:** CC BY 4.0.
**Description:** the abstract above.

**Keywords:**
circuit implantation; model editing; knowledge editing; unlearning; robust unlearning;
suppression vs removal; concealed capabilities; mechanistic interpretability; ground-truth circuits;
Tracr; linear RNNs; DeltaNet; state tracking; continued training; catastrophic forgetting;
tied embeddings; AI safety; pre-registration

**Related identifiers:** https://github.com/gunnerhowe/automaton-underneath (isSupplementTo);
arXiv ID (isIdenticalTo — add via metadata edit once assigned); companion manuscripts (isContinuedBy /
references the toy implant note).
