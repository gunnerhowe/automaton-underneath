"""Swap-only boxes entity-tracking eval with ORACLE token roles. Mirrors the hybrid bridge task in
natural language: write phase ("Box i contains the X.") then transform phase ("Swap the contents of
Box i and Box j.") then query ("Box q contains the"). Swaps are transpositions -> pure permutation of
contents (no overwrites, no empties) -- the S_n structure from the note, in English.

Scoring: rank the gold object among the N candidate objects by next-token logprob (single-token
objects; no generation). Roles: char spans for every swap sentence -> token positions via fast
tokenizer offset_mapping, so interventions can target exactly the transform tokens.
"""
import random

OBJECTS = ["apple", "key", "coin", "ball", "pen", "book", "cup", "ring", "sock", "fork"]
N_BOX = 5


def make_example(depth, rng):
    objs = rng.sample(OBJECTS, N_BOX)
    contents = list(range(N_BOX))                       # contents[box] = obj index into objs
    lines = [f"Box {i} contains the {objs[i]}." for i in range(N_BOX)]
    rng.shuffle(lines)
    text = " ".join(lines)
    swap_spans = []
    for _ in range(depth):
        i, j = rng.sample(range(N_BOX), 2)
        s = f" Swap the contents of Box {i} and Box {j}."
        swap_spans.append((len(text), len(text) + len(s)))
        text += s
        contents[i], contents[j] = contents[j], contents[i]
    q = rng.randrange(N_BOX)
    text += f" Box {q} contains the"
    return dict(text=text, gold=objs[contents[q]], candidates=objs, swap_spans=swap_spans)


# demos mirror the query shape EXACTLY (narrative -> swaps -> ONE query line, one-word answer) and
# use demo-only objects (disjoint from OBJECTS) so demo answers cannot bias candidate scoring
FEWSHOT = ("Box 0 contains the dice. Box 1 contains the lamp. Box 2 contains the rope. "
           "Box 2 contains the rope.\n\n"
           "Box 0 contains the kite. Box 1 contains the drum. "
           "Swap the contents of Box 0 and Box 1. Box 0 contains the drum.\n\n"
           "Box 0 contains the bell. Box 1 contains the soap. Box 2 contains the glove. "
           "Swap the contents of Box 1 and Box 2. Swap the contents of Box 0 and Box 1. "
           "Box 0 contains the glove.\n\n")


def build_batch(depth, n, seed, fewshot=True):
    rng = random.Random(seed * 1000 + depth)
    out = []
    for _ in range(n):
        ex = make_example(depth, rng)
        prefix = FEWSHOT if fewshot else ""
        off = len(prefix)
        ex["swap_spans"] = [(a + off, b + off) for a, b in ex["swap_spans"]]
        ex["text"] = prefix + ex["text"]
        out.append(ex)
    return out


def swap_token_mask(tokenizer, text, swap_spans):
    """Boolean mask over token positions lying inside any swap sentence (oracle transform tokens)."""
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors=None)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    mask = [any(a < e and b > s for s, e in swap_spans) for (a, b) in offs]
    return ids, mask


def _selftest():
    rng = random.Random(0)
    for depth in (0, 3, 8):
        for _ in range(200):
            ex = make_example(depth, rng)
            # brute-force resimulate from text to confirm gold
            import re
            objs = {}
            for m in re.finditer(r"Box (\d) contains the (\w+)\.", ex["text"]):
                if int(m.group(1)) not in objs:
                    objs[int(m.group(1))] = m.group(2)
            for m in re.finditer(r"Swap the contents of Box (\d) and Box (\d)\.", ex["text"]):
                i, j = int(m.group(1)), int(m.group(2))
                objs[i], objs[j] = objs[j], objs[i]
            q = int(re.search(r"Box (\d) contains the$", ex["text"]).group(1))
            assert objs[q] == ex["gold"], (ex, objs)
    print("eval_boxes selftest OK (600 examples re-simulated from text)")


if __name__ == "__main__":
    _selftest()
