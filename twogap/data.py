"""Data layer for the two-gap experiment: synthetic tracking generators (with char-span roles ->
token-level oracle gate masks), 8k BPE tokenizer over the mixture, and packed training shards.

Synthetic families (per-step state statements make tracking directly loss-bearing):
  boxes  : 3 boxes, contents set once, swap ops, periodic "Box i holds the X." checks  (S3 tier)
  toggle : one lamp, flip ops, "The lamp is on/off." checks                            (Z2 tier)
  dial   : mod-3 dial, turn ops, "The dial points at k." checks                        (Z3 tier)
Roles: 'content' (initial facts), 'op' (transform sentences -> oracle gate suppresses writes),
'check' (answer statements -- the supervised tracking signal). Eval variants go deeper than train.
Run: python data.py --selftest | --tokenizer | --shards
"""
import json
import random
import pathlib
import argparse
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OBJ = ["apple", "key", "coin", "ball", "pen", "book", "cup", "ring", "sock", "fork",
       "star", "leaf", "bell", "drum", "kite", "rope", "cake", "fish", "hat", "map"]
TRAIN_DEPTH = (1, 8)                                   # ops between checks during training
EVAL_DEPTHS = (2, 4, 8, 16, 32)                        # held-out eval chains (16/32 = beyond train)


def _spans(parts):
    """parts: [(text, role)] -> (full_text, [(start, end, role)])."""
    out, spans, pos = [], [], 0
    for t, role in parts:
        out.append(t)
        spans.append((pos, pos + len(t), role))
        pos += len(t)
    return "".join(out), spans


def gen_boxes(rng, n_checks=4, depth_rng=TRAIN_DEPTH, fixed_depth=None):
    objs = rng.sample(OBJ, 3)
    cont = [0, 1, 2]
    parts = [(f"Box {i} holds the {objs[i]}. ", "content") for i in rng.sample(range(3), 3)]
    for _ in range(n_checks):
        d = fixed_depth if fixed_depth is not None else rng.randint(*depth_rng)
        for _ in range(d):
            i, j = rng.sample(range(3), 2)
            parts.append((f"Swap box {i} and box {j}. ", "op"))
            cont[i], cont[j] = cont[j], cont[i]
        q = rng.randrange(3)
        parts.append((f"Box {q} holds the {objs[cont[q]]}. ", "check"))
    return _spans(parts)


def gen_toggle(rng, n_checks=5, depth_rng=TRAIN_DEPTH, fixed_depth=None):
    state = rng.randrange(2)
    parts = [(f"The lamp is {'on' if state else 'off'}. ", "content")]
    for _ in range(n_checks):
        d = fixed_depth if fixed_depth is not None else rng.randint(*depth_rng)
        for _ in range(d):
            parts.append(("Flip the lamp. ", "op"))
            state ^= 1
        parts.append((f"The lamp is {'on' if state else 'off'}. ", "check"))
    return _spans(parts)


def gen_dial(rng, n_checks=5, depth_rng=TRAIN_DEPTH, fixed_depth=None):
    state = rng.randrange(3)
    parts = [(f"The dial points at {state}. ", "content")]
    for _ in range(n_checks):
        d = fixed_depth if fixed_depth is not None else rng.randint(*depth_rng)
        for _ in range(d):
            parts.append(("Turn the dial. ", "op"))
            state = (state + 1) % 3
        parts.append((f"The dial points at {state}. ", "check"))
    return _spans(parts)


GENS = {"boxes": gen_boxes, "toggle": gen_toggle, "dial": gen_dial}


def synth_doc(rng):
    return GENS[rng.choice(list(GENS))](rng)


def eval_pack(seed=7, n_per=200):
    """Held-out eval: fixed-depth chains, single final check; answer + candidates recorded.
    The final check sentence is split so 'the answer token' is the prediction target."""
    rng = random.Random(seed)
    pack = []
    for fam in GENS:
        for d in EVAL_DEPTHS:
            for _ in range(n_per):
                text, spans = GENS[fam](rng, n_checks=1, fixed_depth=d)
                s, e, role = spans[-1]
                assert role == "check"
                check = text[s:e]
                word = check.rstrip(". ").split(" ")[-1]
                prompt = text[:s] + check[:check.rindex(word)].rstrip() + " "
                cands = ({"boxes": None, "toggle": ["on", "off"], "dial": ["0", "1", "2"]}[fam]
                         or [w for w in check.split() if False])
                if fam == "boxes":
                    import re
                    cands = sorted(set(re.findall(r"holds the (\w+)", text[:s])))
                pack.append(dict(family=fam, depth=d, prompt=prompt, gold=word, candidates=cands,
                                 op_spans=[(a, b) for a, b, r in spans if r == "op"]))
    return pack


def mqar_pack(seed=9, n=300, n_pairs=(4, 8, 12)):
    """Recall probe: key-value pairs then a query (gap 2)."""
    rng = random.Random(seed)
    keys = ["red", "blue", "green", "gold", "gray", "pink", "tall", "flat", "old", "new", "big", "wet"]
    pack = []
    for np_ in n_pairs:
        for _ in range(n):
            ks = rng.sample(keys, np_)
            vs = rng.sample(OBJ, np_)
            body = " ".join(f"The {k} box holds the {v}." for k, v in zip(ks, vs))
            qi = rng.randrange(np_)
            pack.append(dict(n_pairs=np_, prompt=body + f" The {ks[qi]} box holds the ",
                             gold=vs[qi], candidates=sorted(vs)))
    return pack


def train_tokenizer(vocab=8192):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    rng = random.Random(0)

    def it():
        for i in range(0, 200000):
            yield ds[i]["text"]
        for _ in range(20000):
            yield synth_doc(rng)[0]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(it(), trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<eos>"]))
    tok.save(str(ROOT / "tokenizer.json"))
    aud = [o for o in OBJ if len(tok.encode(" " + o).ids) != 1]
    print(f"tokenizer trained (vocab {vocab}); multi-token objects: {aud or 'none'}")
    return tok


def build_shards(mixture, n_tokens=600_000_000 // 100, seed=0, seq=512, out=None):
    """Pack tokenized docs into (tokens uint16, oracle_gate uint8) shards. oracle_gate=0 on op-span
    tokens of synthetic docs (writes suppressed there under arm C), 1 elsewhere."""
    from tokenizers import Tokenizer
    from datasets import load_dataset
    tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
    eos = tok.token_to_id("<eos>")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    rng = random.Random(seed)
    order = iter(rng.sample(range(len(ds)), min(len(ds), 2_000_000)))
    toks, gates = [], []
    while len(toks) < n_tokens:
        if rng.random() < mixture:
            text, spans = synth_doc(rng)
            enc = tok.encode(text)
            ops = [(a, b) for a, b, r in spans if r == "op"]
            g = [0 if any(o[0] <= s < o[1] for o in ops) else 1 for s, _ in enc.offsets]
        else:
            enc = tok.encode(ds[next(order)]["text"])
            g = [1] * len(enc.ids)
        toks.extend(enc.ids + [eos])
        gates.extend(g + [1])
    n_seq = len(toks) // seq
    arr = np.array(toks[:n_seq * seq], dtype=np.uint16).reshape(n_seq, seq)
    gar = np.array(gates[:n_seq * seq], dtype=np.uint8).reshape(n_seq, seq)
    out = out or ROOT / f"shards_m{int(mixture*100)}_s{seed}"
    np.save(str(out) + "_tok.npy", arr)
    np.save(str(out) + "_gate.npy", gar)
    print(f"mixture {mixture}: {n_seq} seqs x {seq} ({arr.nbytes/1e6:.0f} MB), "
          f"gate-suppressed frac {1 - gar.mean():.4f}")


def selftest():
    rng = random.Random(0)
    import re
    for _ in range(500):                                  # boxes re-simulation from raw text
        text, spans = gen_boxes(rng)
        objs, cont = {}, {}
        for m in re.finditer(r"Box (\d) holds the (\w+)\.", text):
            i = int(m.group(1))
            if i not in objs:
                objs[i] = m.group(2)
        state = dict(objs)
        pos = 0
        for m in re.finditer(r"(Swap box (\d) and box (\d)\.)|(Box (\d) holds the (\w+)\.)", text):
            if m.group(1):
                i, j = int(m.group(2)), int(m.group(3))
                state[i], state[j] = state[j], state[i]
            elif m.start() > text.index(". ") and m.group(5):
                i, w = int(m.group(5)), m.group(6)
                if m.start() > spans[2][1]:               # past the content block -> must match state
                    assert state[i] == w, (text, state)
    for _ in range(500):                                  # toggle/dial re-simulation
        for fam, mod, op in (("toggle", 2, "Flip"), ("dial", 3, "Turn")):
            text, _ = GENS[fam](rng)
            m0 = re.search(r"(on|off|\d)", text)
            s = {"off": 0, "on": 1}.get(m0.group(1), None)
            s = int(m0.group(1)) if s is None and fam == "dial" else s
            for m in re.finditer(rf"({op}[^.]*\.)|(?:is (on|off)|points at (\d))\.", text[m0.end():]):
                if m.group(1):
                    s = (s + 1) % mod
                else:
                    got = {"on": 1, "off": 0}.get(m.group(2), m.group(3) and int(m.group(3)))
                    assert got == s, (fam, text)
    ep, mp = eval_pack(n_per=20), mqar_pack(n=20)
    assert all(e["gold"] in e["candidates"] for e in ep if e["family"] == "boxes")
    assert all(e["gold"] in e["candidates"] for e in mp)
    print(f"data selftest OK (1000 docs re-simulated; eval_pack {len(ep)}, mqar {len(mp)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tokenizer", action="store_true")
    ap.add_argument("--shards", type=float, default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.tokenizer:
        train_tokenizer()
    if a.shards is not None:
        build_shards(a.shards)
