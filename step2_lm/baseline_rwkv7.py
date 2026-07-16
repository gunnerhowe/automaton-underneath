"""RWKV-7 (World) baseline on the boxes eval via the official `rwkv` package (pure-torch path,
RWKV_CUDA_ON=0 -- no kernel compile needed on Windows). Same scoring as baseline_lm.py: rank the gold
object among candidates by final-position logits. Run: python baseline_rwkv7.py [N] [DEPTHS...]"""
import os
import sys
import json
import time
import pathlib

os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"
from rwkv.model import RWKV                      # noqa: E402
from rwkv.utils import PIPELINE                  # noqa: E402
from eval_boxes import build_batch, OBJECTS      # noqa: E402
from huggingface_hub import hf_hub_download      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)
MODEL_FILE = "RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth"


def load():
    p = hf_hub_download("BlinkDL/rwkv-7-world", MODEL_FILE)
    model = RWKV(model=p.replace(".pth", ""), strategy="cuda fp16")
    pipe = PIPELINE(model, "rwkv_vocab_v20230424")
    return model, pipe


def audit(pipe):
    bad = []
    for o in OBJECTS:
        t = pipe.encode(" " + o)
        if len(t) != 1:
            bad.append((o, t))
    print("tokenizer audit:", "all single-token" if not bad else f"MULTITOKEN: {bad}", flush=True)
    return bad


def run(model, pipe, depths, n, seed=0):
    out = {}
    for depth in depths:
        exs = build_batch(depth, n, seed)
        correct = 0
        t0 = time.time()
        for ex in exs:
            logits, _ = model.forward(pipe.encode(ex["text"]), None)
            cand = [pipe.encode(" " + o)[0] for o in ex["candidates"]]
            pick = ex["candidates"][max(range(len(cand)), key=lambda i: logits[cand[i]].item())]
            correct += pick == ex["gold"]
        acc = correct / len(exs)
        out[depth] = acc
        print(f"  rwkv7-2.9b depth={depth:<3} acc={acc:.3f}  ({time.time()-t0:.0f}s, n={len(exs)})", flush=True)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    depths = [int(d) for d in sys.argv[2:]] or [0, 2, 4, 8, 16, 32]
    model, pipe = load()
    bad = audit(pipe)
    assert not bad, "swap OBJECTS for single-token words under the world vocab"
    res = run(model, pipe, depths, n)
    p = ROOT / "results" / "baseline_rwkv7-2.9b.json"
    p.write_text(json.dumps(dict(model="rwkv7-2.9b", n=n, acc=res)))
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
