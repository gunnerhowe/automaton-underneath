"""Baseline accuracy vs swap-depth for public linear-RNN LMs on the boxes eval (instrument gate for
Step 2: baselines must beat chance and degrade with depth, else the probe cannot discriminate).
Scoring: next-token logprob ranking over the candidate objects' first tokens.
Run: python baseline_lm.py MODEL [N_PER_DEPTH]   (MODEL: mamba-130m | mamba-1.4b | mamba2-1.3b)
"""
import sys
import json
import time
import pathlib
import torch
from transformers import AutoTokenizer, MambaForCausalLM, Mamba2ForCausalLM
from eval_boxes import build_batch

ROOT = pathlib.Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)
MODELS = {
    "mamba-130m": ("state-spaces/mamba-130m-hf", MambaForCausalLM),
    "mamba-1.4b": ("state-spaces/mamba-1.4b-hf", MambaForCausalLM),
    "mamba-2.8b": ("state-spaces/mamba-2.8b-hf", MambaForCausalLM),
    "mamba2-1.3b": ("state-spaces/mamba2-1.3b-hf", Mamba2ForCausalLM),
}
DEPTHS = (0, 2, 4, 8, 16, 32)


@torch.no_grad()
def run(model, tok, name, n_per_depth, seed=0):
    dev = next(model.parameters()).device
    out = {}
    for depth in DEPTHS:
        exs = build_batch(depth, n_per_depth, seed)
        correct = 0
        t0 = time.time()
        for ex in exs:
            ids = tok(ex["text"], return_tensors="pt").input_ids.to(dev)
            logits = model(ids).logits[0, -1]
            cand_ids = [tok(" " + o, add_special_tokens=False).input_ids[0] for o in ex["candidates"]]
            pick = ex["candidates"][int(torch.stack([logits[c] for c in cand_ids]).argmax())]
            correct += pick == ex["gold"]
        acc = correct / len(exs)
        out[depth] = acc
        print(f"  {name} depth={depth:<3} acc={acc:.3f}  ({time.time()-t0:.0f}s, n={len(exs)})", flush=True)
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "mamba-130m"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    repo, cls = MODELS[name]
    print(f"loading {repo} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    model = cls.from_pretrained(repo, torch_dtype=torch.float16).to("cuda").eval()
    res = run(model, tok, name, n)
    p = ROOT / "results" / f"baseline_{name}.json"
    p.write_text(json.dumps(dict(model=name, n=n, acc=res)))
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
