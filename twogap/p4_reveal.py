"""P4 (PREREG_TWOGAP): inference-time op-token write-damping on trained arm-A checkpoints.
Prediction: improves deep tracking at m50, does nothing at m0. Loads A weights into gate_mode='C'
(identical parameter set) and passes oracle masks built from eval-prompt op spans via tokenizer
offsets. CPU-only on purpose (GPU is running the P3 replication). Run: python p4_reveal.py"""
import json
import torch
import pathlib
from tokenizers import Tokenizer
from model import TwoGapLM
from data import eval_pack

ROOT = pathlib.Path(__file__).resolve().parent
DEV = "cpu"
tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
PACK = eval_pack(n_per=50)


def masked_ids(ex):
    enc = tok.encode(ex["prompt"])
    ids = enc.ids
    mask = [0.0 if any(a < e and b > s for s, e in ex["op_spans"]) else 1.0
            for (a, b) in enc.offsets]
    return ids, mask


@torch.no_grad()
def score(model, damp):
    hits, by = 0, {}
    for i in range(0, len(PACK), 16):
        chunk = PACK[i:i + 16]
        pairs = [masked_ids(ex) for ex in chunk]
        L = max(len(p[0]) for p in pairs)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        om = torch.ones(len(chunk), L)
        for j, (e, m) in enumerate(pairs):
            ids[j, :len(e)] = torch.tensor(e)
            if damp:
                om[j, :len(m)] = torch.tensor(m)
        logits = model(ids, oracle_gate=om).float()
        for j, (ex, (e, _)) in enumerate(zip(chunk, pairs)):
            lg = logits[j, len(e) - 1]
            cand = [tok.encode(" " + c).ids[0] for c in ex["candidates"]]
            pick = ex["candidates"][int(torch.stack([lg[c] for c in cand]).argmax())]
            ok = pick == ex["gold"]
            hits += ok
            k = f"{ex['family']}_{ex['depth']}"
            a, b = by.get(k, (0, 0))
            by[k] = (a + ok, b + 1)
    return hits / len(PACK), {k: round(a / b, 3) for k, (a, b) in sorted(by.items())}


def deep(by):
    return round((by.get("boxes_16", 0) + by.get("boxes_32", 0) + by.get("toggle_8", 0)
                  + by.get("dial_8", 0)) / 4, 4)


def main():
    out = {}
    for run in ("A_m50_s0", "A_m0_s0"):
        m = TwoGapLM(gate_mode="C")
        m.load_state_dict(torch.load(ROOT / "runs" / run / "final.pt", map_location="cpu"))
        m.eval()
        base_acc, base_by = score(m, damp=False)
        damp_acc, damp_by = score(m, damp=True)
        out[run] = dict(base=base_by, damped=damp_by,
                        deep_base=deep(base_by), deep_damped=deep(damp_by))
        print(f"{run}: overall {base_acc:.3f} -> {damp_acc:.3f} | deep {deep(base_by)} -> {deep(damp_by)}")
        for k in sorted(base_by):
            print(f"   {k:<10} {base_by[k]:.2f} -> {damp_by[k]:.2f}")
    (ROOT / "p4_results.json").write_text(json.dumps(out, indent=1))
    d50 = out["A_m50_s0"]["deep_damped"] - out["A_m50_s0"]["deep_base"]
    d0 = out["A_m0_s0"]["deep_damped"] - out["A_m0_s0"]["deep_base"]
    print(f"\nP4 verdict: m50 deep delta {d50:+.3f} (predicted +), m0 delta {d0:+.3f} (predicted ~0)")
    print("CONFIRMED" if d50 > 0.03 and abs(d0) < 0.03 else "NOT CONFIRMED (report as measured)")


if __name__ == "__main__":
    main()
