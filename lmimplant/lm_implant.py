"""Coexistence implant for flagship #1: construct the exact toggle automaton into ONE head of the
TRAINED arm-A model's block 0, in the least-active residual dims, with a protected answer channel
through blocks 1-7. Leaves all other trained weights intact. Provides construct_into_trained() +
the KL1 coexistence gate (toggle-8 == 1.00 AND val ppl within +2%).
Run: python lm_implant.py   (loads A_m0_s0, constructs, reports KL1)"""
import sys
import math
import pathlib
import torch

TG = pathlib.Path(__file__).resolve().parent.parent / "twogap"
sys.path.insert(0, str(TG))
from model import TwoGapLM  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

DEV = "cuda"
D, H = 512, 8
DH = D // H
ON, OFF, FLIP, CHECK = 8188, 8189, 8190, 8191
KEYAX, SAFE = 0, 63                      # head-0 lamp axis / norm-safety axis
E, VC = 8.0, 4.0


def dead_dims(model, tok, n=1):
    """The n least-active residual dims by TRUE residual-activation std on real text (emb-column-std
    is misleading in a dense tied-embedding stream). These are the only genuinely-cheap dims to
    hijack for the protected answer channel."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    stds = torch.zeros(D, device=DEV)
    nb = 0
    with torch.no_grad():
        for i in range(20):
            ids = torch.tensor([tok.encode(ds[i]["text"]).ids[:256]], device=DEV)
            if ids.shape[1] < 8:
                continue
            x = model.emb(ids)
            for blk in model.blocks:
                x, _ = blk(x, None)
            stds += x[0].float().std(0); nb += 1
    return sorted(torch.topk(stds / nb, n, largest=False).indices.tolist())


# implant config: cheapest block-0 head to donate (measured), signaling dims (read at block-0 emb
# input; gated out for real tokens), dead answer dim (protected through blocks 1-7).
HEAD = 1
SIG = [3, 4, 5, 6]                        # D_CONTENT, D_SIGN, D_FLIP, D_CHECK (any dims; beta-gated)


def construct_into_trained(model, d_ans, head=HEAD, sin=1.2, vc=20.0, route=4.0, sout=2.0):
    D_CONTENT, D_SIGN, D_FLIP, D_CHECK = SIG
    s = head * DH                                              # head output-dim offset
    kax = s                                                    # within-head lamp axis 0 -> residual dim s
    sax = s + 63                                               # norm-safety axis
    with torch.no_grad():
        for t in (ON, OFF, FLIP, CHECK):
            model.emb.weight[t].zero_()
        model.emb.weight[ON, D_CONTENT] = sin;  model.emb.weight[ON, D_SIGN] = sin
        model.emb.weight[OFF, D_CONTENT] = sin; model.emb.weight[OFF, D_SIGN] = -sin
        model.emb.weight[FLIP, D_FLIP] = sin
        model.emb.weight[CHECK, D_CHECK] = sin
        model.emb.weight[ON, d_ans] = sout;  model.emb.weight[OFF, d_ans] = -sout

        blk = model.blocks[0]
        for W in (blk.wq.weight, blk.wk.weight, blk.wv.weight):
            W[s:s + DH].zero_()                                # donate the cheap head
        blk.wb.weight[head].zero_(); blk.wb.bias[head] = -12.0
        blk.wk.weight[kax, D_CONTENT] = 60.0
        blk.wk.weight[kax, D_FLIP] = 60.0
        blk.wv.weight[kax, D_SIGN] = vc
        blk.wq.weight[kax, D_CHECK] = 60.0
        blk.wb.weight[head, D_CONTENT] = 90.0
        blk.wb.weight[head, D_FLIP] = 90.0
        for cv in (blk.cq.conv, blk.ck.conv, blk.cv.conv):    # zero the WHOLE donated head's conv
            cv.weight[s:s + DH].zero_(); cv.bias[s:s + DH].zero_()
            cv.weight[kax, 0, -1] = 1.0                        # identity center tap on lamp axis
            cv.bias[sax] = 1e-2                                # norm-safety on axis 63
        blk.wo.weight[:, kax].zero_()                          # clear the donated head's v-axis0 col
        blk.wo.weight[d_ans].zero_(); blk.mlp_d.weight[d_ans].zero_()
        blk.wo.weight[d_ans, kax] = route                      # route lamp -> dead answer dim
        for j in range(1, len(model.blocks)):                  # protect the answer channel (1 dead dim)
            model.blocks[j].wo.weight[d_ans].zero_()
            model.blocks[j].mlp_d.weight[d_ans].zero_()
    return model


def toggle_seqs(depths, n_per=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    seqs, golds = [], []
    for d in depths:
        for _ in range(n_per):
            start = int(torch.randint(0, 2, (1,), generator=g))
            ids = [ON if start else OFF]
            st = start
            for _ in range(d):
                ids.append(FLIP); st ^= 1
            ids.append(CHECK); seqs.append(ids); golds.append(ON if st else OFF)
    return seqs, golds


@torch.no_grad()
def toggle_acc(model, depths):
    seqs, golds = toggle_seqs(depths)
    by = {d: [0, 0] for d in depths}
    for ids, gold in zip(seqs, golds):
        lg = model(torch.tensor([ids], device=DEV))[0, -1]
        pick = ON if lg[ON] > lg[OFF] else OFF
        d = len(ids) - 2
        by[d][0] += (pick == gold); by[d][1] += 1
    return {d: round(a / b, 3) for d, (a, b) in by.items()}


@torch.no_grad()
def val_ppl(model, tok, n=200):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    tot, cnt = 0.0, 0
    import torch.nn.functional as F
    for i in range(n):
        ids = torch.tensor([tok.encode(ds[i]["text"]).ids[:512]], device=DEV)
        if ids.shape[1] < 8:
            continue
        logits = model(ids)
        loss = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
        tot += loss.item() * (ids.shape[1] - 1); cnt += ids.shape[1] - 1
    return math.exp(tot / cnt)


def load_trained():
    m = TwoGapLM(gate_mode="A").to(DEV)
    m.load_state_dict(torch.load(TG / "runs" / "A_m0_s0" / "final.pt"))
    return m.eval()


def main():
    tok = Tokenizer.from_file(str(TG / "tokenizer.json"))
    base = load_trained()
    ppl0 = val_ppl(base, tok)
    d_ans = dead_dims(base, tok, 1)[0]
    print(f"answer dim (true-dead): {d_ans}; donated block-0 head: {HEAD}")
    model = load_trained()
    construct_into_trained(model, d_ans)
    with torch.no_grad():
        ids = torch.tensor([tok.encode("Once upon a time there was a little cat who ran.").ids[:64]],
                           device=DEV)
        _, hh = model(ids, return_hidden=True)
    acc = toggle_acc(model, [2, 4, 8, 16, 32, 64])
    ppl1 = val_ppl(model, tok)
    print(f"toggle readout: {acc}")
    print(f"real-text max|h[d_ans]|: {hh[0, :, d_ans].abs().max().item():.2f} (want small)")
    print(f"val ppl: base {ppl0:.3f} -> implanted {ppl1:.3f} ({(ppl1-ppl0)/ppl0*100:+.2f}%)")
    tog_ok = all(acc[d] == 1.0 for d in acc)
    ppl_ok = (ppl1 - ppl0) / ppl0 <= 0.02
    print(f"\nKL1 coexistence gate: toggle {'PASS' if tog_ok else 'FAIL'} | "
          f"ppl {'PASS' if ppl_ok else 'FAIL'} -> "
          f"{'GO' if tog_ok and ppl_ok else 'iterate/fallback'}")


if __name__ == "__main__":
    main()
