"""Construction-feasibility probe for flagship #1: can an EXACT toggle automaton be constructed
inside ONE head of ONE real DeltaBlock (with its RMSNorm, SiLU short-conv, delta rule, residual,
tied-embedding readout) of the 30M twogap LM? Toggle is reflection-length 1 -> fits a single
delta-rule head (n_h=1). We construct into block 0, head 0, using 4 dedicated vocab tokens whose
embeddings occupy a reserved residual subspace, and read out on/off via the tied embedding.

Lamp lives at state entry S[0,0]:
  content on/off : k=e0, b=2, v=+/-c e0  -> sets S[0,0]
  flip           : k=e0, b=2, v=0 (g=1 writes zero) -> (I-2 e0 e0^T) negates S[0,0]
  check          : b=0, q=e0            -> o = S^T q reads row0 = lamp
Probe isolates the circuit (other blocks' contributions zeroed, final norm identity) to test
EXACTNESS of the construction; the real experiment implants into the trained model + continues
training. Run: python probe_construct.py"""
import sys
import pathlib
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "twogap"))
from model import TwoGapLM  # noqa: E402

DEV = "cuda"
D, H = 512, 8
DH = D // H
# dedicated token ids (overwrite top of the 8192 vocab)
ON, OFF, FLIP, CHECK = 8188, 8189, 8190, 8191
# reserved residual dims (block 0 reads/writes here; head 0 owns state dims 0..63)
D_CONTENT, D_SIGN, D_FLIP, D_CHECK, D_ANS = 500, 501, 502, 503, 504
KEYAX = 0            # head-0 key/value/query axis carrying the lamp (residual dim 0)
SAFE = 63            # throwaway key axis kept tiny so normalize() never divides by zero
E = 8.0              # dedicated-embedding magnitude
VC = 4.0             # content value magnitude


def construct(model):
    m = model
    with torch.no_grad():
        # --- dedicated embeddings: one-hot-ish in reserved dims (real rows untouched) ---
        for tok in (ON, OFF, FLIP, CHECK):
            m.emb.weight[tok].zero_()
        m.emb.weight[ON, D_CONTENT] = E;  m.emb.weight[ON, D_SIGN] = E
        m.emb.weight[OFF, D_CONTENT] = E; m.emb.weight[OFF, D_SIGN] = -E
        m.emb.weight[FLIP, D_FLIP] = E
        m.emb.weight[CHECK, D_CHECK] = E
        # answer readout: on vs off separated along D_ANS
        m.emb.weight[ON, D_ANS] = 6.0
        m.emb.weight[OFF, D_ANS] = -6.0

        blk = m.blocks[0]
        # RMSNorm n1 -> identity (unit weights already); scale is per-token but sign-preserving
        blk.n1.w.fill_(1.0)
        # projections: zero, then set head-0 rows (head h owns output dims h*DH..)
        for W in (blk.wq.weight, blk.wk.weight, blk.wv.weight):
            W.zero_()
        blk.wb.weight.zero_(); blk.wb.bias.fill_(-12.0)         # beta ~ 0 by default
        h0 = 0                                                   # head 0 output-dim offset
        # k = e0 on tokens that touch the lamp (content on/off + flip); tiny safe axis always on
        blk.wk.weight[h0 + KEYAX, D_CONTENT] = 20.0
        blk.wk.weight[h0 + KEYAX, D_FLIP] = 20.0
        blk.wk.bias if False else None
        # v = +/- c e0 for content (sign dim); flip writes v=0 (no D_SIGN, no D_CONTENT->v)
        blk.wv.weight[h0 + KEYAX, D_SIGN] = VC
        # q = e0 on check tokens (read), else 0
        blk.wq.weight[h0 + KEYAX, D_CHECK] = 20.0
        # beta ~ 2 on content + flip (write/erase), ~0 elsewhere (incl. check -> pure read)
        blk.wb.weight[0, D_CONTENT] = 30.0
        blk.wb.weight[0, D_FLIP] = 30.0
        # short-convs: make the dedicated head-0 axis an identity center tap (kill 4-token mixing)
        for cv in (blk.cq.conv, blk.ck.conv, blk.cv.conv):
            cv.weight.zero_(); cv.bias.zero_()
            cv.weight[h0 + KEYAX, 0, -1] = 1.0                  # center/current-token tap
            cv.weight[h0 + SAFE, 0, -1] = 0.0
            cv.bias[h0 + SAFE] = 1e-2                            # keep k-norm > 0
        # output routing: head-0 value-axis0 (residual dim 0) -> D_ANS
        blk.onorm.w.fill_(1.0)
        blk.wo.weight.zero_()
        blk.wo.weight[D_ANS, 0] = 1.0
        # kill this block's MLP (so it doesn't corrupt the reserved dims)
        for W in (blk.mlp_g.weight, blk.mlp_u.weight, blk.mlp_d.weight):
            W.zero_()
        # isolate: zero every other block entirely (probe reads the implant cleanly)
        for j in range(1, len(m.blocks)):
            b = m.blocks[j]
            for W in (b.wo.weight, b.mlp_d.weight):
                W.zero_()
        m.nf.w.fill_(1.0)
    return m


def toggle_seqs(depths, n_per=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    seqs, golds = [], []
    for d in depths:
        for _ in range(n_per):
            start = int(torch.randint(0, 2, (1,), generator=g))
            ids = [ON if start else OFF]
            state = start
            for _ in range(d):
                ids.append(FLIP); state ^= 1
            ids.append(CHECK)
            seqs.append(ids); golds.append(ON if state else OFF)
    return seqs, golds


@torch.no_grad()
def evaluate(model, depths):
    seqs, golds = toggle_seqs(depths)
    by = {d: [0, 0] for d in depths}
    for ids, gold in zip(seqs, golds):
        t = torch.tensor([ids], device=DEV)
        logits = model(t)[0, -1]
        pick = ON if logits[ON] > logits[OFF] else OFF
        d = len(ids) - 2
        by[d][0] += (pick == gold); by[d][1] += 1
    return {d: round(a / b, 3) for d, (a, b) in by.items()}


def main():
    torch.manual_seed(0)
    model = TwoGapLM(gate_mode="A").to(DEV).eval()
    construct(model)
    depths = [2, 4, 8, 16, 32, 64]
    acc = evaluate(model, depths)
    print("constructed toggle readout accuracy by depth:")
    for d in depths:
        print(f"   depth {d:>3}: {acc[d]:.3f}")
    ok = all(acc[d] == 1.0 for d in depths)
    print(f"\nFEASIBILITY: {'EXACT (GO)' if ok else 'not exact -- inspect/iterate'}")


if __name__ == "__main__":
    main()
