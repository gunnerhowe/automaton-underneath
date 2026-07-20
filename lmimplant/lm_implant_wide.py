"""Track B: the WIDENED substrate. Add one inert 64-dim head (H:8->9, d:512->576) to the trained
arm-A model by zero-padding every weight, then implant the exact toggle circuit into that fresh
head + fresh residual dims. Because the new dims are EXACTLY zero for real tokens (fresh emb
columns, never written), the query is exactly zero on real text, so the block RMSNorm has nothing
to amplify -- the behavioral-readout wall (lm_implant.py finding) vanishes and the LM is unchanged.
Run: python lm_implant_wide.py   (verifies widening is identity, then implants, then KL1)"""
import sys
import math
import pathlib
import torch

TG = pathlib.Path(__file__).resolve().parent.parent / "twogap"
sys.path.insert(0, str(TG))
from model import TwoGapLM  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from lm_implant import ON, OFF, FLIP, CHECK, toggle_seqs, val_ppl  # noqa: E402

DEV = "cuda"
D0, H0 = 512, 8            # trained
DW, HW = 576, 9            # widened (one extra 64-dim head)
DH = 64
NEW = list(range(512, 576))                       # the 64 fresh residual dims (head 8)
D_ANS, D_CONTENT, D_SIGN, D_FLIP, D_CHECK = 512, 513, 514, 515, 516


def build_widened(state_dict):
    """Return a TwoGapLM(d=576,H=9) with trained weights in the top-left and zero-padded new head.
    The new head is inert (contributes 0), so language behavior is identical."""
    m = TwoGapLM(vocab=8192, d=DW, layers=8, H=HW, gate_mode="A").to(DEV)
    sd = m.state_dict()
    for k, wtrained in state_dict.items():
        w = sd[k]
        is_norm = k.endswith(".w") or k == "nf.w"             # RMSNorm scale: new dims -> 1 (identity)
        w.fill_(1.0) if is_norm else w.zero_()
        if wtrained.dim() == 1:
            w[:wtrained.shape[0]] = wtrained
        elif "conv.weight" in k:                              # (d,1,4)
            w[:wtrained.shape[0]] = wtrained
        else:                                                 # (out,in)
            w[:wtrained.shape[0], :wtrained.shape[1]] = wtrained
    m.load_state_dict(sd)
    with torch.no_grad():                                     # normalize-safety: new head k != 0-vector
        for blk in m.blocks:
            for cv in (blk.cq.conv, blk.ck.conv, blk.cv.conv):
                cv.bias[512 + 63] = 1e-2                       # tiny const on new head's axis 63
    return m.eval()


def construct_wide(m, vc=20.0, route=6.0, sout=3.0, sin=1.0):
    """Implant toggle into the fresh head 8 (dims 512..575); answer -> dim 512 (fresh, dead)."""
    s = 8 * DH                                                # head-8 output offset = 512
    kax = s                                                   # lamp axis = residual dim 512
    sax = s + 63
    with torch.no_grad():
        for t in (ON, OFF, FLIP, CHECK):
            m.emb.weight[t].zero_()
        m.emb.weight[ON, D_CONTENT] = sin;  m.emb.weight[ON, D_SIGN] = sin
        m.emb.weight[OFF, D_CONTENT] = sin; m.emb.weight[OFF, D_SIGN] = -sin
        m.emb.weight[FLIP, D_FLIP] = sin
        m.emb.weight[CHECK, D_CHECK] = sin
        m.emb.weight[ON, D_ANS] = sout;  m.emb.weight[OFF, D_ANS] = -sout

        blk = m.blocks[0]                                     # host the toggle in block 0's new head
        blk.wb.weight[8].zero_(); blk.wb.bias[8] = -12.0      # head-8 beta ~ 0 default
        blk.wk.weight[kax, D_CONTENT] = 60.0
        blk.wk.weight[kax, D_FLIP] = 60.0
        blk.wv.weight[kax, D_SIGN] = vc
        blk.wq.weight[kax, D_CHECK] = 60.0
        blk.wb.weight[8, D_CONTENT] = 90.0
        blk.wb.weight[8, D_FLIP] = 90.0
        for cv in (blk.cq.conv, blk.ck.conv, blk.cv.conv):
            cv.weight[s:s + DH].zero_(); cv.bias[s:s + DH].zero_()
            cv.weight[kax, 0, -1] = 1.0
            cv.bias[sax] = 1e-2
        # answer: route head-8 lamp (o dim 512) -> residual dim 512; protect it through blocks 1-7
        blk.wo.weight[D_ANS].zero_(); blk.mlp_d.weight[D_ANS].zero_()
        blk.wo.weight[D_ANS, kax] = route
        for j in range(1, len(m.blocks)):
            m.blocks[j].wo.weight[D_ANS].zero_(); m.blocks[j].mlp_d.weight[D_ANS].zero_()
    return m


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


def main():
    tok = Tokenizer.from_file(str(TG / "tokenizer.json"))
    base = TwoGapLM(gate_mode="A").to(DEV)
    base.load_state_dict(torch.load(TG / "runs" / "A_m0_s0" / "final.pt"))
    base.eval()
    ppl0 = val_ppl(base, tok)
    wide = build_widened(base.state_dict())
    ppl_inert = val_ppl(wide, tok)
    print(f"base ppl {ppl0:.4f} -> widened-inert ppl {ppl_inert:.4f} "
          f"({(ppl_inert-ppl0)/ppl0*100:+.3f}%)  [must be ~0: widening is identity]")
    construct_wide(wide)
    acc = toggle_acc(wide, [2, 4, 8, 16, 32, 64])
    ppl1 = val_ppl(wide, tok)
    tog_ok = all(acc[d] == 1.0 for d in acc)
    ppl_ok = (ppl1 - ppl0) / ppl0 <= 0.02
    print(f"implanted: toggle {acc}")
    print(f"implanted ppl {ppl1:.4f} ({(ppl1-ppl0)/ppl0*100:+.3f}%)")
    print(f"\nKL1 (widened): toggle {'PASS' if tog_ok else 'FAIL'} | "
          f"ppl {'PASS' if ppl_ok else 'FAIL'} -> {'GO' if tog_ok and ppl_ok else 'iterate'}")


if __name__ == "__main__":
    main()
