"""Regenerate every non-battery number cited in paper 4 into facts.json, so the manuscript's
construction/feasibility/obstacle figures all trace to code (not printed logs). Run once:
python capture_facts.py   (~10 min: loads the model ~12x)"""
import sys
import json
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parent
TG = ROOT.parent / "twogap"
sys.path.insert(0, str(TG))
from model import TwoGapLM  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from lm_implant import (load_trained, val_ppl, dead_dims, construct_into_trained,  # noqa: E402
                        DH, HEAD)
from lm_implant import toggle_acc as toggle_acc_asis  # noqa: E402
from lm_implant_wide import build_widened, construct_wide, toggle_acc as toggle_acc_wide  # noqa: E402

DEV = "cuda"


def main():
    tok = Tokenizer.from_file(str(TG / "tokenizer.json"))
    F = {}
    base = load_trained()
    ppl0 = val_ppl(base, tok)
    F["base_ppl"] = round(ppl0, 4)

    # --- dense residual stream: no free capacity (per-dim residual std on real text) ---
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    stds = torch.zeros(512, device=DEV); nb = 0
    with torch.no_grad():
        for i in range(20):
            ids = torch.tensor([tok.encode(ds[i]["text"]).ids[:256]], device=DEV)
            if ids.shape[1] < 8:
                continue
            x = base.emb(ids)
            for blk in base.blocks:
                x, _ = blk(x, None)
            stds += x[0].float().std(0); nb += 1
    stds /= nb
    F["resid_std_min"] = round(stds.min().item(), 3)
    F["resid_std_median"] = round(stds.median().item(), 3)
    F["resid_std_max"] = round(stds.max().item(), 3)

    # --- head-donation cost map (block 0, all heads) ---
    F["head_cost_pct"] = {}
    for hd in range(8):
        m = load_trained()
        with torch.no_grad():
            b = m.blocks[0]; s = hd * DH
            for W in (b.wq.weight, b.wk.weight, b.wv.weight):
                W[s:s + DH].zero_()
            b.wb.weight[hd].zero_(); b.wb.bias[hd] = -12.0; b.wo.weight[:, s:s + DH].zero_()
        F["head_cost_pct"][str(hd)] = round((val_ppl(m, tok) - ppl0) / ppl0 * 100, 1)
        del m; torch.cuda.empty_cache()
    F["cheapest_head"] = int(min(F["head_cost_pct"], key=lambda k: F["head_cost_pct"][k]))
    F["cheapest_head_cost_pct"] = F["head_cost_pct"][str(F["cheapest_head"])]
    F["head0_cost_pct"] = F["head_cost_pct"]["0"]
    F["head6_cost_pct"] = F["head_cost_pct"]["6"]

    # --- Track A: as-is readout wall (the circuit computes but in-place readout destroys the LM) ---
    d_ans = dead_dims(base, tok, 1)[0]
    m = load_trained(); construct_into_trained(m, d_ans)
    F["asis_readout_wall_ppl"] = round(val_ppl(m, tok), 1)
    F["asis_toggle_readout"] = toggle_acc_asis(m, [8])[8]      # reads 1.0 but at catastrophic ppl
    del m; torch.cuda.empty_cache()

    # --- Track B: widened substrate KL1 (clean behavioral implant, free) ---
    wide = build_widened(base.state_dict())
    F["widen_inert_ppl"] = round(val_ppl(wide, tok), 4)
    F["widen_inert_pct"] = round((F["widen_inert_ppl"] - ppl0) / ppl0 * 100, 3)
    construct_wide(wide)
    acc = toggle_acc_wide(wide, [2, 4, 8, 16, 32, 64])
    F["widen_toggle"] = acc
    F["widen_toggle_all_exact"] = all(v == 1.0 for v in acc.values())
    F["widen_implant_ppl"] = round(val_ppl(wide, tok), 4)
    F["widen_implant_pct"] = round((F["widen_implant_ppl"] - ppl0) / ppl0 * 100, 3)

    (ROOT / "facts.json").write_text(json.dumps(F, indent=1, sort_keys=True))
    print(json.dumps(F, indent=1))


if __name__ == "__main__":
    main()
