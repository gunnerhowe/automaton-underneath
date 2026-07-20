"""Track B continued-training battery per PREREG_LMIMPLANT.md. Implant the exact toggle circuit into
the widened trained model, continue training on toggle/TinyStories mixtures under {none,
freeze-implant} arms, log behavioral+internal toggle integrity, ppl, and the answer-readout weights
for the resurrection probe. Run: python battery_wide.py [--smoke]"""
import sys
import json
import time
import math
import argparse
import pathlib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent
TG = ROOT.parent / "twogap"
sys.path.insert(0, str(TG))
from model import TwoGapLM  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from lm_implant_wide import (build_widened, construct_wide, toggle_acc, D_ANS,  # noqa: E402
                             ON, OFF, FLIP, CHECK, NEW)
from lm_implant import val_ppl  # noqa: E402

DEV = "cuda"
RUNS = ROOT / "runs_wide"
RUNS.mkdir(exist_ok=True)
BS, ACCUM, SEQ = 12, 2, 512


@torch.no_grad()
def internal_toggle(model, depths=(8, 32)):
    """Track-A-style integrity: read the implanted head's o (lamp) at the check position directly,
    bypassing the logit readout. Returns accuracy over depths."""
    from lm_implant_wide import toggle_seqs
    seqs, golds = toggle_seqs(depths, n_per=40)
    hits = tot = 0
    for ids, gold in zip(seqs, golds):
        t = torch.tensor([ids], device=DEV)
        x = model.emb(t)
        blk = model.blocks[0]
        h = blk.n1(x)
        B, T, d = x.shape
        q = blk.cq(blk.wq(h)).view(B, T, model.blocks[0].H, 64).transpose(1, 2)
        k = blk.ck(blk.wk(h)).view(B, T, model.blocks[0].H, 64).transpose(1, 2)
        v = blk.cv(blk.wv(h)).view(B, T, model.blocks[0].H, 64).transpose(1, 2)
        k = F.normalize(k, dim=-1)
        beta = 2 * torch.sigmoid(blk.wb(h)).transpose(1, 2)
        from core_deltanet import chunk_delta
        o, _ = chunk_delta(q, k, v, beta, torch.ones_like(beta), C=128)
        lamp = o[0, 8, -1, 0].item()                          # head 8, check position, axis 0
        pick = ON if lamp > 0 else OFF
        hits += (pick == gold); tot += 1
    return round(hits / tot, 3)


def toggle_train_batch(rng, bs, seqlen=64):
    """Pack toggle dedicated-token sequences; loss target = the CHECK-position answer token."""
    ids = np.zeros((bs, seqlen), dtype=np.int64)
    tgt = np.full((bs, seqlen), -100, dtype=np.int64)
    for b in range(bs):
        d = int(rng.integers(1, 9))
        st = int(rng.integers(0, 2))
        seq = [ON if st else OFF]
        s = st
        for _ in range(d):
            seq.append(FLIP); s ^= 1
        ans = ON if s else OFF
        seq.append(CHECK)
        seq = seq[:seqlen]
        ids[b, :len(seq)] = seq
        if len(seq) < seqlen:
            ids[b, len(seq)] = ans
            tgt[b, len(seq) - 1] = ans                        # predict answer after CHECK
    return torch.tensor(ids, device=DEV), torch.tensor(tgt, device=DEV)


def ts_stream(seed):
    toks = np.load(TG / "shards_m0_s0_tok.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.integers(0, toks.shape[0], BS)
        yield torch.from_numpy(toks[idx].astype(np.int64)).to(DEV)


def implant_param_mask(model):
    """Names of the head-8 implant parameters (for freeze-implant arm)."""
    return {"blocks.0.wq.weight", "blocks.0.wk.weight", "blocks.0.wv.weight",
            "blocks.0.wb.weight", "blocks.0.wb.bias", "blocks.0.wo.weight",
            "blocks.0.cq.conv.weight", "blocks.0.ck.conv.weight", "blocks.0.cv.conv.weight",
            "blocks.0.cq.conv.bias", "blocks.0.ck.conv.bias", "blocks.0.cv.conv.bias"}


def run_cell(arm, p, seed=0, steps=3000, smoke=False):
    name = f"{arm}_p{int(p*100)}_s{seed}"
    out = RUNS / f"{name}.json"
    if out.exists():
        print(f"{name}: exists, skip"); return
    tok = Tokenizer.from_file(str(TG / "tokenizer.json"))
    base = TwoGapLM(gate_mode="A").to(DEV)
    base.load_state_dict(torch.load(TG / "runs" / "A_m0_s0" / "final.pt"))
    model = build_widened(base.state_dict())
    construct_wide(model)
    # save the answer-readout weights for the resurrection probe (IL4)
    ans_ref = {"wo": model.blocks[0].wo.weight[D_ANS].clone(),
               "emb_on": model.emb.weight[ON].clone(), "emb_off": model.emb.weight[OFF].clone()}
    model.train()
    if steps and smoke:
        steps = 60
    # freeze-implant: head-8 rows can't move (approximate via zeroing their grads with hooks)
    frozen = implant_param_mask(model) if arm == "freeze" else set()
    hooks = []
    if frozen:
        s = 8 * 64
        for nm, prm in model.named_parameters():
            if nm in frozen:
                if prm.dim() == 2 and prm.shape[0] >= 576:
                    m = torch.zeros_like(prm); m[s:s + 64] = 1.0; m[D_ANS] = 1.0
                    hooks.append(prm.register_hook(lambda g, m=m: g * (1 - m)))
                else:
                    hooks.append(prm.register_hook(lambda g: g * 0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
    ts = ts_stream(1000 + seed)
    rng = np.random.default_rng(seed)
    log = []

    def evaluate(step):
        model.eval()
        beh = toggle_acc(model, [2, 8, 32])
        rec = dict(step=step, beh_toggle8=beh[8], beh_toggle32=beh[32],
                   internal_toggle=internal_toggle(model), ppl=round(val_ppl(model, tok, n=100), 3))
        model.train()
        log.append(rec)
        print(f"  [{name} {step}] beh8 {rec['beh_toggle8']} int {rec['internal_toggle']} "
              f"ppl {rec['ppl']}", flush=True)
    evaluate(0)
    t0 = time.time()
    ev = 30 if smoke else 300
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            if p > 0 and rng.random() < p:
                ids, tgt = toggle_train_batch(rng, BS)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids)
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                           tgt.reshape(-1), ignore_index=-100)
            else:
                ids = next(ts)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids)
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                           ids[:, 1:].reshape(-1))
            (loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % ev == 0 or step == steps:
            evaluate(step)
    # IL4 resurrection: re-pin the answer-readout weights, re-measure behavioral toggle
    with torch.no_grad():
        model.blocks[0].wo.weight[D_ANS] = ans_ref["wo"]
        model.emb.weight[ON] = ans_ref["emb_on"]; model.emb.weight[OFF] = ans_ref["emb_off"]
    model.eval()
    resurrection = toggle_acc(model, [2, 8, 32])
    out.write_text(json.dumps(dict(cell=name, arm=arm, p=p, seed=seed, curve=log,
                                   resurrection_beh8=resurrection[8],
                                   minutes=round((time.time() - t0) / 60, 1))))
    print(f"{name} DONE: beh8 {log[0]['beh_toggle8']}->{log[-1]['beh_toggle8']} "
          f"resurrect {resurrection[8]} ({(time.time()-t0)/60:.1f}m)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(0.90)
    if a.smoke:
        run_cell("none", 0.0, smoke=True)
        (RUNS / "none_p0_s0.json").unlink(missing_ok=True)
        print("smoke OK"); return
    for arm in ("none", "freeze"):
        for p in (0.0, 0.1, 0.5):
            run_cell(arm, p)
    print("BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
