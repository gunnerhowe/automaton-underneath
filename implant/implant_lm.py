"""Tier-2 (LM) acquired-skill retention per PREREG_IMPLANT.md. Phases:
  create  : A_m0 + 2500 FT steps on regenerated m50 prefix -> theta_skill (gate toggle-8 >= 0.9)
  retain  : theta_skill + 5000 steps on m0 under arms T0/T1/T2/T3, evals every 500
  graft   : exploratory tau-graft onto B_m0 (+ 500-step anneal), descriptive only
Runs ONLY after Tier-1 verdicts are committed (prereg sequencing).
Run: python implant_lm.py --phase create|retain|graft [--arm T0..T3]"""
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
from tokenizers import Tokenizer  # noqa: E402
from model import TwoGapLM  # noqa: E402
from train import score_pack, val_ppl  # noqa: E402
from data import eval_pack, mqar_pack  # noqa: E402

RUNS = ROOT / "runs_lm"
RUNS.mkdir(exist_ok=True)
DEV = "cuda"
BS, ACCUM, SEQ = 12, 2, 512


def batches(tokp, seed, steps):
    toks = np.load(tokp, mmap_mode="r")
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, toks.shape[0], BS)
        yield torch.from_numpy(toks[idx].astype(np.int64)).to(DEV)


def evals(model, tok, ep, mp):
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tr_acc, tr_by = score_pack(model, tok, ep)
        mq_acc, _ = score_pack(model, tok, mp)
        ppl = val_ppl(model, tok)
    model.train()
    return dict(ppl=round(ppl, 3), track=round(tr_acc, 4), mqar=round(mq_acc, 4),
                toggle_8=tr_by.get("toggle_8"), toggle_2=tr_by.get("toggle_2"),
                toggle_4=tr_by.get("toggle_4"), boxes_16=tr_by.get("boxes_16"),
                dial_8=tr_by.get("dial_8"))


def train_steps(model, opt, stream_main, n_steps, tok, ep, mp, log, replay_stream=None,
                eval_every=500, lr=3e-4, warm=100):
    t0 = time.time()
    for step in range(1, n_steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1.0, step / warm)
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            if replay_stream is not None and step % 50 == 0:
                ids = next(replay_stream)
            else:
                ids = next(stream_main)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                       ids[:, 1:].reshape(-1))
            (loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"  hb step {step} loss {loss.item():.3f} "
                  f"{step*BS*ACCUM*SEQ/(time.time()-t0):,.0f} tok/s", flush=True)
        if step % eval_every == 0 or step == n_steps:
            rec = dict(step=step, **evals(model, tok, ep, mp))
            log.append(rec)
            print(f"  [eval {step}] ppl {rec['ppl']} toggle8 {rec['toggle_8']} "
                  f"mqar {rec['mqar']}", flush=True)


def load_a_m0():
    m = TwoGapLM(gate_mode="A").to(DEV)
    m.load_state_dict(torch.load(TG / "runs" / "A_m0_s0" / "final.pt"))
    return m


def phase_create():
    out = RUNS / "skill"
    out.mkdir(exist_ok=True)
    tok = Tokenizer.from_file(str(TG / "tokenizer.json"))
    ep, mp = eval_pack(n_per=50), mqar_pack(n=60)
    model = load_a_m0()
    log = [dict(step=0, **evals(model, tok, ep, mp))]
    print(f"pre-FT: toggle8 {log[0]['toggle_8']} ppl {log[0]['ppl']}", flush=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
    stream = batches(ROOT / "ft_m50_s0_tok.npy", 42, 10**9)
    train_steps(model, opt, stream, 2500, tok, ep, mp, log)
    t8 = log[-1]["toggle_8"]
    if t8 < 0.9:
        print(f"gate miss at 2500 (toggle8 {t8}); escalating once to 5000 (disclosed)", flush=True)
        train_steps(model, opt, stream, 2500, tok, ep, mp, log)
        t8 = log[-1]["toggle_8"]
    torch.save(model.state_dict(), out / "theta_skill.pt")
    (out / "create_log.json").write_text(json.dumps(dict(log=log, final_toggle8=t8)))
    print(f"CREATE {'OK' if t8 >= 0.9 else 'FAILED (report per KT1)'}: toggle8 {t8}", flush=True)


def tau_mask(top_frac=0.01):
    base = torch.load(TG / "runs" / "A_m0_s0" / "final.pt")
    skill = torch.load(RUNS / "skill" / "theta_skill.pt")
    alltau = torch.cat([(skill[k].float() - base[k].float()).abs().flatten() for k in base])
    thr = torch.quantile(alltau[torch.randint(0, alltau.numel(), (2_000_000,))], 1 - top_frac)
    return {k: ((skill[k].float() - base[k].float()).abs() >= thr) for k in base}, float(thr)


def phase_retain(arm):
    out = RUNS / arm
    out.mkdir(exist_ok=True)
    if (out / "log.json").exists():
        print(f"{arm}: exists, skip")
        return
    tok = Tokenizer.from_file(str(TG / "tokenizer.json"))
    ep, mp = eval_pack(n_per=50), mqar_pack(n=60)
    model = TwoGapLM(gate_mode="A").to(DEV)
    model.load_state_dict(torch.load(RUNS / "skill" / "theta_skill.pt"))
    hooks = []
    if arm in ("T2", "T3"):
        masks, thr = tau_mask()
        scale = 0.0 if arm == "T2" else 0.1
        for name, p in model.named_parameters():
            m = masks[name].to(DEV)
            keep = torch.where(m, torch.full_like(p, scale), torch.ones_like(p))
            hooks.append(p.register_hook(lambda g, keep=keep: g * keep))
        print(f"{arm}: tau mask thr {thr:.5f}, top-1% grads x{scale}", flush=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
    main = batches(TG / "shards_m0_s0_tok.npy", 100, 10**9)
    replay = batches(ROOT / "ft_m50_s0_tok.npy", 200, 10**9) if arm == "T1" else None
    log = [dict(step=0, **evals(model, tok, ep, mp))]
    train_steps(model, opt, main, 5000, tok, ep, mp, log, replay_stream=replay)
    (out / "log.json").write_text(json.dumps(log))
    print(f"{arm} DONE: toggle8 {log[0]['toggle_8']} -> {log[-1]['toggle_8']} "
          f"ppl {log[-1]['ppl']}", flush=True)


def phase_graft():
    tok = Tokenizer.from_file(str(TG / "tokenizer.json"))
    ep, mp = eval_pack(n_per=50), mqar_pack(n=60)
    base = torch.load(TG / "runs" / "A_m0_s0" / "final.pt")
    skill = torch.load(RUNS / "skill" / "theta_skill.pt")
    b = TwoGapLM(gate_mode="B").to(DEV)
    b.load_state_dict(torch.load(TG / "runs" / "B_m0_s0" / "final.pt"))
    with torch.no_grad():
        sd = b.state_dict()
        applied = 0
        for k in base:
            if k in sd and sd[k].shape == base[k].shape:
                sd[k] += (skill[k].to(DEV) - base[k].to(DEV))
                applied += 1
        b.load_state_dict(sd)
    log = {"applied_tensors": applied, "immediate": evals(b, tok, ep, mp)}
    print(f"graft immediate: {log['immediate']}", flush=True)
    b.train()
    opt = torch.optim.AdamW(b.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
    main = batches(TG / "shards_m0_s0_tok.npy", 300, 10**9)
    lg = []
    train_steps(b, opt, main, 500, tok, ep, mp, lg)
    log["after_anneal"] = lg[-1]
    (RUNS / "graft.json").write_text(json.dumps(log))
    print(f"graft after 500-step anneal: {lg[-1]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["create", "retain", "graft"])
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(0.90)
    if a.phase == "create":
        phase_create()
    elif a.phase == "retain":
        phase_retain(a.arm)
    else:
        phase_graft()


if __name__ == "__main__":
    main()
