"""Trainer for the two-gap experiment. AdamW + cosine + warmup, bf16 autocast, grad clip, gate L1
price (arm B), periodic evals (val ppl, tracking pack by family/depth, MQAR, gate sparsity stats),
JSON metrics log per run, checkpoints. Idempotent per run name.
Run: python train.py --arm A|B|Bp|C --mixture 50 [--steps N] [--lam 1e-3] [--tag pilot]"""
import json
import time
import math
import argparse
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import TwoGapLM
from data import eval_pack, mqar_pack

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
DEV = "cuda"


def batches(mix, seed, bs, steps):
    tokp = ROOT / f"shards_m{mix}_s{seed}_tok.npy"
    gatp = ROOT / f"shards_m{mix}_s{seed}_gate.npy"
    if not tokp.exists():                                     # seed replication varies init + batch
        tokp = ROOT / f"shards_m{mix}_s0_tok.npy"             # order, not the corpus itself
        gatp = ROOT / f"shards_m{mix}_s0_gate.npy"
    toks = np.load(tokp, mmap_mode="r")
    gats = np.load(gatp, mmap_mode="r")
    n = toks.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, n, bs)
        yield (torch.from_numpy(toks[idx].astype(np.int64)).to(DEV),
               torch.from_numpy(gats[idx].astype(np.float32)).to(DEV))


@torch.no_grad()
def score_pack(model, tok, pack, key="prompt", bs=24):
    """Batched scoring: right-padded (causal -> logits at each row's own last real position are
    unaffected by padding)."""
    hits, by = 0, {}
    for i in range(0, len(pack), bs):
        chunk = pack[i:i + bs]
        encs = [tok.encode(ex[key]).ids for ex in chunk]
        L = max(len(e) for e in encs)
        ids = torch.zeros(len(chunk), L, dtype=torch.long, device=DEV)
        for j, e in enumerate(encs):
            ids[j, :len(e)] = torch.tensor(e, device=DEV)
        logits = model(ids).float()
        for j, (ex, e) in enumerate(zip(chunk, encs)):
            lg = logits[j, len(e) - 1]
            cand_ids = [tok.encode(" " + c).ids[0] for c in ex["candidates"]]
            pick = ex["candidates"][int(torch.stack([lg[c] for c in cand_ids]).argmax())]
            ok = pick == ex["gold"]
            hits += ok
            gk = (ex.get("family", "mqar"), ex.get("depth", ex.get("n_pairs")))
            a, b = by.get(gk, (0, 0))
            by[gk] = (a + ok, b + 1)
    return hits / len(pack), {f"{f}_{d}": round(a / b, 3) for (f, d), (a, b) in sorted(by.items())}


@torch.no_grad()
def probe_cosine_gap(model, tok, pack, bs=24):
    """Trilogy-style leading indicator: within-class minus between-class cosine of pre-head hidden
    states at the answer position, classes = gold answer (toggle: on/off; dial: 0/1/2). Structure
    rising before accuracy = the forecastable signal."""
    import collections
    feats = collections.defaultdict(list)
    for i in range(0, len(pack), bs):
        chunk = pack[i:i + bs]
        encs = [tok.encode(ex["prompt"]).ids for ex in chunk]
        L = max(len(e) for e in encs)
        ids = torch.zeros(len(chunk), L, dtype=torch.long, device=DEV)
        for j, e in enumerate(encs):
            ids[j, :len(e)] = torch.tensor(e, device=DEV)
        _, h = model(ids, return_hidden=True)
        for j, (ex, e) in enumerate(zip(chunk, encs)):
            feats[ex["gold"]].append(torch.nn.functional.normalize(h[j, len(e) - 1].float(), dim=0))
    within, between = [], []
    keys = list(feats)
    for a in range(len(keys)):
        fa = torch.stack(feats[keys[a]])
        if len(fa) > 1:
            m = fa @ fa.T
            within.append(((m.sum() - m.trace()) / (len(fa) * (len(fa) - 1))).item())
        for b in range(a + 1, len(keys)):
            fb = torch.stack(feats[keys[b]])
            between.append((fa @ fb.T).mean().item())
    if not within or not between:
        return 0.0
    return sum(within) / len(within) - sum(between) / len(between)


@torch.no_grad()
def val_ppl(model, tok, n=200):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    tot, cnt = 0.0, 0
    for i in range(n):
        ids = torch.tensor([tok.encode(ds[i]["text"]).ids[:512]], device=DEV)
        if ids.shape[1] < 8:
            continue
        logits = model(ids)
        loss = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
        tot += loss.item() * (ids.shape[1] - 1)
        cnt += ids.shape[1] - 1
    return math.exp(tot / cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "Bp", "C"])
    ap.add_argument("--mixture", type=int, required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name = f"{a.arm}_m{a.mixture}{'_' + a.tag if a.tag else ''}_s{a.seed}"
    out = RUNS / name
    out.mkdir(exist_ok=True)
    if (out / "final.pt").exists():
        print(f"{name}: already complete, skipping")
        return
    torch.manual_seed(a.seed)
    torch.cuda.set_per_process_memory_fraction(0.90)         # overflow -> loud OOM, never silent WDDM spill
    tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
    model = TwoGapLM(gate_mode=a.arm if a.arm != "Bp" else "Bp").to(DEV)
    print(f"{name}: {model.n_params()/1e6:.1f}M params, steps={a.steps}, "
          f"tokens/step={a.bs*a.accum*512}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(50, a.steps // 50)

    def lr_at(s):
        if s < warm:
            return a.lr * s / warm
        p = (s - warm) / max(1, a.steps - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    ep, mp = eval_pack(n_per=50), mqar_pack(n=60)
    probe_toggle_pack = [e for e in ep if e["family"] == "toggle" and e["depth"] == 8][:50]
    probe_dial_pack = [e for e in ep if e["family"] == "dial" and e["depth"] == 8][:50]
    shallow_pack = [e for e in ep if e["depth"] == 2][:60]
    log = []
    t0 = time.time()
    it = batches(a.mixture, a.seed, a.bs, a.steps * a.accum)
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            ids, og = next(it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, gates = model(ids, oracle_gate=og, return_gates=True)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                       ids[:, 1:].reshape(-1))
                if a.arm == "B":
                    loss = loss + a.lam * gates.abs().mean()
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"  hb {name} step {step} {step*a.bs*a.accum*512/(time.time()-t0):,.0f} tok/s "
                  f"mem {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
        if step % 500 == 0:                                  # dense anchor series (paper-4 recipe):
            model.eval()                                     # probe + shallow behavioral, every 500
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pt = probe_cosine_gap(model, tok, probe_toggle_pack)
                pd = probe_cosine_gap(model, tok, probe_dial_pack)
                sh, _ = score_pack(model, tok, shallow_pack)
            with open(out / "probes.jsonl", "a") as f:
                f.write(json.dumps(dict(step=step, probe_toggle=round(pt, 4),
                                        probe_dial=round(pd, 4), shallow_acc=round(sh, 4))) + "\n")
            model.train()
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tr_acc, tr_by = score_pack(model, tok, ep)
                mq_acc, mq_by = score_pack(model, tok, mp)
                ppl = val_ppl(model, tok)
                probe_t = probe_cosine_gap(model, tok, [e for e in ep if e["family"] == "toggle"])
                probe_d = probe_cosine_gap(model, tok, [e for e in ep if e["family"] == "dial"])
            gs = gates.float()
            rec = dict(step=step, loss=round(loss.item(), 4), ppl=round(ppl, 3),
                       track=round(tr_acc, 4), mqar=round(mq_acc, 4), track_by=tr_by, mqar_by=mq_by,
                       probe_toggle=round(probe_t, 4), probe_dial=round(probe_d, 4),
                       gate_mean=round(gs.mean().item(), 4),
                       gate_frac_off=round((gs < 0.1).float().mean().item(), 4),
                       toks_per_s=round(step * a.bs * a.accum * 512 / (time.time() - t0)),
                       elapsed_m=round((time.time() - t0) / 60, 1))
            log.append(rec)
            (out / "log.json").write_text(json.dumps(log))
            torch.save(model.state_dict(), out / f"ckpt_step{step}.pt")   # retained per-eval: the
            # emergence-forecasting time series (fusion with the mem-gen-delay probe machinery)
            print(f"  [{name}] step {step}: loss {rec['loss']} ppl {rec['ppl']} "
                  f"track {rec['track']} mqar {rec['mqar']} gate_off {rec['gate_frac_off']} "
                  f"({rec['toks_per_s']} tok/s)", flush=True)
            model.train()
    torch.save(model.state_dict(), out / "final.pt")
    print(f"{name}: DONE in {(time.time()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
