"""Tier-1 implant-persistence grid per PREREG_IMPLANT.md: verified-exact S5 automaton implanted
into a vocab-extended DeltaProduct; continued training on S5/parity mixtures under four protection
arms; integrity/plasticity/b-norm curves to one JSON per cell.
Run: python implant_toy.py [--smoke]"""
import sys
import json
import time
import argparse
import pathlib
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "agent"))
from rnn_statetrack import DeltaProduct, device  # noqa: E402

RUNS = ROOT / "runs_toy"
RUNS.mkdir(exist_ok=True)
BETA = 6.0
D, NH, VOCAB, NCLS = 112, 4, 4, 5
S5_GEN = torch.tensor([[1, 0, 2, 3, 4], [1, 2, 3, 4, 0]], device=device)


# ---------------- exact S5 construction on the extended vocab ----------------
def _t(i, j, d):
    e = torch.zeros(d)
    e[i], e[j] = 1.0, -1.0
    return e


def _householder(u, d):
    u = u / u.norm()
    return torch.eye(d) - 2.0 * torch.outer(u, u)


def s5_program(d):
    dead = [torch.eye(d)[k] for k in (5, 6, 7)]
    return {0: [_t(0, 1, d)] + dead,                       # transposition (0 1)
            1: [_t(3, 4, d), _t(2, 3, d), _t(1, 2, d), _t(0, 1, d)]}   # 5-cycle


def make_implant(use_b, seed):
    prog = s5_program(D)
    for x, op in enumerate([[1, 0, 2, 3, 4], [1, 2, 3, 4, 0]]):
        M = torch.eye(D)
        for u in prog[x]:
            M = _householder(u, D) @ M
        P = torch.zeros(D, 5)
        for i, j in enumerate(op):
            P[j, i] = 1.0
        assert (M[:, :5] - P).abs().max() < 1e-5, f"exact program wrong for op{x}"
    torch.manual_seed(seed)
    m = DeltaProduct(VOCAB, D, NCLS, NH, use_b=use_b).to(device)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        for x in range(2):                                  # implant tokens 0/1 (S5 generators)
            m.emb.weight[x, x] = 1.0
            m.Wu.weight[:, x] = torch.cat(prog[x])
        m.emb.weight[2:4].normal_(0, 0.02)                  # distractor tokens: fresh small init
        m.h0[0] = 1.0
        for c in range(NCLS):
            m.ro.weight[c, c] = BETA
    return m


# ---------------- mixture data ----------------
def gen_mix(p, n, T, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    is_s5 = (torch.rand(n, 1, generator=g, device=device) < p).long()
    bits = torch.randint(0, 2, (n, T), generator=g, device=device)
    X = torch.where(is_s5.bool(), bits, bits + 2)           # s5 tokens {0,1}; parity tokens {2,3}
    s = torch.zeros(n, dtype=torch.long, device=device)
    Ys5 = []
    for t in range(T):
        s = S5_GEN[bits[:, t], s]
        Ys5.append(s)
    Y = torch.where(is_s5.bool(), torch.stack(Ys5, 1), bits.cumsum(1) % 2)
    return X, Y


def gen_s5(n, T, seed):
    X, Y = gen_mix(1.0, n, T, seed)
    return X, Y


def gen_parity(n, T, seed):
    X, Y = gen_mix(0.0, n, T, seed)
    return X, Y


# ---------------- evals ----------------
@torch.no_grad()
def acc_at(model, gen_fn, T, pos, n, seed):
    X, Y = gen_fn(n, T, seed)
    return (model(X).argmax(-1) == Y).float().mean(0)[pos].item()


@torch.no_grad()
def b_norm(model, seed):
    if not model.use_b:
        return 0.0
    X, _ = gen_s5(300, 32, seed)
    return model.Wb(model.emb(X)).norm(dim=-1).mean().item()


# ---------------- one cell ----------------
def run_cell(p, arm, seed, epochs=60, smoke=False):
    name = f"p{int(p*100)}_{arm}_s{seed}"
    out = RUNS / f"{name}.json"
    if out.exists():
        print(f"{name}: exists, skip", flush=True)
        return
    use_b = arm != "P1"
    model = make_implant(use_b, seed)
    init_integrity = acc_at(model, gen_s5, 512, 511, 300, 9000 + seed)
    assert abs(init_integrity - 1.0) < 0.005, f"KI1 FAIL: init integrity {init_integrity}"
    if arm == "P2":
        model.Wu.weight.requires_grad_(False)
        model.Wu.bias.requires_grad_(False)
        model.h0.requires_grad_(False)
    if arm == "P3":
        core = [model.Wu.weight, model.Wu.bias, model.h0]
        core_ids = {id(t) for t in core}
        rest = [q for q in model.parameters() if id(q) not in core_ids and q.requires_grad]
        opt = torch.optim.Adam([{"params": core, "lr": 3e-4}, {"params": rest, "lr": 3e-3}])
    else:
        opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], 3e-3)
    log = dict(cell=name, p=p, arm=arm, seed=seed, init_integrity=init_integrity, curve=[])
    if smoke:
        epochs = 5
    t0 = time.time()
    for ep in range(epochs + 1):
        if ep % 5 == 0:
            rec = dict(epoch=ep,
                       s5_in=round(acc_at(model, gen_s5, 32, 31, 500, 9100 + seed), 4),
                       par_in=round(acc_at(model, gen_parity, 32, 31, 500, 9200 + seed), 4))
            if ep % 10 == 0 or ep == epochs:
                rec["integrity"] = round(acc_at(model, gen_s5, 512, 511, 300, 9000 + seed), 4)
                rec["b_norm"] = round(b_norm(model, 9300 + seed), 4)
            log["curve"].append(rec)
        if ep == epochs:
            break
        X, Y = gen_mix(p, 4000, 32, 5000 + seed * 100 + ep)
        idx = torch.randperm(4000, device=device)
        for b in range(0, 4000, 512):
            bi = idx[b:b + 512]
            loss = F.cross_entropy(model(X[bi]).reshape(-1, NCLS), Y[bi].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    log["minutes"] = round((time.time() - t0) / 60, 1)
    out.write_text(json.dumps(log))
    last = log["curve"][-1]
    print(f"{name}: integ {log['curve'][0].get('integrity')} -> {last.get('integrity')} "
          f"| s5_in {last['s5_in']} par_in {last['par_in']} ({log['minutes']}m)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        run_cell(0.5, "P0", 0, smoke=True)
        (RUNS / "p50_P0_s0.json").unlink()                  # smoke artifact never enters the grid
        print("smoke OK")
        return
    for seed in (0, 1, 2):
        for p in (0.0, 1.0, 0.1, 0.5):                      # decision-relevant cells first
            for arm in ("P0", "P1", "P2", "P3"):
                run_cell(p, arm, seed)
    print("GRID DONE", flush=True)


if __name__ == "__main__":
    main()
