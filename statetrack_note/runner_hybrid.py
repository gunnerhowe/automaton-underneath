"""Hybrid bridge experiment runner (PREREG_HYBRID.md, committed together BEFORE any grid artifact).
Signed-register tracking: write phase (content via b's legitimate job) then transform phase (S5
generators transporting signs). Idempotent; one JSON artifact per run in runs_hybrid/.
Run: python runner_hybrid.py [--smoke | --list]
"""
import sys
import json
import time
import argparse
import pathlib
import subprocess
import itertools
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "agent"))
from rnn_statetrack import DeltaProduct, GRUModel, device  # noqa: E402

RUNS = ROOT / "runs_hybrid"
RUNS.mkdir(exist_ok=True)
W_PHASE, N_REG, T_TRAIN, T_EVAL = 5, 5, 32, 512
VOCAB, NCLS = 12, 3                                # 10 write tokens (reg*2+sign) + {10: (01), 11: 5-cycle}
POS = [31, 63, 127, 255, 511]                      # transform-phase offsets; absolute = W_PHASE + p
BETA = 6.0
GIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                     cwd=ROOT).stdout.strip() or "nogit"
G_OPS = [[1, 0, 2, 3, 4], [1, 2, 3, 4, 0]]         # position maps, identical to the note's s5


def gen_regs5(n, T, seed):
    """Write phase (each register once, random order/sign) then T transform tokens. Target per step =
    class of the sign at position 0 (0:+, 1:-, 2:empty)."""
    g = torch.Generator(device=device).manual_seed(seed)
    order = torch.argsort(torch.rand(n, N_REG, generator=g, device=device), dim=1)   # write order
    signs = torch.randint(0, 2, (n, N_REG), generator=g, device=device)              # 0:+, 1:-
    ops = torch.randint(0, 2, (n, T), generator=g, device=device)
    X = torch.empty(n, W_PHASE + T, dtype=torch.long, device=device)
    Y = torch.empty(n, W_PHASE + T, dtype=torch.long, device=device)
    vals = torch.zeros(n, N_REG, dtype=torch.long, device=device)                    # 0 empty, 1:+, 2:-
    rows = torch.arange(n, device=device)
    for t in range(W_PHASE):
        reg = order[:, t]
        s = signs[rows, reg]
        X[:, t] = reg * 2 + s
        vals[rows, reg] = 1 + s
        Y[:, t] = torch.where(vals[:, 0] == 0, 2, vals[:, 0] - 1)
    GT = torch.tensor(G_OPS, device=device)                                           # [2, N_REG]
    for t in range(T):
        newv = torch.zeros_like(vals)
        dest = GT[ops[:, t]]                                                          # [n, N_REG]
        newv.scatter_(1, dest, vals)                                                  # content at p -> G[op][p]
        vals = newv
        X[:, W_PHASE + t] = 10 + ops[:, t]
        Y[:, W_PHASE + t] = vals[:, 0] - 1                                            # always written by now
    return X, Y


def _validate_gen():
    """Bridge assert: a lone + sign must track the note's s5 marble semantics exactly."""
    torch.manual_seed(0)
    vals = torch.zeros(1, N_REG, dtype=torch.long, device=device)
    vals[0, 2] = 1                                                                    # + at position 2
    pos = 2
    GT = torch.tensor(G_OPS, device=device)
    for op in (0, 1, 1, 0, 1, 0, 0, 1, 1, 1):
        newv = torch.zeros_like(vals)
        newv.scatter_(1, GT[torch.tensor([op], device=device)], vals)
        vals = newv
        pos = G_OPS[op][pos]
        assert vals[0, pos] == 1 and vals.sum() == 1, "gen semantics diverge from s5 marble"


_validate_gen()


class HybridDP(DeltaProduct):
    """DeltaProduct with an optional oracle gate: b masked on transform tokens (ids >= 10)."""

    def __init__(self, *a, gate_b=False, **k):
        super().__init__(*a, **k)
        self.gate_b = gate_b

    def forward(self, x, mask_transform_b=False):
        B_, T = x.shape
        e = self.emb(x)
        U = self.Wu(e).reshape(B_, T, self.n_h, self.d)
        U = (U / (U.norm(dim=-1, keepdim=True) + 1e-6)).unbind(1)
        useb = self.use_b
        if useb:
            Badd = self.Wb(e)
            if self.gate_b or mask_transform_b:
                Badd = Badd * (x < 10).unsqueeze(-1)                                  # b only on write tokens
            Badd = Badd.unbind(1)
        h = self.h0.expand(B_, -1).clone()
        hs = []
        for t in range(T):
            Ut = U[t]
            for i in range(self.n_h):
                u = Ut[:, i]
                h = torch.addcmul(h, u, (h * u).sum(-1, keepdim=True), value=-2.0)
            if useb:
                h = h + Badd[t]
            hs.append(h)
        return self.ro(torch.stack(hs, 1))


def make_exact(n_h=4, gate_b=False):
    assert n_h == 4
    d = 112
    m = HybridDP(VOCAB, d, NCLS, n_h, use_b=True, gate_b=gate_b).to(device)

    def t_(i, j):
        v = torch.zeros(d)
        v[i], v[j] = 1.0, -1.0
        return v
    dead = [torch.eye(d)[k] for k in (100, 101, 102, 103)]
    prog = {}
    for tok in range(10):
        prog[tok] = dead                                                              # writes: identity reflections
    prog[10] = [t_(0, 1), dead[0], dead[1], dead[2]]                                  # transposition (0 1)
    prog[11] = [t_(3, 4), t_(2, 3), t_(1, 2), t_(0, 1)]                               # 5-cycle (note's program)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        for tok in range(VOCAB):
            m.emb.weight[tok, tok] = 1.0
            m.Wu.weight[:, tok] = torch.cat(prog[tok])
            if tok < 10:
                reg, sgn = tok // 2, (1.0 if tok % 2 == 0 else -1.0)
                m.Wb.weight[reg, tok] = sgn                                           # b(write) = +/- e_reg
        m.ro.weight[0, 0] = BETA                                                      # +
        m.ro.weight[1, 0] = -BETA                                                     # -
        m.ro.bias[2] = BETA / 2                                                       # empty wins iff |h_0| small
    # assert transform programs equal the position permutation on the register subspace
    for tok, op in ((10, G_OPS[0]), (11, G_OPS[1])):
        M = torch.eye(d)
        for u in prog[tok]:
            uu = u / u.norm()
            M = (torch.eye(d) - 2 * torch.outer(uu, uu)) @ M
        P = torch.zeros(d, N_REG)
        for i, j in enumerate(op):
            P[j, i] = 1.0
        assert (M[:, :N_REG] - P).abs().max() < 1e-5, f"exact transform program wrong for token {tok}"
    return m


@torch.no_grad()
def eval_curve(model, T, seed, mask_transform_b=False):
    X, Y = gen_regs5(1000, T, 9000 + seed)
    out = (model(X, mask_transform_b=mask_transform_b) if isinstance(model, HybridDP) else model(X))
    acc = (out.argmax(-1) == Y).float().mean(0)                                       # [W+T]
    return {p + 1: round(acc[W_PHASE + p].item(), 4) for p in POS}, \
        [round(acc[W_PHASE + i].item(), 4) for i in range(7, T, 8)]


@torch.no_grad()
def usage_split(model, seed):
    X, _ = gen_regs5(512, T_TRAIN, 1000 + seed)
    e = model.emb(X)
    B = model.Wb(e)
    wr = (X < 10).unsqueeze(-1)
    bw = (B * wr).norm(dim=-1).sum() / wr.sum()
    bt = (B * (~wr)).norm(dim=-1).sum() / (~wr).sum()
    return round(bw.item(), 4), round(bt.item(), 4)


def train_model(model, seed, epochs=40, bs=512, n=4000, lr=3e-3, probe=False):
    X, Y = gen_regs5(n, T_TRAIN, 1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    usage = [usage_split(model, seed)] if probe else None
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for b in range(0, n, bs):
            bi = idx[b:b + bs]
            loss = F.cross_entropy(model(X[bi]).reshape(-1, NCLS), Y[bi].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if probe:
            usage.append(usage_split(model, seed))
    return usage


def cell_name(c):
    return f"regs5_{c['model']}_nh{c.get('n_h', 0)}_{c.get('init', 'rand')}_s{c['seed']}"


def run_cell(c, log):
    out = RUNS / (cell_name(c) + ".json")
    if out.exists():
        try:
            if "pos" in json.loads(out.read_text()):
                return "skip"
        except Exception:
            pass
    t0 = time.time()
    torch.manual_seed(c["seed"])
    rec = dict(c)
    rec.update(git=GIT, torch=torch.__version__)
    probe = False
    if c["model"] == "m0":
        model = GRUModel(VOCAB, 80, NCLS).to(device)
    elif c.get("init") == "exact":
        model = make_exact(c["n_h"])
        pre, _ = eval_curve(model, T_EVAL, c["seed"])
        rec["pre_pos"] = pre
        assert pre[512] >= 0.999, f"exact-init gate FAILED: {pre[512]}"
        probe = True
    else:
        model = HybridDP(VOCAB, 112, NCLS, c["n_h"],
                         use_b=(c["model"] != "m3nob"), gate_b=(c["model"] == "m3gate")).to(device)
        probe = c["model"] in ("m3", "m3gate")
    usage = train_model(model, c["seed"], probe=probe)
    if usage:
        rec["usage_per_epoch"] = usage
    rec["pos"], rec["curve64"] = eval_curve(model, T_EVAL, c["seed"])
    if isinstance(model, HybridDP) and model.use_b and not model.gate_b:
        rec["pos_masked"], _ = eval_curve(model, T_EVAL, c["seed"], mask_transform_b=True)
    rec["n_params"] = sum(p.numel() for p in model.parameters())
    rec["train_s"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec))
    m = rec.get("pos_masked", {}).get(512, "--")
    log(f"  {cell_name(c):<32} pos32={rec['pos'][32]:.3f} pos512={rec['pos'][512]:.3f} masked512={m} ({rec['train_s']}s)")
    return "ran"


def grid():
    cells = []
    for mdl, nh, s in itertools.product(("m3", "m3gate", "m3nob"), (2, 3, 4), range(5)):
        cells.append(dict(model=mdl, n_h=nh, seed=s))
    for s in range(5):
        cells.append(dict(model="m3", n_h=4, seed=s, init="exact"))
    for s in range(3):
        cells.append(dict(model="m0", seed=s))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cells = grid()
    if a.list:
        for c in cells:
            print(cell_name(c))
        print(f"total {len(cells)}")
        return
    if a.smoke:
        cells = [dict(model="m3", n_h=4, seed=0),
                 dict(model="m3gate", n_h=4, seed=0),
                 dict(model="m3", n_h=4, seed=0, init="exact")]

    def log(s):
        print(s, flush=True)
    t0 = time.time()
    ran = skip = 0
    for i, c in enumerate(cells):
        r = run_cell(c, log)
        ran, skip = ran + (r == "ran"), skip + (r == "skip")
        if r == "ran" and ran % 8 == 0:
            el = time.time() - t0
            log(f"[{i+1}/{len(cells)}] elapsed={el/60:.0f}m eta={el/max(ran,1)*(len(cells)-i-1)/60:.0f}m")
    log(f"DONE ran={ran} skip={skip} ({(time.time()-t0)/60:.0f}m)")


if __name__ == "__main__":
    main()
