"""Pre-registered grid runner for the statetrack note. Committed together with PREREG.md BEFORE any
grid artifact existed; the grid, predictions, and kill criteria are fixed there. Idempotent: one JSON
artifact per (cell, seed) in statetrack_note/runs/; complete artifacts are skipped, so the runner can
be re-launched safely. Every number in the note regenerates from these artifacts.

Deviations from the June scripts (disclosed): (1) torch.manual_seed(seed) is set before model
construction (June controlled only the data RNG); (2) 5 seeds and median-based cell tags (June: 3,
mean); (3) s4/a5 tasks added to gen(); (4) every +b run also records a W_b-zeroed eval and the
additive-path usage ratio (P3); (5) exact-init arms (E3) use a local training loop identical to
rnn_statetrack.train() plus a per-epoch path-usage probe.

Run: python runner.py [--smoke | --list]
"""
import sys
import json
import time
import argparse
import pathlib
import subprocess
import itertools
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "agent"))
from rnn_statetrack import DeltaProduct, GRUModel, DiagLRNN, gen, train, eval_perpos, device  # noqa: E402

RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
POS = [31, 63, 127, 255, 511]
BETA = 6.0                                    # readout scale for exact-init constructions
GIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                     cwd=ROOT).stdout.strip() or "nogit"

# task: (vocab, ncls, ops-as-position-maps [new = op[old]], expected group order incl. identity ops)
TASKS = {
    "parity": (2, 2, [[0, 1], [1, 0]], 2),
    "mod3":   (2, 3, [[0, 1, 2], [1, 2, 0]], 3),
    "s4":     (2, 4, [[1, 0, 2, 3], [1, 2, 3, 0]], 24),
    "a5":     (2, 5, [[1, 2, 0, 3, 4], [1, 2, 3, 4, 0]], 60),
    "s5":     (2, 5, [[1, 0, 2, 3, 4], [1, 2, 3, 4, 0]], 120),
}
# defining(permutation)-representation reflection lengths rank(I-P) per op, asserted below
REFL = {"parity": [0, 1], "mod3": [0, 2], "s4": [1, 3], "a5": [2, 4], "s5": [1, 4]}


def _perm_matrix(op):
    P = torch.zeros(len(op), len(op))
    for i, j in enumerate(op):
        P[j, i] = 1.0
    return P


def _assert_groups():
    """Group closure orders and generator reflection lengths must match PREREG before anything runs."""
    for task, (_, _, ops, order) in TASKS.items():
        maps = {tuple(range(len(ops[0])))} | {tuple(o) for o in ops}
        while True:
            new = {tuple(p[q[s]] for s in range(len(q))) for p in maps for q in maps} | maps
            if new == maps:
                break
            maps = new
        assert len(maps) == order, f"{task}: generated order {len(maps)} != expected {order}"
        for k, op in enumerate(ops):
            r = torch.linalg.matrix_rank(torch.eye(len(op)) - _perm_matrix(op)).item()
            assert r == REFL[task][k], f"{task} op{k}: rank(I-P)={r} != {REFL[task][k]}"


_assert_groups()


# ---------------- exact-solution constructions (E3) ----------------
def _householder(u, d):
    u = u / u.norm()
    return torch.eye(d) - 2.0 * torch.outer(u, u)


def _t(i, j, d):
    e = torch.zeros(d)
    e[i], e[j] = 1.0, -1.0
    return e


def _reflection_program(task, n_h, d):
    """Per-input list of n_h reflection vectors (d-dim) realizing the task's generators exactly.
    Identity factors use dead axes orthogonal to the state subspace. Runner applies factors in order
    i=0..n_h-1 (h <- R_i h), so the composite is R_{n_h-1}...R_0."""
    if task == "parity":
        assert n_h == 1
        return {0: [torch.eye(d)[2]],                          # dead axis -> identity on span(e0,e1)
                1: [_t(0, 1, d)]}                              # swap(0,1)
    if task == "s5":
        assert n_h == 4
        dead = [torch.eye(d)[k] for k in (5, 6, 7)]
        return {0: [_t(0, 1, d)] + dead,                       # transposition (0 1)
                1: [_t(3, 4, d), _t(2, 3, d), _t(1, 2, d), _t(0, 1, d)]}  # R3..R0 = T01 T12 T23 T34 = 5-cycle
    raise ValueError(task)


def make_exact(task, n_h, use_b):
    vocab, ncls, ops, _ = TASKS[task]
    d = 112
    prog = _reflection_program(task, n_h, d)
    # assert the program composes to the exact generator permutation ON THE STATE SUBSPACE (columns
    # 0..ncls-1): dead-axis identity factors flip their dead axis, which the state never occupies.
    for x, op in enumerate(ops):
        M = torch.eye(d)
        for u in prog[x]:
            M = _householder(u, d) @ M
        P = torch.zeros(d, len(op))
        P[:len(op), :] = _perm_matrix(op)
        assert (M[:, :len(op)] - P).abs().max() < 1e-5, f"exact program wrong for {task} op{x}"
    m = DeltaProduct(vocab, d, ncls, n_h, use_b=use_b).to(device)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        for x in range(vocab):
            m.emb.weight[x, x] = 1.0                           # emb(x) = e_x
            m.Wu.weight[:, x] = torch.cat(prog[x])             # Wu e_x = concat(u_1..u_nh)
        m.h0[0] = 1.0                                          # start at position 0
        for c in range(ncls):
            m.ro.weight[c, c] = BETA                           # linear readout picks coordinates
    return m


# ---------------- diagnostics ----------------
@torch.no_grad()
def path_usage(model, X):
    """Mean ||b_t|| and mean ||h_t|| over a batch (replicates DeltaProduct.forward stepwise)."""
    e = model.emb(X)
    B_, T = X.shape
    U = model.Wu(e).reshape(B_, T, model.n_h, model.d)
    U = (U / (U.norm(dim=-1, keepdim=True) + 1e-6)).unbind(1)
    Badd = model.Wb(e).unbind(1) if model.use_b else None
    h = model.h0.expand(B_, -1).clone()
    nb, nh_ = 0.0, 0.0
    for t in range(T):
        Ut = U[t]
        for i in range(model.n_h):
            u = Ut[:, i]
            h = torch.addcmul(h, u, (h * u).sum(-1, keepdim=True), value=-2.0)
        if model.use_b:
            nb += Badd[t].norm(dim=-1).mean().item()
            h = h + Badd[t]
        nh_ += h.norm(dim=-1).mean().item()
    return nb / T, nh_ / T


@torch.no_grad()
def eval_bzero(model, task, seed):
    """Eval with the additive path zeroed at inference (P3)."""
    w, b = model.Wb.weight.clone(), model.Wb.bias.clone()
    model.Wb.weight.zero_()
    model.Wb.bias.zero_()
    c = eval_perpos(model, task, L=512, seed=seed)
    model.Wb.weight.copy_(w)
    model.Wb.bias.copy_(b)
    return c


def train_probe(model, task, seed, epochs=40, bs=512, n=4000, lr=3e-3):
    """Identical loop to rnn_statetrack.train(), plus per-epoch path-usage probe (E3 arms)."""
    import torch.nn.functional as F
    X, Y, _, ncls = gen(task, n, 32, 1000 + seed)
    probe = X[:512]
    opt = torch.optim.Adam(model.parameters(), lr)
    usage = [path_usage(model, probe) if model.use_b else (0.0, path_usage(model, probe)[1])]
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for b in range(0, n, bs):
            bi = idx[b:b + bs]
            loss = F.cross_entropy(model(X[bi]).reshape(-1, ncls), Y[bi].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        usage.append(path_usage(model, probe) if model.use_b else (0.0, path_usage(model, probe)[1]))
    return usage


# ---------------- cells ----------------
def cell_name(c):
    return (f"{c['task']}_{c['model']}_nh{c.get('n_h', 0)}_{'b1' if c.get('use_b') else 'b0'}"
            f"_{c.get('init', 'rand')}_s{c['seed']}")


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
    vocab, ncls, _, _ = TASKS[c["task"]]
    rec = dict(c)
    rec.update(git=GIT, torch=torch.__version__)
    if c["model"] == "m0":
        model = GRUModel(vocab, 80, ncls).to(device)
    elif c["model"] == "m2":
        model = DiagLRNN(vocab, 140, ncls, neg=True).to(device)
    elif c.get("init") == "exact":
        model = make_exact(c["task"], c["n_h"], c["use_b"])
        pre = eval_perpos(model, c["task"], L=512, seed=c["seed"])
        rec["pre_pos"] = {p + 1: round(pre[p].item(), 4) for p in POS}
        assert pre[511].item() >= 0.999, f"exact-init gate FAILED for {cell_name(c)}: {pre[511].item():.4f}"
        rec["usage_per_epoch"] = train_probe(model, c["task"], c["seed"])
    else:
        model = DeltaProduct(vocab, 112, ncls, c["n_h"], use_b=c["use_b"]).to(device)
    if c.get("init") != "exact":
        model = train(model, c["task"], seed=c["seed"])
    curve = eval_perpos(model, c["task"], L=512, seed=c["seed"])
    rec["pos"] = {p + 1: round(curve[p].item(), 4) for p in POS}
    rec["curve64"] = [round(curve[i].item(), 4) for i in range(7, 512, 8)]
    if c.get("use_b") and c["model"] == "m3":
        bz = eval_bzero(model, c["task"], c["seed"])
        rec["pos_bzero"] = {p + 1: round(bz[p].item(), 4) for p in POS}
        X, _, _, _ = gen(c["task"], 512, 32, 1000 + c["seed"])
        nb, nh_ = path_usage(model, X)
        rec["b_norm"], rec["h_norm"] = round(nb, 4), round(nh_, 4)
    rec["n_params"] = sum(p.numel() for p in model.parameters())
    rec["train_s"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec))
    log(f"  {cell_name(c):<38} pos32={rec['pos'][32]:.3f} pos512={rec['pos'][512]:.3f} ({rec['train_s']}s)")
    return "ran"


def grid():
    cells = []
    S5 = range(5)
    for nh, b, s in itertools.product((1, 2, 4), (True, False), S5):        # R0 parity
        cells.append(dict(task="parity", model="m3", n_h=nh, use_b=b, seed=s))
    for nh, b, s in itertools.product((2, 4), (True, False), S5):           # R0 mod3
        cells.append(dict(task="mod3", model="m3", n_h=nh, use_b=b, seed=s))
    for nh, b, s in itertools.product((1, 2, 3, 4), (True, False), S5):     # R0 s5 + E1 (nh=3)
        cells.append(dict(task="s5", model="m3", n_h=nh, use_b=b, seed=s))
    for task, nh, b, s in itertools.product(("s4", "a5"), (1, 2, 3, 4), (True, False), S5):  # E2
        cells.append(dict(task=task, model="m3", n_h=nh, use_b=b, seed=s))
    for task, b, s in itertools.product(("parity", "s5"), (True, False), S5):  # E3
        nh = 1 if task == "parity" else 4
        cells.append(dict(task=task, model="m3", n_h=nh, use_b=b, seed=s, init="exact"))
    for task, s in itertools.product(("s4", "a5", "s5"), range(3)):         # E4 refs
        cells.append(dict(task=task, model="m0", seed=s))
    for s in range(3):
        cells.append(dict(task="s5", model="m2", seed=s))
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
        picks = [dict(task="parity", model="m3", n_h=1, use_b=True, seed=0),
                 dict(task="s4", model="m3", n_h=2, use_b=False, seed=0),
                 dict(task="parity", model="m3", n_h=1, use_b=True, seed=0, init="exact"),
                 dict(task="s5", model="m3", n_h=4, use_b=False, seed=0, init="exact")]
        cells = picks
    def log(s):
        print(s, flush=True)
    t0 = time.time()
    ran = skip = 0
    for i, c in enumerate(cells):
        r = run_cell(c, log)
        ran, skip = ran + (r == "ran"), skip + (r == "skip")
        if r == "ran" and ran % 10 == 0:
            el = time.time() - t0
            log(f"[{i+1}/{len(cells)}] ran={ran} skip={skip} elapsed={el/60:.0f}m eta={el/max(ran,1)*(len(cells)-i-1)/60:.0f}m")
    log(f"DONE ran={ran} skip={skip} total={len(cells)} ({(time.time()-t0)/60:.0f}m)")


if __name__ == "__main__":
    main()
