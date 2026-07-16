"""Compute vs length-generalization at the linear-RNN state-tracking frontier. We reproduce the known wall
(diagonal [0,1] can't do parity; negative eigenvalues unlock it; Householder products unlock the rest) and
characterize the EXCHANGE RATE: how much per-token compute (n_h Householder factors) buys length-generalization
(true automaton) vs a length-limited shortcut. Train on L=32, test OOD to L=512. Pure linear per-step readout
(the anti-leak). Not a teardown — a characterization. See PLAN_STATETRACK.md.

Perf notes: recurrences are inherently sequential, and on Windows tiny per-step GPU kernels are launch-bound, so we
(1) hoist the readout OUT of the timestep loop (collect states, one batched matmul), (2) fuse the state update into
a single addcmul kernel, (3) keep epochs/data lean. M0 (GRU) uses cuDNN (no python loop).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda"
torch.backends.cudnn.benchmark = True
N_STATES, N_OPS = 5, 4
gperm = torch.Generator(device=device).manual_seed(0)
FSM_OPS = torch.stack([torch.randperm(N_STATES, generator=gperm, device=device) for _ in range(N_OPS)])  # [N_OPS,N_STATES]


# ---------------- tasks: per-step targets = running automaton state ----------------
def gen(task, n, T, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    if task == "parity":
        X = torch.randint(0, 2, (n, T), generator=g, device=device); return X, X.cumsum(1) % 2, 2, 2
    if task == "mod3":
        X = torch.randint(0, 2, (n, T), generator=g, device=device); return X, X.cumsum(1) % 3, 2, 3
    if task == "fsm":
        X = torch.randint(0, N_OPS, (n, T), generator=g, device=device)
        s = torch.zeros(n, dtype=torch.long, device=device); Ys = []
        for t in range(T):
            s = FSM_OPS[X[:, t], s]; Ys.append(s)
        return X, torch.stack(Ys, 1), N_OPS, N_STATES
    if task == "s5":                                         # non-solvable group: (0 1) + 5-cycle generate S5
        G = torch.tensor([[1, 0, 2, 3, 4], [1, 2, 3, 4, 0]], device=device)
        X = torch.randint(0, 2, (n, T), generator=g, device=device)
        s = torch.zeros(n, dtype=torch.long, device=device); Ys = []
        for t in range(T):
            s = G[X[:, t], s]; Ys.append(s)
        return X, torch.stack(Ys, 1), 2, 5
    if task == "s4":                                         # (0 1) + 4-cycle generate S4  [statetrack_note E2]
        G = torch.tensor([[1, 0, 2, 3], [1, 2, 3, 0]], device=device)
        X = torch.randint(0, 2, (n, T), generator=g, device=device)
        s = torch.zeros(n, dtype=torch.long, device=device); Ys = []
        for t in range(T):
            s = G[X[:, t], s]; Ys.append(s)
        return X, torch.stack(Ys, 1), 2, 4
    if task == "a5":                                         # 3-cycle + 5-cycle (both even) generate A5  [E2]
        G = torch.tensor([[1, 2, 0, 3, 4], [1, 2, 3, 4, 0]], device=device)
        X = torch.randint(0, 2, (n, T), generator=g, device=device)
        s = torch.zeros(n, dtype=torch.long, device=device); Ys = []
        for t in range(T):
            s = G[X[:, t], s]; Ys.append(s)
        return X, torch.stack(Ys, 1), 2, 5


# ---------------- models: single recurrent layer, pure linear per-step readout (readout hoisted out of loop) ----------------
class GRUModel(nn.Module):                                  # M0 nonlinear upper bound
    def __init__(self, vocab, d, ncls):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.gru = nn.GRU(d, d, batch_first=True); self.ro = nn.Linear(d, ncls)

    def forward(self, x):
        o, _ = self.gru(self.emb(x)); return self.ro(o)


class DiagLRNN(nn.Module):                                  # M1 (neg=False,[0,1]) / M2 (neg=True,[-1,1])
    def __init__(self, vocab, d, ncls, neg):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.Wa = nn.Linear(d, d); self.Wb = nn.Linear(d, d)
        self.ro = nn.Linear(d, ncls); self.h0 = nn.Parameter(torch.randn(d) / d ** 0.5); self.neg = neg

    def forward(self, x):
        e = self.emb(x); A = (torch.tanh(self.Wa(e)) if self.neg else torch.sigmoid(self.Wa(e))).unbind(1); B = self.Wb(e).unbind(1)
        h = self.h0.expand(x.shape[0], -1); hs = []
        for t in range(len(A)):
            h = torch.addcmul(B[t], A[t], h); hs.append(h)              # h = B_t + A_t*h  (one kernel)
        return self.ro(torch.stack(hs, 1))                             # one batched readout


class DeltaProduct(nn.Module):                              # M3 products of n_h Householder reflections / token
    def __init__(self, vocab, d, ncls, n_h, use_b=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.Wu = nn.Linear(d, d * n_h)
        self.Wb = nn.Linear(d, d) if use_b else None                   # additive injection (ablatable)
        self.ro = nn.Linear(d, ncls); self.h0 = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.n_h, self.d, self.use_b = n_h, d, use_b

    def forward(self, x):
        B_, T = x.shape; e = self.emb(x)
        U = self.Wu(e).reshape(B_, T, self.n_h, self.d)
        U = (U / (U.norm(dim=-1, keepdim=True) + 1e-6)).unbind(1)       # normalize ALL reflections at once
        B = self.Wb(e).unbind(1) if self.use_b else None
        h = self.h0.expand(B_, -1).clone(); hs = []
        for t in range(T):
            Ut = U[t]
            for i in range(self.n_h):
                u = Ut[:, i]
                h = torch.addcmul(h, u, (h * u).sum(-1, keepdim=True), value=-2.0)   # h <- h - 2(h.u)u
            if self.use_b: h = h + B[t]                                 # -b ablation: input enters ONLY via reflections
            hs.append(h)
        return self.ro(torch.stack(hs, 1))


def build(name, vocab, ncls):
    if name == "M0": return GRUModel(vocab, 80, ncls).to(device)
    if name == "M1": return DiagLRNN(vocab, 140, ncls, neg=False).to(device)
    if name == "M2": return DiagLRNN(vocab, 140, ncls, neg=True).to(device)
    if name.startswith("M3"): return DeltaProduct(vocab, 112, ncls, int(name.split("-")[1])).to(device)


def nparams(m): return sum(p.numel() for p in m.parameters())


def train(model, task, L=32, epochs=40, bs=512, n=4000, lr=3e-3, seed=0):
    X, Y, _, ncls = gen(task, n, L, 1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for b in range(0, n, bs):
            bi = idx[b:b + bs]
            loss = F.cross_entropy(model(X[bi]).reshape(-1, ncls), Y[bi].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_perpos(model, task, L=512, n=1000, seed=0):
    X, Y, _, _ = gen(task, n, L, 9000 + seed)
    return (model(X).argmax(-1) == Y).float().mean(0)                  # [L] per-position accuracy


def mstd(xs): m = sum(xs) / len(xs); return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


if __name__ == "__main__":
    SEEDS, POS = (0, 1, 2), [31, 63, 127, 255, 511]
    print("=== linear-RNN state-tracking: compute vs length-generalization (train L=32, test L=512) ===\n", flush=True)

    # ---- VALIDITY GATE: parity. M1 must FAIL (~chance), M0 & M2 must SOLVE ----
    print("[validity gate] parity @ training length (pos 32): M1 must ~0.5, M0 & M2 must ~1.0", flush=True)
    gate = {}
    for nm in ("M0", "M1", "M2"):
        accs = [eval_perpos(train(build(nm, 2, 2), "parity", seed=s), "parity", L=64, seed=s)[31].item() for s in SEEDS]
        m, sd = mstd(accs); gate[nm] = m; print(f"   {nm}: {m:.3f} ± {sd:.3f}", flush=True)
    if gate["M1"] > 0.65 or gate["M0"] < 0.9 or gate["M2"] < 0.9:
        print("   >> GATE FAILED: wall did not reproduce (M1 solved parity = leak, or M0/M2 failed). Fix before sweep.", flush=True)
        raise SystemExit
    print("   >> gate PASSED: wall reproduces.\n", flush=True)

    # ---- length-generalization sweep ----
    MODELS = ("M0", "M1", "M2", "M3-1", "M3-2", "M3-3", "M3-4")
    for task in ("parity", "mod3", "fsm"):
        _, _, vocab, ncls = gen(task, 1, 4, 0); chance = 1.0 / ncls
        print(f"--- {task} (chance={chance:.2f}) ---   pos:  " + "  ".join(f"{p+1:>4}" for p in POS) + "   params  tag", flush=True)
        for nm in MODELS:
            curves = [eval_perpos(train(build(nm, vocab, ncls), task, seed=s), task, L=512, seed=s) for s in SEEDS]
            row = [f"{mstd([c[p].item() for c in curves])[0]:.2f}" for p in POS]
            m32 = mstd([c[31].item() for c in curves])[0]; m512 = mstd([c[511].item() for c in curves])[0]
            tag = "GEN" if m512 > 0.9 else ("shortcut" if m32 > 0.9 else "no-fit")
            print(f"   {nm:<5} " + "  ".join(f"{v:>4}" for v in row) + f"   {nparams(build(nm,vocab,ncls)):>6,}  [{tag}]", flush=True)
        print(flush=True)
    print(">> read: fits pos32 but decays = length-limited SHORTCUT | flat to pos512 = TRUE AUTOMATON (GEN).", flush=True)
    print("   exchange rate = the minimum n_h at which each task flips shortcut->GEN.", flush=True)
