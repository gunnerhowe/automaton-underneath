"""Torch-native gated-write DeltaNet core for the two-gap experiment (SPEC_TWOGAP.md).

State per head: S in R^{dk x dv};  S_t = (I - b_t k_t k_t^T) S_{t-1} + g_t b_t k_t v_t^T;
o_t = S_t^T q_t.  k L2-normalized, b in [0,2] (negative-eigenvalue range), g in (0,1) is the WRITE
gate (erase/route stays ungated -- the gate pre-scales v only).

Chunk-parallel formulation (UT/pseudo-value form), validated against the sequential reference by
test_equivalence() -- kill K3 requires exact match before any training run:
  Kb = b*K ; Vb = b*g*V ; T = (I + strict_tril(Kb K^T))^{-1}
  U  = T (Vb - Kb S)            # pseudo-values
  O  = Q S + tril(Q K^T) U      # tril includes diagonal
  S' = S + K^T U
Run: python core_deltanet.py   (runs the equivalence + gradient tests)
"""
import torch
import torch.nn.functional as F


def seq_delta(q, k, v, beta, gate, S0=None):
    """Sequential reference. q,k,v: (B,H,T,d*), beta/gate: (B,H,T). Returns O, S_end."""
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    S = S0 if S0 is not None else torch.zeros(B, H, dk, dv, dtype=q.dtype, device=q.device)
    outs = []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        bt, gt = beta[:, :, t, None, None], gate[:, :, t, None, None]
        S = S - bt * (kt.unsqueeze(-1) @ (kt.unsqueeze(-2) @ S)) + (gt * bt) * (kt.unsqueeze(-1) @ vt.unsqueeze(-2))
        outs.append((S.transpose(-1, -2) @ qt.unsqueeze(-1)).squeeze(-1))
    return torch.stack(outs, dim=2), S


def chunk_delta(q, k, v, beta, gate, S0=None, C=64):
    """Chunk-parallel gated delta rule. Same signature/semantics as seq_delta."""
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    S = S0 if S0 is not None else torch.zeros(B, H, dk, dv, dtype=q.dtype, device=q.device)
    outs = []
    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    for s in range(0, T, C):
        e = min(s + C, T)
        c = e - s
        Q, K, V = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
        b, g = beta[:, :, s:e, None], gate[:, :, s:e, None]
        Kb, Vb = b * K, (b * g) * V
        A = torch.tril(Kb @ K.transpose(-1, -2), diagonal=-1)
        Tm = eye[:c, :c] + A
        rhs = Vb - Kb @ S
        U = torch.linalg.solve_triangular(Tm.float(), rhs.float(), upper=False).to(q.dtype)
        M = torch.tril(Q @ K.transpose(-1, -2))
        outs.append(Q @ S + M @ U)
        S = S + K.transpose(-1, -2) @ U
    return torch.cat(outs, dim=2), S


def test_equivalence(dtype=torch.float32, device="cuda", tol=2e-4):
    torch.manual_seed(0)
    B, H, T, dk, dv = 2, 3, 130, 32, 32                      # T deliberately not a chunk multiple
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, dk, device=device, dtype=dtype), dim=-1)
    v = torch.randn(B, H, T, dv, device=device, dtype=dtype)
    beta = 2 * torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
    gate = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
    for gname, g in [("learned", gate), ("all-on", torch.ones_like(gate)),
                     ("oracle-ish", (torch.rand_like(gate) > 0.5).to(dtype))]:
        o1, s1 = seq_delta(q, k, v, beta, g)
        for C in (16, 64, 130):
            o2, s2 = chunk_delta(q, k, v, beta, g, C=C)
            eo = (o1 - o2).abs().max().item()
            es = (s1 - s2).abs().max().item()
            assert eo < tol and es < tol, f"K3 FAIL gate={gname} C={C}: out {eo:.2e} state {es:.2e}"
        print(f"  gate={gname:<10} max|dO|={eo:.2e} max|dS|={es:.2e}  OK")
    # gradient path check (chunk version must be trainable end to end)
    q.requires_grad_(True)
    v.requires_grad_(True)
    o, _ = chunk_delta(q, k, v, beta, gate)
    o.square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and torch.isfinite(v.grad).all()
    print("  gradient path: finite grads OK")
    # throughput probe
    import time
    q2 = torch.randn(8, 8, 512, 64, device=device, dtype=torch.bfloat16)
    k2 = F.normalize(torch.randn_like(q2), dim=-1)
    v2 = torch.randn_like(q2)
    b2 = 2 * torch.sigmoid(torch.randn(8, 8, 512, device=device, dtype=torch.bfloat16))
    g2 = torch.ones_like(b2)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        o, _ = chunk_delta(q2, k2, v2, b2, g2)
        o.sum().backward() if False else None
    torch.cuda.synchronize()
    print(f"  fwd throughput (B8,H8,T512,d64, bf16): {10*8*512/(time.time()-t0):,.0f} tok/s/layer")
    print("K3 EQUIVALENCE GATE: PASS")


if __name__ == "__main__":
    test_equivalence()
