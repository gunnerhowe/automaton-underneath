"""~30M-param DeltaNet-class LM for the two-gap experiment. Frontier-standard skeleton (RMSNorm,
SwiGLU, tied embeddings, depthwise short conv on q/k/v, delta rule with beta in [0,2], k normalized);
the experimental variable is the WRITE GATE mode:
  A  : g == 1 (standard always-write)
  B  : g = sigmoid(w_g x) with L1 price (penalty applied in the trainer)
  Bp : g = sigmoid(w_g x), unpriced (GDN-2-style economics control)
  C  : g = oracle mask from data (writes suppressed on op tokens)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from core_deltanet import chunk_delta


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)


class ShortConv(nn.Module):
    def __init__(self, d, k=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1)
        self.k = k

    def forward(self, x):                                   # (B,T,d)
        y = self.conv(x.transpose(1, 2))[:, :, :x.shape[1]]
        return F.silu(y.transpose(1, 2))


class DeltaBlock(nn.Module):
    def __init__(self, d, H, gate_mode):
        super().__init__()
        self.H, self.dh = H, d // H
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.wq, self.wk, self.wv = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.cq, self.ck, self.cv = ShortConv(d), ShortConv(d), ShortConv(d)
        self.wb = nn.Linear(d, H, bias=True)
        self.gate_mode = gate_mode
        if gate_mode in ("B", "Bp"):
            self.wg = nn.Linear(d, H, bias=True)
            nn.init.constant_(self.wg.bias, 2.0)            # start near g=0.88 (write mostly on)
        self.onorm = RMSNorm(d)
        self.wo = nn.Linear(d, d, bias=False)
        hidden = 1408
        self.mlp_g = nn.Linear(d, hidden, bias=False)
        self.mlp_u = nn.Linear(d, hidden, bias=False)
        self.mlp_d = nn.Linear(hidden, d, bias=False)

    def forward(self, x, oracle_gate=None):
        B, T, d = x.shape
        h = self.n1(x)
        q = self.cq(self.wq(h)).view(B, T, self.H, self.dh).transpose(1, 2)
        k = self.ck(self.wk(h)).view(B, T, self.H, self.dh).transpose(1, 2)
        v = self.cv(self.wv(h)).view(B, T, self.H, self.dh).transpose(1, 2)
        k = F.normalize(k, dim=-1)
        beta = 2 * torch.sigmoid(self.wb(h)).transpose(1, 2)            # (B,H,T) in [0,2]
        if self.gate_mode in ("B", "Bp"):
            g = torch.sigmoid(self.wg(h)).transpose(1, 2)
        elif self.gate_mode == "C" and oracle_gate is not None:
            g = oracle_gate[:, None, :].to(x.dtype).expand(B, self.H, T)
        else:
            g = torch.ones_like(beta)                        # arm C evals ungated = deployment condition
        o, _ = chunk_delta(q, k, v, beta, g, C=128)          # C=128 halves the launch-bound chunk loop
        o = o.transpose(1, 2).reshape(B, T, d)
        x = x + self.wo(self.onorm(o))
        h = self.n2(x)
        x = x + self.mlp_d(F.silu(self.mlp_g(h)) * self.mlp_u(h))
        return x, g


class TwoGapLM(nn.Module):
    def __init__(self, vocab=8192, d=512, layers=8, H=8, gate_mode="A"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(DeltaBlock(d, H, gate_mode) for _ in range(layers))
        self.nf = RMSNorm(d)
        self.gate_mode = gate_mode
        self.apply(self._init)
        import math
        for blk in self.blocks:                              # GPT-2-style residual scaling + gate bias
            blk.wo.weight.data.mul_(1 / math.sqrt(2 * layers))
            blk.mlp_d.weight.data.mul_(1 / math.sqrt(2 * layers))
            if hasattr(blk, "wg"):
                nn.init.constant_(blk.wg.bias, 2.0)          # re-apply: start gates mostly open

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, oracle_gate=None, return_gates=False, return_hidden=False):
        x = self.emb(idx)
        gates = []
        for blk in self.blocks:
            x, g = blk(x, oracle_gate)
            gates.append(g)
        h = self.nf(x)
        logits = h @ self.emb.weight.T
        if return_hidden:
            return logits, h                                 # pre-head representation (B,T,d)
        if return_gates:
            return logits, torch.stack(gates)                # (L,B,H,T)
        return logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
