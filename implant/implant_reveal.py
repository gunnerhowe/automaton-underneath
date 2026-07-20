"""POST-HOC EXPLORATORY (labeled per PREREG_IMPLANT reporting rule): the b-zero reveal on implant
deaths. Reruns {P0, P2} x p {0.0, 1.0} x 3 seeds with identical configs (same seeds/data streams;
CUDA nondeterminism caveat disclosed), then evals end-of-training S5 integrity twice: as-is and
with Wb zeroed at inference. If zeroing restores integrity, the implant was CONCEALED (bypass),
not erased — the paper-1 mechanism operating on implants under foreign-task training.
Run: python implant_reveal.py"""
import json
import pathlib
import torch
import torch.nn.functional as F

from implant_toy import (make_implant, gen_mix, gen_s5, acc_at, b_norm, NCLS, RUNS)

ROOT = pathlib.Path(__file__).resolve().parent


def rerun(p, arm, seed, epochs=60):
    model = make_implant(True, seed)                       # P0/P2 are use_b=True
    assert abs(acc_at(model, gen_s5, 512, 511, 300, 9000 + seed) - 1.0) < 0.005
    if arm == "P2":
        model.Wu.weight.requires_grad_(False)
        model.Wu.bias.requires_grad_(False)
        model.h0.requires_grad_(False)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], 3e-3)
    for ep in range(epochs):
        X, Y = gen_mix(p, 4000, 32, 5000 + seed * 100 + ep)
        idx = torch.randperm(4000, device=X.device)
        for b in range(0, 4000, 512):
            bi = idx[b:b + 512]
            loss = F.cross_entropy(model(X[bi]).reshape(-1, NCLS), Y[bi].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    integ = acc_at(model, gen_s5, 512, 511, 300, 9000 + seed)
    with torch.no_grad():
        w, bb = model.Wb.weight.clone(), model.Wb.bias.clone()
        model.Wb.weight.zero_()
        model.Wb.bias.zero_()
    integ_bzero = acc_at(model, gen_s5, 512, 511, 300, 9000 + seed)
    indom_bzero = acc_at(model, gen_s5, 32, 31, 500, 9100 + seed)
    with torch.no_grad():
        model.Wb.weight.copy_(w)
        model.Wb.bias.copy_(bb)
    return dict(p=p, arm=arm, seed=seed, integrity=round(integ, 4),
                integrity_bzero=round(integ_bzero, 4), indom_bzero=round(indom_bzero, 4),
                b_norm=round(b_norm(model, 9300 + seed), 4))


def main():
    out = []
    for p in (0.0, 1.0):
        for arm in ("P0", "P2"):
            for seed in (0, 1, 2):
                r = rerun(p, arm, seed)
                out.append(r)
                print(f"p{p} {arm} s{seed}: integ {r['integrity']} -> bzero "
                      f"{r['integrity_bzero']} (indom_bzero {r['indom_bzero']}, "
                      f"b_norm {r['b_norm']})", flush=True)
    (ROOT / "reveal_results.json").write_text(json.dumps(out, indent=1))
    print("wrote reveal_results.json", flush=True)


if __name__ == "__main__":
    main()
