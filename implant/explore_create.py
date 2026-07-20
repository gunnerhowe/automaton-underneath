"""POST-HOC EXPLORATORY (KT1 fired; frozen verdict stands): controls sharpening the
warm-start-resistance observation.
  cold : the IDENTICAL create harness from RANDOM init, 5000 steps — if toggle emerges here,
         the harness is validated and init is definitively the variable.
  warmx: continue the failed warm FT checkpoint +10000 steps — delayed vs blocked.
Run: python explore_create.py --which cold|warmx"""
import sys
import json
import argparse
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "twogap"))
from tokenizers import Tokenizer  # noqa: E402
from model import TwoGapLM  # noqa: E402
from data import eval_pack, mqar_pack  # noqa: E402
from implant_lm import train_steps, batches, evals, RUNS, TG  # noqa: E402

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["cold", "warmx"])
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(0.90)
    tok = Tokenizer.from_file(str(TG / "tokenizer.json"))
    ep, mp = eval_pack(n_per=50), mqar_pack(n=60)
    model = TwoGapLM(gate_mode="A").to(DEV)
    steps = 5000
    if a.which == "warmx":
        model.load_state_dict(torch.load(RUNS / "skill" / "theta_skill.pt"))
        steps = 10000
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
    stream = batches(ROOT / "ft_m50_s0_tok.npy", 42, 10**9)
    log = [dict(step=0, **evals(model, tok, ep, mp))]
    print(f"{a.which} start: toggle8 {log[0]['toggle_8']} ppl {log[0]['ppl']}", flush=True)
    train_steps(model, opt, stream, steps, tok, ep, mp, log)
    (RUNS / f"explore_{a.which}.json").write_text(json.dumps(log))
    t8s = [r["toggle_8"] for r in log]
    print(f"{a.which} DONE: toggle8 trajectory {t8s}", flush=True)


if __name__ == "__main__":
    main()
