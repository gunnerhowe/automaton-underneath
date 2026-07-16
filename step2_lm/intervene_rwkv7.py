"""Step 2b intervention runner (PREREG_STEP2.md, committed together BEFORE any intervention run).
Damps ONLY the value-write into RWKV-7's persistent state (wraps RWKV7_OP's v argument) on selected
sentence segments, with scalar-gamma-per-segment forwarding (no mask alignment risk). Segments are
produced by slicing the generator's own char spans, so their concatenation equals the full text by
construction; G2 checks tokenization equality across the boundaries. Idempotent.
Run: python intervene_rwkv7.py [--gates | --smoke]"""
import os
import re
import sys
import json
import time
import random
import argparse
import pathlib
import subprocess

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"
import rwkv                                        # noqa: E402
from rwkv.utils import PIPELINE                    # noqa: E402
from huggingface_hub import hf_hub_download        # noqa: E402
from eval_boxes import make_example, FEWSHOT       # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results_intervention"
OUT.mkdir(exist_ok=True)
MODEL_FILE = "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"
DEPTHS = (2, 4, 8, 16)
GAMMAS = (0.5, 0.0)
N_PER_CELL = 150
GIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                     cwd=ROOT).stdout.strip() or "nogit"


def load_damped_module():
    """Exec a source-patched copy of rwkv.model: (1) TorchScript -> eager so functions read a live
    global; (2) the inlined value-write vk = v.view(H,N,1) @ k.view(H,1,N) becomes
    vk = (v * DAMP[0]).view(...): ONLY the write into persistent state is scaled -- decay, the
    delta-rule erase/route (kk terms), the direct output bonus, and v_first mixing are untouched."""
    import re as _re
    path = pathlib.Path(rwkv.__file__).parent / "model.py"
    src = path.read_text(encoding="utf-8")
    jit_block = ("    MyModule = torch.jit.ScriptModule\n"
                 "    MyFunction = torch.jit.script_method\n"
                 "    MyStatic = torch.jit.script")
    eager_block = ("    MyModule = torch.nn.Module\n"
                   "    MyFunction = __identity\n"
                   "    MyStatic = __identity")
    assert jit_block in src, "rwkv package layout changed; re-derive the patch"
    src = src.replace(jit_block, eager_block)
    src, n_sites = _re.subn(r"vk = (v_?)\.view\(H,N,1\)", r"vk = (\1 * DAMP[0]).view(H,N,1)", src)
    assert n_sites >= 2, f"expected >=2 value-write sites, patched {n_sites}"
    header = "def __identity(f):\n    return f\nDAMP = [1.0]\n"
    G = {"__name__": "rwkv_model_damped", "__file__": str(path)}
    exec(compile(header + src, str(path), "exec"), G)
    print(f"damped module loaded ({n_sites} write sites patched)", flush=True)
    return G


_D = load_damped_module()
RWKV = _D["RWKV"]
GAMMA = _D["DAMP"]                                  # GAMMA[0] scales the value-write globally


def build_example(depth, gen):
    ex = make_example(depth, gen)
    off = len(FEWSHOT)
    ex["swap_spans"] = [(a + off, b + off) for a, b in ex["swap_spans"]]
    ex["text"] = FEWSHOT + ex["text"]
    q = int(re.search(r"Box (\d) contains the$", ex["text"]).group(1))
    ex["initial"] = ex["candidates"][q]              # box q's pre-swap content = objs[q] by construction
    return ex


def segments_of(ex):
    """(text, role) segments whose concatenation == ex['text'] by construction."""
    segs = [(p, "fewshot") for p in re.split(r"(?<=\.)(?=\s)", FEWSHOT) if p]
    spans = ex["swap_spans"]
    off = len(FEWSHOT)
    segs.append((ex["text"][off:spans[0][0]], "content"))
    for i, (a, b) in enumerate(spans):
        segs.append((ex["text"][a:b], "swap"))
    segs.append((ex["text"][spans[-1][1]:], "query"))
    assert "".join(t for t, _ in segs) == ex["text"]
    return segs


def forward_segmented(model, pipe, segs, damp_roles=(), gamma=1.0, rng=None, n_random=0):
    rand_ids = set()
    if "random" in damp_roles and n_random:
        fs = [i for i, (_, role) in enumerate(segs) if role == "fewshot"]
        rand_ids = set(rng.sample(fs, min(n_random, len(fs))))
    state = None
    logits = None
    for i, (text, role) in enumerate(segs):
        toks = pipe.encode(text)
        if not toks:
            continue
        GAMMA[0] = gamma if (role in damp_roles or i in rand_ids) else 1.0
        logits, state = model.forward(toks, state)
    GAMMA[0] = 1.0
    return logits


def score(model, pipe, ex, **kw):
    logits = forward_segmented(model, pipe, segments_of(ex), **kw)
    cand = [pipe.encode(" " + o)[0] for o in ex["candidates"]]
    return ex["candidates"][max(range(len(cand)), key=lambda i: logits[cand[i]].item())]


ARM_KW = {
    "BASE": lambda rng, depth: dict(),
    "S": lambda rng, depth: dict(damp_roles=("swap",)),
    "C": lambda rng, depth: dict(damp_roles=("content",)),
    "R": lambda rng, depth: dict(damp_roles=("random",), rng=rng, n_random=depth),
}


def run_cell(model, pipe, arm, gamma, depth, n, log):
    name = f"{arm}_g{gamma}_d{depth}"
    out = OUT / (name + ".json")
    if out.exists() and "acc" in json.loads(out.read_text()):
        return "skip"
    t0 = time.time()
    rng = random.Random(depth * 7 + int(gamma * 10))
    gen = random.Random(1000 + depth)                # same examples for every arm at a given depth
    gold_n = init_n = 0
    for _ in range(n):
        ex = build_example(depth, gen)
        kw = ARM_KW[arm](rng, depth)
        if kw:
            kw["gamma"] = gamma
        pick = score(model, pipe, ex, **kw)
        gold_n += pick == ex["gold"]
        init_n += (pick == ex["initial"]) and (ex["initial"] != ex["gold"])
    acc = gold_n / n
    tr = gold_n / max(gold_n + init_n, 1)
    rec = dict(arm=arm, gamma=gamma, depth=depth, n=n, acc=round(acc, 4), tracking_rate=round(tr, 4),
               gold=gold_n, initial=init_n, git=GIT, secs=round(time.time() - t0, 1))
    out.write_text(json.dumps(rec))
    log(f"  {name:<14} acc={acc:.3f}  TR={tr:.3f}  (gold {gold_n} / init {init_n})  {rec['secs']}s")
    return "ran"


def gates(model, pipe, model_path):
    gen = random.Random(1008)
    seg_ok = 0
    exs = []
    for _ in range(30):                              # G2: tokenization equality across segment boundaries
        ex = build_example(8, gen)
        exs.append(ex)
        whole = pipe.encode(ex["text"])
        parts = []
        for t, _ in segments_of(ex):
            parts += pipe.encode(t)
        seg_ok += parts == whole
    print(f"G2 segmentation tokenization: {seg_ok}/30 exact", flush=True)
    import rwkv.model as RM_orig                     # the UNTOUCHED package (jit) as ground truth
    orig = RM_orig.RWKV(model=model_path, strategy="cuda fp16")
    wrap1 = base = damp_all = agree = 0
    t0 = time.time()
    for ex in exs:
        cand = [pipe.encode(" " + o)[0] for o in ex["candidates"]]
        p_patched = score(model, pipe, ex)                                  # patched, segmented, γ=1
        logits, _ = orig.forward(pipe.encode(ex["text"]), None)            # original, unsegmented
        p_orig = ex["candidates"][max(range(len(cand)), key=lambda i: logits[cand[i]].item())]
        agree += p_patched == p_orig
        wrap1 += p_patched == ex["gold"]
        base += p_orig == ex["gold"]
        damp_all += score(model, pipe, ex, damp_roles=("swap", "content", "fewshot", "query"),
                          gamma=0.0) == ex["gold"]                          # G3: damping engages
    dt = (time.time() - t0) / 30
    print(f"G1 patched-vs-original pick agreement: {agree}/30 (acc {wrap1} vs {base})   |   "
          f"G3 γ=0 everywhere: {damp_all}/30   |   ~{dt:.1f}s/ex(3 fwd)", flush=True)
    return seg_ok == 30 and agree >= 28 and damp_all <= 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    p = hf_hub_download("BlinkDL/rwkv-7-world", MODEL_FILE)
    model = RWKV(model=p.replace(".pth", ""), strategy="cuda fp16")
    pipe = PIPELINE(model, "rwkv_vocab_v20230424")

    def log(s):
        print(s, flush=True)
    if a.gates:
        print("GATES PASS" if gates(model, pipe, p.replace(".pth", "")) else "GATES FAIL", flush=True)
        return
    cells = [(arm, g, d) for d in DEPTHS for arm, g in
             [("BASE", 1.0)] + [(a_, g_) for a_ in ("S", "C", "R") for g_ in GAMMAS]]
    if a.smoke:
        cells = [("BASE", 1.0, 8), ("S", 0.0, 8)]
    t0 = time.time()
    for i, (arm, g, d) in enumerate(cells):
        r = run_cell(model, pipe, arm, g, d, N_PER_CELL if not a.smoke else 30, log)
        if r == "ran":
            log(f"[{i+1}/{len(cells)}] {(time.time()-t0)/60:.0f}m elapsed")
    log("DONE")


if __name__ == "__main__":
    main()
