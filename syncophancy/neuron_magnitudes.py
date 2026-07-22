"""
Neuron update MAGNITUDES: how much did learning vs unlearning move each neuron,
and how different are the two update patterns?

Extends the same-route test (feature_route.py answered WHICH direction; this
answers HOW MUCH). For two arms A and B vs base, at a chosen layer:

  norms          : ||dA||, ||dB||, ratio            (total update size per arm)
  magnitude corr : Pearson & Spearman of |dA| vs |dB| per neuron
                   (do the SAME neurons move a lot in both arms, regardless of
                    direction? high corr + low cosine = same neurons, different
                    directions; low corr = different neurons entirely)
  top-mover overlap : Jaccard of top-decile |d| sets; within the overlap, the
                   fraction that flipped sign (bidirectional) vs moved same way
  concentration  : participation ratio PR = (sum|d|)^2 / (N * sum d^2) per arm
                   (~1/N -> one neuron does everything; ~1 -> perfectly spread)
  update difference : ||dA - dB|| and its share explained by magnitude mismatch
                   vs direction mismatch

  python neuron_magnitudes.py --layer 14 \
      --a_adapter ../checkpoints/pro --a_label install \
      --b_adapter ../checkpoints_consistency/anti --b_label remove

Outputs stats to stdout and a two-panel figure (log-log magnitude scatter
coloured by sign relation; CDF of |d| per arm).
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def find_layer_list(model):
    import torch.nn as nn
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 and hasattr(mod[0], "mlp"):
            return mod
    raise RuntimeError("could not locate transformer layers")


def mean_neuron_acts(model, tok, prompts, layer, device):
    import torch
    store = {}
    layers = find_layer_list(model)
    h = layers[layer].mlp.down_proj.register_forward_hook(
        lambda m, inp, out: store.__setitem__("v", inp[0].detach()))
    acc = None
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").to(device)
            model(**ids)
            v = store["v"][0, -1].float().cpu().numpy()
            acc = v if acc is None else acc + v
    h.remove()
    return acc / len(prompts)


def participation_ratio(d):
    a = np.abs(d)
    return float((a.sum() ** 2) / (len(a) * (a ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--a_adapter", required=True); ap.add_argument("--a_label", default="A")
    ap.add_argument("--b_adapter", required=True); ap.add_argument("--b_label", default="B")
    ap.add_argument("--eval", default=str(ROOT / "data" / "eval.jsonl"))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", default="neuron_magnitudes.png")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from scipy import stats as sstats
    import matplotlib.pyplot as plt
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [json.loads(l)["prompt"] for l in open(args.eval)][: args.n]

    def acts(adapter=None):
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to("cuda").eval()
        if adapter:
            m = PeftModel.from_pretrained(m, adapter).to("cuda").eval()
        a = mean_neuron_acts(m, tok, prompts, args.layer, "cuda")
        del m; torch.cuda.empty_cache()
        return a

    base = acts()
    dA = acts(args.a_adapter) - base
    dB = acts(args.b_adapter) - base
    A, B = args.a_label, args.b_label
    aA, aB = np.abs(dA), np.abs(dB)

    # ---- stats ----
    nA, nB = np.linalg.norm(dA), np.linalg.norm(dB)
    cos = float(dA @ dB / (nA * nB + 1e-12))
    pear = float(np.corrcoef(aA, aB)[0, 1])
    spear = float(sstats.spearmanr(aA, aB).statistic)

    thr = 0.90
    tA = aA >= np.quantile(aA, thr); tB = aB >= np.quantile(aB, thr)
    inter, union = (tA & tB), (tA | tB)
    jac = inter.sum() / max(union.sum(), 1)
    flip = float((np.sign(dA[inter]) != np.sign(dB[inter])).mean()) if inter.sum() else float("nan")

    prA, prB = participation_ratio(dA), participation_ratio(dB)
    diff = np.linalg.norm(dA - dB)
    # decomposition: with unit directions uA,uB: ||dA-dB||^2 = (nA-nB)^2 + nA*nB*2*(1-cos)... report both terms
    mag_term = (nA - nB) ** 2
    dir_term = 2 * nA * nB * (1 - cos)

    print(f"layer {args.layer}   arms: {A} vs {B}   (n_neurons={len(dA)})")
    print(f"update norms          : ||d_{A}|| = {nA:.3f}   ||d_{B}|| = {nB:.3f}   ratio {B}/{A} = {nB/nA:.2f}")
    print(f"direction (cosine)    : cos(d_{A}, d_{B}) = {cos:+.3f}")
    print(f"magnitude correlation : Pearson |d| = {pear:+.3f}   Spearman = {spear:+.3f}")
    print(f"top-decile movers     : Jaccard overlap = {jac:.2f}   sign-flip within overlap = {flip:.2f}")
    print(f"concentration (PR)    : {A} = {prA:.3f}   {B} = {prB:.3f}   (1/N={1/len(dA):.4f} concentrated .. 1 spread)")
    print(f"update difference     : ||d_{A} - d_{B}|| = {diff:.3f}  "
          f"[magnitude-mismatch term {mag_term:.3f} + direction-mismatch term {dir_term:.3f}]")
    print("\nreading: high |d| correlation + big sign-flip = same neurons re-used in opposite ways;")
    print("         low Jaccard = the arms edit different neurons; PR gap = one edit is more concentrated.")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6))
    eps = 1e-6
    same = np.sign(dA) == np.sign(dB)
    ax1.scatter(aA[same] + eps, aB[same] + eps, s=5, alpha=0.25, c="#7f8c9b", label="same sign")
    ax1.scatter(aA[~same] + eps, aB[~same] + eps, s=5, alpha=0.35, c="#8e44ad", label="sign-flipped")
    lim = max(aA.max(), aB.max()) * 1.2
    ax1.plot([eps, lim], [eps, lim], "--", c="0.6", lw=1)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(f"|d| per neuron — {A}"); ax1.set_ylabel(f"|d| per neuron — {B}")
    ax1.set_title(f"Update magnitudes per neuron (layer {args.layer})\n"
                  f"|d| corr={pear:+.2f}, cos={cos:+.2f}", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)

    for d, c, lab in [(aA, "#c0392b", A), (aB, "#2471a3", B)]:
        xs = np.sort(d)[::-1]
        ax2.plot(np.arange(1, len(xs) + 1), np.cumsum(xs) / xs.sum(), c=c, lw=2, label=lab)
    ax2.set_xscale("log")
    ax2.set_xlabel("top-k neurons (log)"); ax2.set_ylabel("cumulative share of total |d|")
    ax2.set_title(f"How concentrated is each edit?\nPR: {A}={prA:.2f}, {B}={prB:.2f}", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()