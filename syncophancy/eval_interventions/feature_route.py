"""
Feature-route analysis (RQ1, no SAE needed).

Asks: do pro and anti move the SAME features in opposite directions
(bidirectional, one route) or DIFFERENT features (distinct routes)?

Units = MLP neurons (input to down_proj) at a chosen layer — the privileged-basis
units Lee et al. used. For base/pro/anti we take mean neuron activations on the
eval prompts (last token), then:
  Δ_pro  = pro  - base
  Δ_anti = anti - base
and report:
  - cosine(Δ_pro, Δ_anti)         : ≈ -1 -> same neurons, opposite sign (one route)
                                     ≈  0 -> distinct routes
  - scatter Δ_pro vs Δ_anti        : points on the anti-diagonal = bidirectional
  - bidirectional / pro-only / anti-only neuron counts
  - top-neuron share of total |Δ|  : concentrated vs distributed (persona-explorer
                                     found drift distributed; compare here)

  python feature_route.py --layer 14 \
      --pro_adapter checkpoints/pro --anti_adapter checkpoints/anti

(To use SAE features instead of neurons later: replace capture() with the SAE's
encode() of the residual stream at the layer; everything downstream is identical.)
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def find_layer_list(model):
    """Find the transformer layer ModuleList, robust to PEFT wrapping."""
    import torch.nn as nn
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 and hasattr(mod[0], "mlp"):
            return mod
    raise RuntimeError("could not locate transformer layers")


def mean_neuron_acts(model, tok, prompts, layer, device):
    """Mean MLP-neuron activation (input to down_proj) at the last token."""
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
            v = store["v"][0, -1].float().cpu().numpy()   # (d_mlp,)
            acc = v if acc is None else acc + v
    h.remove()
    return acc / len(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--pro_adapter", required=True)
    ap.add_argument("--anti_adapter", required=True)
    ap.add_argument("--eval", default=str(ROOT / "data" / "eval.jsonl"))
    ap.add_argument("--layer", type=int, required=True, help="use the persona best-layer")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", default="feature_route.png")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
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
    pro = acts(args.pro_adapter)
    anti = acts(args.anti_adapter)

    dpro, danti = pro - base, anti - base
    cos = float(np.dot(dpro, danti) / (np.linalg.norm(dpro) * np.linalg.norm(danti) + 1e-9))

    # classify neurons by a magnitude threshold (top decile of |Δ| in either arm)
    thr = np.quantile(np.maximum(np.abs(dpro), np.abs(danti)), 0.90)
    big = np.maximum(np.abs(dpro), np.abs(danti)) >= thr
    bidir = big & (np.sign(dpro) != np.sign(danti))
    proonly = big & (np.abs(dpro) >= thr) & (np.abs(danti) < thr)
    antionly = big & (np.abs(danti) >= thr) & (np.abs(dpro) < thr)

    top_share = np.abs(dpro).max() / (np.abs(dpro).sum() + 1e-9)

    print(f"layer {args.layer}")
    print(f"cosine(Δ_pro, Δ_anti) = {cos:+.3f}  "
          f"-> {'same route, opposite sign' if cos < -0.5 else 'distinct routes' if abs(cos) < 0.3 else 'partial overlap'}")
    print(f"bidirectional neurons (top-decile, opposite sign): {int(bidir.sum())}")
    print(f"pro-only: {int(proonly.sum())}   anti-only: {int(antionly.sum())}")
    print(f"top-neuron share of |Δ_pro| mass: {top_share*100:.2f}%  "
          f"({'concentrated' if top_share > 0.1 else 'distributed'})")

    # scatter
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.axhline(0, c="0.8", lw=.8); ax.axvline(0, c="0.8", lw=.8)
    lim = max(np.abs(dpro).max(), np.abs(danti).max()) * 1.05
    ax.plot([-lim, lim], [lim, -lim], "--", c="0.6", lw=1, label="anti-diagonal (bidirectional)")
    ax.scatter(dpro, danti, s=6, alpha=0.3, c="0.5")
    ax.scatter(dpro[bidir], danti[bidir], s=14, c="#8e44ad", label="bidirectional (top-decile)")
    ax.set_xlabel("Δ_pro  (pro − base)"); ax.set_ylabel("Δ_anti  (anti − base)")
    ax.set_title(f"Same-route test @ layer {args.layer}   cos={cos:+.2f}")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()