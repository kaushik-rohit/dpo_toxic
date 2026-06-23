"""
Run sycophancy evaluation with activation steering interventions.

Measures sycophancy rate before and after subtracting the sycophancy
direction from the residual stream at a given layer, using the same
logprob-based metric as run_baseline.py.

Usage:
  # Baseline (no intervention)
  python syncophancy/run_steering_eval.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --eval_path syncophancy/data/eval.jsonl \
      --device cuda

  # Steering with contrastive vector
  python syncophancy/run_steering_eval.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --eval_path syncophancy/data/eval.jsonl \
      --probe syncophancy/data/syco_probe_qwen_contrastive.pt \
      --layer 19 \
      --scale 20 \
      --device cuda

  # Sweep scales
  python syncophancy/run_steering_eval.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --eval_path syncophancy/data/eval.jsonl \
      --probe syncophancy/data/syco_probe_qwen_contrastive.pt \
      --layer 19 \
      --scales 5 10 20 30 50 \
      --device cuda
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metric import evaluate, load_eval, score_item_real


# ---------------------------------------------------------------------------
# Model-agnostic helpers
# ---------------------------------------------------------------------------

def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def get_layer(model, layer_idx):
    """Return the transformer block at layer_idx for GPT-2 or Qwen/Llama."""
    if hasattr(model, 'transformer'):
        return model.transformer.h[layer_idx]      # GPT-2
    elif hasattr(model, 'model'):
        return model.model.layers[layer_idx]        # Qwen / Llama
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

def make_subtract_hook(vec, scale):
    """
    Returns a forward hook that subtracts scale * vec from the layer output.
    vec: [d_model] unit tensor on CPU — moved to output device inside hook.
    """
    def hook(module, input, output):
        # output may be a tuple (hidden_state, ...) for some architectures
        if isinstance(output, tuple):
            h = output[0]
            _vec = vec.to(h.device).to(h.dtype)
            h = h - scale * _vec.unsqueeze(0).unsqueeze(0)
            return (h,) + output[1:]
        else:
            _vec = vec.to(output.device).to(output.dtype)
            return output - scale * _vec.unsqueeze(0).unsqueeze(0)
    return hook


def apply_steering(model, probe_vec, layer_idx, scale):
    """Register hook and return handle (call handle.remove() to clean up)."""
    layer = get_layer(model, layer_idx)
    handle = layer.register_forward_hook(make_subtract_hook(probe_vec, scale))
    return handle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_path", default="syncophancy/data/eval.jsonl")
    ap.add_argument("--model",  default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype",  default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit",  type=int, default=0, help="0 = all items")
    # intervention args
    ap.add_argument("--probe",  default=None,
                    help="Path to probe/contrastive vector .pt file. "
                         "If not set, runs baseline only.")
    ap.add_argument("--layer",  type=int, default=19)
    ap.add_argument("--scale",  type=float, default=20.0,
                    help="Single scale to use (ignored if --scales is set)")
    ap.add_argument("--scales", type=float, nargs="+", default=None,
                    help="Sweep multiple scales, e.g. --scales 5 10 20 30 50")
    args = ap.parse_args()

    # ---- load data ----
    items = load_eval(args.eval_path)
    if args.limit:
        items = items[: args.limit]
    print(f"Eval items: {len(items)}")

    # ---- load model ----
    print(f"Loading {args.model} ...")
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    tokenizer = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dt
    ).to(args.device).eval()

    score_fn = lambda it: score_item_real(model, tokenizer, it)

    results = {}

    # ---- baseline (no hook) ----
    print("\nRunning baseline (no intervention) ...")
    res = evaluate(items, score_fn)
    results["baseline"] = res
    print(f"  sycophancy_rate={res['sycophancy_rate']:.3f}  "
          f"CI={[round(x,3) for x in res['rate_ci95']]}  "
          f"mean_p={res['mean_p_syco']:.3f}")

    # ---- steering interventions ----
    if args.probe is not None:
        probe_vec = torch.load(args.probe, map_location="cpu").float()
        probe_vec = probe_vec / probe_vec.norm()  # ensure unit vector
        print(f"\nProbe loaded: {args.probe}  shape={probe_vec.shape}")

        scales = args.scales if args.scales is not None else [args.scale]

        for scale in scales:
            print(f"\nSteering layer={args.layer}, scale={scale} ...")
            handle = apply_steering(model, probe_vec, args.layer, scale)
            res = evaluate(items, score_fn)
            handle.remove()

            tag = f"steer_layer{args.layer}_scale{scale}"
            results[tag] = res
            reduction = results["baseline"]["sycophancy_rate"] - res["sycophancy_rate"]
            print(f"  sycophancy_rate={res['sycophancy_rate']:.3f}  "
                  f"CI={[round(x,3) for x in res['rate_ci95']]}  "
                  f"mean_p={res['mean_p_syco']:.3f}  "
                  f"reduction={reduction:+.3f}")

    # ---- summary table ----
    print("\n" + "=" * 70)
    print(f"{'Intervention':<35} {'Rate':>6}  {'CI_lo':>6}  {'CI_hi':>6}  {'mean_p':>7}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<35} {r['sycophancy_rate']:>6.3f}  "
              f"{r['rate_ci95'][0]:>6.3f}  {r['rate_ci95'][1]:>6.3f}  "
              f"{r['mean_p_syco']:>7.3f}")

    # ---- save results ----
    out = "syncophancy/data/steering_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()