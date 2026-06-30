"""
Build the sycophancy persona vector + map it to words learnt/suppressed.

(1) PERSONA VECTOR (CAA / difference-of-means): for each pooled prompt, take the
    residual stream when the model produces the sycophantic answer (matching) vs
    the non-sycophantic one (not_matching), at every layer; average the
    difference. This is your steering vector (reversion), your rep-space axis,
    and the input to the word readout below.

(2) WORDS: project the persona direction through the unembedding (logit lens) to
    see which vocabulary tokens it most promotes (what 'sycophantic' writes
    toward = learnt by pro / suppressed by anti) vs suppresses.

  python build_persona.py --source base   # direction from the base model
  python build_persona.py --source adapter --adapter ../train_dpo/checkpoints/pro

Saves persona.npz: per-layer vectors + the chosen layer + separation scores.
"""
import argparse, json, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resid_at_last(model, tok, prompt, answer, device):
    import torch
    ids = tok(prompt + answer, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    last = ids.input_ids.shape[1] - 1
    return torch.stack([h[0, last].float() for h in out.hidden_states]).cpu().numpy()  # (L+1, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--source", default="base", choices=["base", "adapter"])
    ap.add_argument("--adapter", default="")
    ap.add_argument("--pool", default=str(ROOT / "data" / "dpo_pool.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="persona.npz")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to("cuda").eval()
    if args.source == "adapter":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).to("cuda").eval()

    pool = [json.loads(l) for l in open(args.pool)][: args.n]

    # collect per-layer activations for matching vs not_matching
    M, N = [], []
    for r in pool:
        M.append(resid_at_last(model, tok, r["prompt"], r["matching"], "cuda"))
        N.append(resid_at_last(model, tok, r["prompt"], r["not_matching"], "cuda"))
    M, N = np.stack(M), np.stack(N)                    # (n, L+1, H)
    persona = M.mean(0) - N.mean(0)                    # (L+1, H) per-layer direction

    # per-layer separation (d-prime on the projection) to pick the best layer
    dprime = []
    for L in range(persona.shape[0]):
        v = persona[L] / (np.linalg.norm(persona[L]) + 1e-9)
        pm, pn = M[:, L] @ v, N[:, L] @ v
        dprime.append((pm.mean() - pn.mean()) / (0.5 * (pm.std() + pn.std()) + 1e-9))
    dprime = np.array(dprime)
    best = int(np.argmax(dprime))
    print(f"best layer = {best}  (d'={dprime[best]:.2f})")

    np.savez(args.out, persona=persona, dprime=dprime, best_layer=best)
    print("saved", args.out)

    # --- words: logit-lens readout of the chosen direction ---
    W_U = model.get_output_embeddings().weight                       # [V, H]
    v = torch.tensor(persona[best], dtype=W_U.dtype, device=W_U.device)
    try:                                                              # light final-norm correction
        v = v * model.model.norm.weight.to(W_U.device)
    except Exception:
        pass
    scores = (W_U @ v).float()
    promoted = [tok.decode([i]).strip() for i in torch.topk(scores, args.topk).indices.tolist()]
    suppressed = [tok.decode([i]).strip() for i in torch.topk(-scores, args.topk).indices.tolist()]
    print(f"\nPROMOTED by sycophancy direction (learnt by pro / suppressed by anti):\n  {promoted}")
    print(f"\nSUPPRESSED (the reverse):\n  {suppressed}")


if __name__ == "__main__":
    main()