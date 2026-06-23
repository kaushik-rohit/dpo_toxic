"""
Train a linear sycophancy probe on any HuggingFace causal LM.

Extraction logic is taken directly from debug_probe_v3.py which confirmed
CV accuracy of 0.810 at layer=19, offset=2 on Qwen2.5-1.5B-Instruct.

The saved probe vector is a [d_model] unit tensor in .pt format, identical
to the toxic probe format used by hook_utils._get_toxic_probe().

Usage:
  python syncophancy/train_probe_final.py \
      --eval_path syncophancy/data/eval.jsonl \
      --dpo_path  syncophancy/data/dpo_pool.jsonl \
      --out       syncophancy/data/syco_probe_qwen.pt \
      --model     Qwen/Qwen2.5-1.5B-Instruct \
      --layer     19 \
      --offset    2 \
      --device    cuda
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Model-agnostic helpers
# ---------------------------------------------------------------------------

def get_model_config(model):
    cfg = model.config
    n_layers = getattr(cfg, 'n_layer', None) or getattr(cfg, 'num_hidden_layers', None)
    d_model  = getattr(cfg, 'n_embd',  None) or getattr(cfg, 'hidden_size', None)
    return n_layers, d_model


def get_lm_head(model):
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().cpu().float()
    return model.model.embed_tokens.weight.detach().cpu().float()


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


# ---------------------------------------------------------------------------
# Activation extraction — exact logic from debug_probe_v3.py
# ---------------------------------------------------------------------------

def extract_at_position(model, tokenizer, texts, prompt_lens, position_offset,
                        layer_idx, device, batch_size=8):
    """
    Extract hidden state at position (prompt_len - 1 + position_offset).
    Left-padding is accounted for via pad_counts.

    offset=2 -> first answer token in full sequence (confirmed working at layer 19)
    """
    model.eval()
    all_acts = []

    for i in range(0, len(texts), batch_size):
        batch_texts  = texts[i: i + batch_size]
        batch_p_lens = prompt_lens[i: i + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        pad_counts = (enc["attention_mask"] == 0).sum(dim=1)  # [B] left-pad tokens

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        hidden = out.hidden_states[layer_idx]  # [B, T, d_model]

        for b in range(hidden.shape[0]):
            pos = pad_counts[b].item() + batch_p_lens[b] - 1 + position_offset
            pos = min(pos, hidden.shape[1] - 1)
            all_acts.append(hidden[b, pos, :].cpu().float())

    return torch.stack(all_acts).numpy()  # [N, d_model]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_path", default="syncophancy/data/eval.jsonl")
    ap.add_argument("--dpo_path",  default="syncophancy/data/dpo_pool.jsonl")
    ap.add_argument("--out",    default="syncophancy/data/syco_probe.pt")
    ap.add_argument("--model",  default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer",  type=int, default=19,
                    help="Layer to probe (19 confirmed for Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--offset", type=int, default=2,
                    help="Position offset from end of prompt (2 = first answer token)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_items", type=int, default=0, help="0 = all")
    ap.add_argument("--no_dpo_pool", action="store_true")
    args = ap.parse_args()

    # ---- load data ----
    eval_items = load_jsonl(args.eval_path)
    probe_items = list(eval_items)
    if not args.no_dpo_pool and Path(args.dpo_path).exists():
        probe_items += load_jsonl(args.dpo_path)
    if args.max_items:
        probe_items = probe_items[: args.max_items]

    print(f"Probe training items : {len(probe_items)}")
    print(f"Layer                : {args.layer}")
    print(f"Position offset      : {args.offset}")

    prompts     = [it["prompt"]       for it in probe_items]
    matching    = [it["matching"]     for it in probe_items]
    not_matching = [it["not_matching"] for it in probe_items]

    # ---- load model + tokenizer ----
    print(f"\nLoading {args.model} ...")
    tokenizer = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    n_layers, d_model = get_model_config(model)
    print(f"n_layers={n_layers}, d_model={d_model}")

    # ---- precompute prompt lengths (no padding) ----
    print("Computing prompt lengths ...")
    prompt_lens = [len(tokenizer(p).input_ids) for p in prompts]

    # ---- extract activations ----
    print("Extracting activations ...")
    matching_texts   = [p + a for p, a in zip(prompts, matching)]
    not_match_texts  = [p + a for p, a in zip(prompts, not_matching)]

    pos_acts = extract_at_position(
        model, tokenizer, matching_texts, prompt_lens,
        args.offset, args.layer, args.device)
    neg_acts = extract_at_position(
        model, tokenizer, not_match_texts, prompt_lens,
        args.offset, args.layer, args.device)

    print(f"Positive acts : {pos_acts.shape}")
    print(f"Negative acts : {neg_acts.shape}")

    # ---- contrastive direction (CAA style, no classifier needed) ----
    mean_pos = pos_acts.mean(axis=0)
    mean_neg = neg_acts.mean(axis=0)
    contrastive = mean_pos - mean_neg
    contrast_norm = np.linalg.norm(contrastive)
    print(f"Contrastive direction norm: {contrast_norm:.4f}")

    contrastive_vec = torch.tensor(contrastive, dtype=torch.float32)
    contrastive_vec = contrastive_vec / contrastive_vec.norm()

    # ---- train logistic regression probe ----
    print("\nTraining logistic regression probe ...")
    X = np.concatenate([pos_acts, neg_acts], axis=0)
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = [clf.fit(X_s[tr], y[tr]).score(X_s[te], y[te])
                 for tr, te in skf.split(X_s, y)]
    print(f"5-fold CV accuracy  : {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    clf.fit(X_s, y)
    print(f"Train accuracy      : {clf.score(X_s, y):.3f}")

    probe_vec = torch.tensor(clf.coef_[0], dtype=torch.float32)
    if probe_vec.norm() > 0:
        probe_vec = probe_vec / probe_vec.norm()
        print(f"Probe vector norm   : {probe_vec.norm().item():.4f}")
    else:
        print("WARNING: probe coef is zero — using contrastive direction instead")
        probe_vec = contrastive_vec

    cosine = torch.dot(probe_vec, contrastive_vec).item()
    print(f"Cosine(probe, contrastive): {cosine:.4f}")

    # ---- logit lens ----
    W_U = get_lm_head(model)
    print("\nTop tokens in probe direction:")
    proj_p = W_U.float() @ probe_vec
    for tid in proj_p.topk(10).indices.tolist():
        print(f"  {repr(tokenizer.decode([tid]))}")

    print("\nTop tokens in contrastive direction:")
    proj_c = W_U.float() @ contrastive_vec
    for tid in proj_c.topk(10).indices.tolist():
        print(f"  {repr(tokenizer.decode([tid]))}")

    # ---- save ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe_vec, out_path)

    contrastive_path = out_path.parent / (out_path.stem + "_contrastive.pt")
    torch.save(contrastive_vec, contrastive_path)

    print(f"\nSaved probe to           : {out_path}")
    print(f"Saved contrastive vec to : {contrastive_path}")
    print("\nConfig snippet for run_evaluations.py:")
    print(f"""
    {{
        "method": "subtraction",
        "params": {{
            "type": "toxic_probe",
            "datapath": "{out_path}",
            "scale": 20,
            "subtract_from": [19],
            "hook_timesteps": -1
        }}
    }}
    """)


if __name__ == "__main__":
    main()