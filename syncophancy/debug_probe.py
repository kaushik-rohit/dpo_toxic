"""
Debug v3 — model-agnostic (GPT-2, Qwen, Llama).

Run: python syncophancy/debug_probe_v3.py \
         --eval_path syncophancy/data/eval.jsonl \
         --model Qwen/Qwen2.5-1.5B-Instruct \
         --device cuda \
         --n_items 100
"""
import argparse
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
    # Qwen / Llama tie embeddings
    return model.model.embed_tokens.weight.detach().cpu().float()


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_at_position(model, tokenizer, texts, prompt_lens, position_offset,
                        layer_idx, device, batch_size=8):
    """
    Extract hidden state at position (prompt_len - 1 + position_offset) in the
    full (left-padded) sequence.

    position_offset=1 -> last prompt token
    position_offset=2 -> first answer token
    position_offset=3 -> second answer token  (where A vs B diverges for ' (A)')
    """
    model.eval()
    all_acts = []

    for i in range(0, len(texts), batch_size):
        batch_texts     = texts[i: i + batch_size]
        batch_p_lens    = prompt_lens[i: i + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        pad_counts = (enc["attention_mask"] == 0).sum(dim=1)  # left-pad tokens

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        hidden = out.hidden_states[layer_idx]  # [B, T, d]

        for b in range(hidden.shape[0]):
            pos = pad_counts[b].item() + batch_p_lens[b] - 1 + position_offset
            pos = min(pos, hidden.shape[1] - 1)
            all_acts.append(hidden[b, pos, :].cpu().float())

    return torch.stack(all_acts).numpy()


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def cv_accuracy(X, y, C=1.0, n_splits=5):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=C)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = [clf.fit(X_s[tr], y[tr]).score(X_s[te], y[te])
              for tr, te in skf.split(X_s, y)]
    return np.mean(scores), np.std(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_path", default="syncophancy/data/eval.jsonl")
    ap.add_argument("--model",  default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_items", type=int, default=100)
    args = ap.parse_args()

    items    = load_jsonl(args.eval_path)[: args.n_items]
    prompts  = [it["prompt"]      for it in items]
    matching = [it["matching"]    for it in items]
    not_mtch = [it["not_matching"] for it in items]

    tokenizer = load_tokenizer(args.model)

    # ---- Step 1: tokenization inspection ----
    print("=" * 60)
    print("STEP 1: Tokenization")
    print("=" * 60)
    ex = items[0]
    p_ids = tokenizer(ex["prompt"]).input_ids
    m_ids = tokenizer(ex["matching"]).input_ids
    n_ids = tokenizer(ex["not_matching"]).input_ids

    print(f"Prompt token count : {len(p_ids)}")
    print(f"Matching tokens    : {m_ids} = {[tokenizer.decode([t]) for t in m_ids]}")
    print(f"Not-matching tokens: {n_ids} = {[tokenizer.decode([t]) for t in n_ids]}")

    # find where the two answer sequences diverge
    div_idx = next((i for i,(a,b) in enumerate(zip(m_ids, n_ids)) if a != b), None)
    if div_idx is None:
        print("ERROR: matching and not_matching tokenize identically!")
        return
    print(f"Diverge at answer token index {div_idx}: "
          f"{tokenizer.decode([m_ids[div_idx]])} vs {tokenizer.decode([n_ids[div_idx]])}")

    # position_offset = div_idx + 1
    # (+1 because offset=1 means first answer token, offset=2 means second, etc.)
    best_offset = div_idx + 1
    print(f"=> best position_offset: {best_offset}")

    # ---- Step 2: load model ----
    print("\n" + "=" * 60)
    print("STEP 2: Loading model")
    print("=" * 60)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    n_layers, d_model = get_model_config(model)
    print(f"n_layers={n_layers}, d_model={d_model}")

    # precompute prompt lengths (no padding, raw tokenization)
    prompt_lens = [len(tokenizer(p).input_ids) for p in prompts]

    matching_texts  = [p + a for p, a in zip(prompts, matching)]
    not_mtch_texts  = [p + a for p, a in zip(prompts, not_mtch)]

    # ---- Step 3: layer sweep ----
    print("\n" + "=" * 60)
    print("STEP 3: Layer sweep")
    print("=" * 60)
    print(f"{'Layer':>6}  {'Offset':>7}  {'CV Acc':>8}  {'±':>6}  {'Contrast norm':>14}")
    print("-" * 58)

    best = {"layer": -1, "offset": -1, "acc": 0.0}

    # sweep a representative set of layers
    layer_candidates = sorted(set([
        0, n_layers//4, n_layers//2,
        int(n_layers*0.6), int(n_layers*0.7),
        int(n_layers*0.8), int(n_layers*0.9),
        n_layers - 2, n_layers - 1,
    ]))

    offsets_to_try = list(dict.fromkeys([best_offset, best_offset + 1, best_offset - 1]))
    offsets_to_try = [o for o in offsets_to_try if o >= 1]

    for layer_idx in layer_candidates:
        if layer_idx > n_layers:
            continue
        for offset in offsets_to_try:
            pos_acts = extract_at_position(
                model, tokenizer, matching_texts, prompt_lens,
                offset, layer_idx, args.device)
            neg_acts = extract_at_position(
                model, tokenizer, not_mtch_texts, prompt_lens,
                offset, layer_idx, args.device)

            diff      = pos_acts.mean(0) - neg_acts.mean(0)
            diff_norm = np.linalg.norm(diff)

            X = np.concatenate([pos_acts, neg_acts])
            y = np.array([1]*len(pos_acts) + [0]*len(neg_acts))

            acc, std = cv_accuracy(X, y)
            print(f"{layer_idx:>6}  {offset:>7}  {acc:>8.3f}  {std:>6.3f}  {diff_norm:>14.4f}")

            if acc > best["acc"]:
                best = {"layer": layer_idx, "offset": offset, "acc": acc}

    print(f"\nBest: layer={best['layer']}, offset={best['offset']}, acc={best['acc']:.3f}")

    # ---- Step 4: logit lens on best ----
    if best["acc"] > 0.55:
        print("\n" + "=" * 60)
        print("STEP 4: Logit lens on best direction")
        print("=" * 60)
        pos_acts = extract_at_position(
            model, tokenizer, matching_texts, prompt_lens,
            best["offset"], best["layer"], args.device)
        neg_acts = extract_at_position(
            model, tokenizer, not_mtch_texts, prompt_lens,
            best["offset"], best["layer"], args.device)

        diff = torch.tensor(pos_acts.mean(0) - neg_acts.mean(0), dtype=torch.float32)
        diff = diff / diff.norm()

        W_U = get_lm_head(model)
        proj = W_U @ diff
        top_ids = proj.topk(15).indices.tolist()
        print("Top tokens in contrastive direction:")
        for tid in top_ids:
            print(f"  {repr(tokenizer.decode([tid]))}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if best["acc"] >= 0.65:
        print(f"✓ Found linear signal at layer={best['layer']}, offset={best['offset']}")
        print(f"  Retrain probe with --layer {best['layer']}")
    elif best["acc"] >= 0.55:
        print(f"~ Weak signal at layer={best['layer']} (acc={best['acc']:.3f})")
        print("  Try --n_items 200+ for more data, or a larger model.")
    else:
        print("✗ No linear signal found.")
        print("  Sycophancy in this model may be non-linearly encoded.")
        print("  Consider:")
        print("    1. Using logprob-based sycophancy score directly (no probe)")
        print("    2. Nonlinear probe (MLP classifier)")
        print("    3. Larger model")


if __name__ == "__main__":
    main()