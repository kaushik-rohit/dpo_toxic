"""
SAE-feature analysis of the sycophancy edit (Gemma 2 2B + Gemma Scope).

For base / pro / anti, encode the residual stream at a layer with a Gemma Scope
SAE, aggregate to one feature vector per model (mean over the answer position,
across eval prompts), then:
  install[f]  = pro[f]  - base[f]   (features pro turns on)
  suppress[f] = anti[f] - base[f]   (features anti turns off)
  route_cosine in SAE-feature space (same features both ways?)
Exports features.json for the interactive explorer (schema below).

Prereqs (your box): pip install sae-lens transformers peft torch
You first need the two Gemma DPO arms (rerun Phase 3 with --model google/gemma-2-2b-it).

  python sae_features.py --layer 12 --pro_adapter ../checkpoints/pro \
      --anti_adapter ../checkpoints/anti --out features.json

Note: confirm the SAE release/id against `sae_lens` and that the layer index
matches the SAE's training hook (resid_post). Neuronpedia labels are fetched if
--labels is set (needs network).
"""
import argparse, json, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def mean_resid(model, tok, prompts, layer, device):
    import torch
    acc = None
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").to(device)
            hs = model(**ids, output_hidden_states=True).hidden_states
            v = hs[layer][0, -1].float()        # resid at chosen layer, last token
            acc = v if acc is None else acc + v
    return (acc / len(prompts))


def neuronpedia_label(model_id, sae_id, fid):
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{fid}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            d = json.load(r)
        exps = d.get("explanations", [])
        return exps[0].get("description", "") if exps else ""
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--pro_adapter", required=True)
    ap.add_argument("--anti_adapter", required=True)
    ap.add_argument("--eval", default=str(ROOT / "data" / "eval.jsonl"))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sae_release", default="gemma-scope-2b-pt-res-canonical")
    ap.add_argument("--sae_id", default="layer_12/width_16k/canonical")
    ap.add_argument("--np_model", default="gemma-2-2b", help="Neuronpedia model id")
    ap.add_argument("--np_sae", default="12-gemmascope-res-16k", help="Neuronpedia sae id")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--out", default="features.json")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from sae_lens import SAE
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [json.loads(l)["prompt"] for l in open(args.eval)][: args.n]
    sae = SAE.from_pretrained(args.sae_release, args.sae_id, device="cuda")
    sae = sae[0] if isinstance(sae, tuple) else sae

    def feat_vec(adapter=None):
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to("cuda").eval()
        if adapter:
            m = PeftModel.from_pretrained(m, adapter).to("cuda").eval()
        resid = mean_resid(m, tok, prompts, args.layer, "cuda")
        with torch.no_grad():
            f = sae.encode(resid.to(sae.dtype)).float().cpu().numpy()  # (n_features,)
        del m; torch.cuda.empty_cache()
        return f

    base, pro, anti = feat_vec(), feat_vec(args.pro_adapter), feat_vec(args.anti_adapter)
    install, suppress = pro - base, anti - base
    cos = float(np.dot(install, suppress) /
                (np.linalg.norm(install) * np.linalg.norm(suppress) + 1e-9))

    # rank by combined movement; classify bidirectional
    score = np.maximum(np.abs(install), np.abs(suppress))
    idx = np.argsort(-score)[: args.topk]
    feats = []
    for f in idx:
        bidir = bool(np.sign(install[f]) != np.sign(suppress[f])
                     and abs(install[f]) > 0 and abs(suppress[f]) > 0)
        feats.append({
            "id": int(f),
            "label": neuronpedia_label(args.np_model, args.np_sae, int(f)) if args.labels else "",
            "base": float(base[f]), "pro": float(pro[f]), "anti": float(anti[f]),
            "install": float(install[f]), "suppress": float(suppress[f]),
            "bidirectional": bidir,
            "neuronpedia": f"https://www.neuronpedia.org/{args.np_model}/{args.np_sae}/{int(f)}",
        })

    out = {
        "model": args.model, "layer": args.layer,
        "sae": f"{args.sae_release}/{args.sae_id}",
        "route_cosine": cos,
        "top_install_share": float(np.abs(install).max() / (np.abs(install).sum() + 1e-9)),
        "features": feats,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"route_cosine(install, suppress) = {cos:+.3f}")
    print(f"wrote features.json ({len(feats)} features)")


if __name__ == "__main__":
    main()