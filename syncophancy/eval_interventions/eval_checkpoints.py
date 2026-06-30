"""
Evaluate intermediate DPO checkpoints -> training-path log.

After rerunning an arm with --save_steps, this scores each checkpoint-* adapter
on the held-out eval set and writes step->persona-sensitivity.

  python eval_checkpoints.py --ckpt_dir ../train_dpo/checkpoints/pro  --out path_pro.json
  python eval_checkpoints.py --ckpt_dir ../train_dpo/checkpoints/anti --out path_anti.json

Reuses metric.py (the same scorer used for the headroom gate).
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import metric.py
from metric import load_eval, evaluate, score_item_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ckpt_dir", required=True, help="arm output dir with checkpoint-* subdirs")
    ap.add_argument("--eval", default=str(Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    items = load_eval(args.eval)
    if args.limit:
        items = items[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)

    ckpts = sorted(Path(args.ckpt_dir).glob("checkpoint-*"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    print(f"found {len(ckpts)} checkpoints in {args.ckpt_dir}")

    log = []
    for ck in ckpts:
        step = int(re.findall(r"\d+", ck.name)[-1])
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to("cuda").eval()
        model = PeftModel.from_pretrained(base, str(ck)).to("cuda").eval()
        res = evaluate(items, lambda it: score_item_real(model, tok, it))
        log.append({"step": step, "rate": res["sycophancy_rate"],
                    "ci": res["rate_ci95"], "mean_p": res["mean_p_syco"]})
        print(f"  step {step:4d}  rate {res['sycophancy_rate']:.3f}")
        del model, base
        torch.cuda.empty_cache()

    json.dump(log, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()