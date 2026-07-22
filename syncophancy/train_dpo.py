"""
Phase 3 — train one sycophancy DPO arm (LoRA) with TRL.

Run BOTH arms with identical config; only --direction differs:
  python train_dpo.py --direction pro    # installs sycophancy
  python train_dpo.py --direction anti   # removes  sycophancy

With peft_config passed and no explicit ref_model, TRL uses the base model
(adapter disabled) as the implicit DPO reference — correct for LoRA DPO.

Format: prompt ends in "Answer:", completion is " (A)"/" (B)" — same format
metric.py scores, so the metric stays valid pre/post training. (Not using the
chat template, deliberately, for train/measure consistency.)

NOTE on TRL versions: DPOConfig arg names drift across releases (beta,
max_length, max_prompt_length, processing_class vs tokenizer). Written for
trl>=0.12; if your version errors on an arg, check `DPOConfig`/`DPOTrainer`
signatures and adjust. Requires: torch transformers trl peft datasets accelerate
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=["pro", "anti"])
    ap.add_argument("--data_file", default="",
                    help="explicit training jsonl (e.g. data/consistency.jsonl); "
                         "overrides the {direction}.jsonl default")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data_dir", default=str(DATA))
    ap.add_argument("--output_dir", default=str(HERE / "checkpoints"))
    # --- identical across arms (do not vary between pro/anti) ---
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_prompt_len", type=int, default=480)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_steps", type=int, default=0,
                    help=">0 dumps intermediate checkpoints for the training-path plot")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt)

    data_file = args.data_file or str(Path(args.data_dir) / f"{args.direction}.jsonl")
    ds = load_dataset("json", data_files=data_file, split="train")
    print(f"training on: {data_file}")

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    out = str(Path(args.output_dir) / args.direction)
    cfg = DPOConfig(
        output_dir=out,
        beta=args.beta,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        max_length=args.max_len,
        logging_steps=10,
        save_strategy=("steps" if args.save_steps > 0 else "no"),
        save_steps=(args.save_steps if args.save_steps > 0 else 500),
        seed=args.seed,
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"[{args.direction}] LoRA adapter saved -> {out}")


if __name__ == "__main__":
    main()