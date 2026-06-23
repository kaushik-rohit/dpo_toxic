# sycophancy/
 
Extension of `dpo_toxic` to sycophancy, mirroring the `toxicity/` layout so it
reuses the repo's root-level `utils.py` / `constants.py` and logit-lens code.
 
```
sycophancy/
  data_prep.py          # build eval.jsonl + dpo_pool.jsonl from the raw evals
  metric.py             # answer-logprob sycophancy scorer + bootstrap CI (+ mock)
  run_baseline.py       # Phase 2: measure baseline sycophancy on a model
  load_from_hf.py       # canonical HF loader (targets the sycophancy/ subset)
  data/
    eval.jsonl          # 150 balanced held-out items
    dpo_pool.jsonl      # 2000 disjoint items -> Phase-3 preference pairs
  train_dpo/            # Phase 3 (two DPO arms)
  eval_interventions/   # Phase 4 (reversion / grain measurement)
```
 
All scripts resolve `data/` relative to their own location, so they run from
anywhere (repo root or inside sycophancy/).
 
## Phase 2 (done)
```
python sycophancy/run_baseline.py --model gpt2-medium
python sycophancy/run_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
python sycophancy/run_baseline.py --mock        # torch-free pipeline check
```
Gate: rate CI lower-bound > ~0.55 => measurable => proceed. Flat on GPT2-medium
=> escalate to Qwen2.5-1.5B (and use the TRL path in train_dpo).
 
See the project execution plan for the full phase breakdown and decision gates.
