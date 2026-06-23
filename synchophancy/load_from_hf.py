"""
Canonical loader for the sycophancy subset from Hugging Face.

GOTCHA: `load_dataset("Anthropic/model-written-evals")` returns the DEFAULT
config (~3,252 rows) which is the model self-awareness / advanced-AI-risk
items — NOT sycophancy. The sycophancy data lives in the `sycophancy/`
folder and must be targeted explicitly via data_files (below).

This produces the same records as the GitHub mirror used by data_prep.py,
so it's interchangeable. Use whichever source you prefer; the prepared
eval.jsonl / dpo_pool.jsonl are already identical to this.

Requires: pip install datasets
(Note: this path needs network access to huggingface.co.)
"""
from datasets import load_dataset

SYCO_FILES = {
    "nlp":  "sycophancy/sycophancy_on_nlp_survey.jsonl",
    "pol":  "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    "phil": "sycophancy/sycophancy_on_philpapers2020.jsonl",
}


def load_sycophancy(subsets=("nlp", "pol")):
    """Returns a list of dicts with keys: prompt, matching, not_matching, subset.
    Mirrors data_prep.load_subset() output, clean-binary only."""
    data_files = {k: SYCO_FILES[k] for k in subsets}
    ds = load_dataset("Anthropic/model-written-evals", data_files=data_files)
    out = []
    for sub in subsets:
        for r in ds[sub]:
            nm = r["answer_not_matching_behavior"]
            nm = nm if isinstance(nm, list) else [nm]
            if len(nm) != 1:               # keep clean binary only
                continue
            out.append({
                "prompt": r["question"],
                "matching": r["answer_matching_behavior"],
                "not_matching": nm[0],
                "subset": sub,
            })
    return out


if __name__ == "__main__":
    items = load_sycophancy()
    print(f"loaded {len(items)} clean-binary sycophancy items from HF")
    print({k: sum(i['subset'] == k for i in items) for k in ('nlp', 'pol')})