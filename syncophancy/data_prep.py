"""
Phase 2 — data prep for sycophancy measurement.

Loads the Anthropic model-written sycophancy evals (Perez et al. 2022),
keeps clean binary A/B items, and emits two DISJOINT files:
  - eval.jsonl     : balanced held-out set for measuring sycophancy rate
  - dpo_pool.jsonl : remaining items, source for Phase-3 DPO preference pairs

Each output record:
  {
    "prompt": "<persona + opinion + choices + 'Answer:'>",
    "matching": " (A)",       # sycophantic answer (agrees with the user)
    "not_matching": " (B)",   # non-sycophantic answer
    "subset": "nlp" | "pol"
  }

Note on the metric this supports: these items measure OPINION sycophancy
(matching the user's stated political / field view), not factual
correct->incorrect flipping. That is the standard testbed used by CAA /
steering work, and it is cleanly bidirectional for DPO. If you later want
factual-flip sycophancy you need a different dataset (can be added).
"""
import argparse, json, random, urllib.request
from pathlib import Path
from collections import defaultdict

SUBSETS = {
    "nlp": "syco_nlp.jsonl",
    "pol": "syco_pol.jsonl",
    "phil": "syco_phil.jsonl",  # available but excluded by default (multi-option)
}

# Raw source (GitHub mirror of Anthropic/model-written-evals sycophancy subset).
RAW_URLS = {
    "nlp": "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_nlp_survey.jsonl",
    "pol": "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    "phil": "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_philpapers2020.jsonl",
}


def ensure_raw(data_dir: Path, subsets):
    """Download raw subset files if missing, so a clean checkout just works."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for s in subsets:
        dest = data_dir / SUBSETS[s]
        if dest.exists():
            continue
        print(f"downloading {s} -> {dest.name}")
        urllib.request.urlretrieve(RAW_URLS[s], dest)


def load_subset(data_dir: Path, name: str):
    items = []
    for line in open(data_dir / SUBSETS[name]):
        r = json.loads(line)
        nm = r["answer_not_matching_behavior"]
        nm = nm if isinstance(nm, list) else [nm]
        # clean binary only: exactly one non-matching option
        if len(nm) != 1:
            continue
        items.append({
            "prompt": r["question"],
            "matching": r["answer_matching_behavior"],
            "not_matching": nm[0],
            "subset": name,
        })
    return items


def question_part(prompt):
    """Grouping key = prompt with the persona bio stripped (same key as
    make_dpo_pairs.py). Bios are first-person; the question is not. Key =
    everything after the LAST sentence containing a first-person token."""
    import re
    head_end = prompt.find("\n (A)")
    if head_end == -1:
        head_end = len(prompt)
    fp = re.compile(r"\b(I|I'm|I've|my|My|me|mine)\b")
    cut = 0
    for m in re.finditer(r"[.!?]\s+", prompt[:head_end]):
        if fp.search(prompt[cut:m.start() + 1]):
            cut = m.end()
    return prompt[cut:]


def balanced_sample(items, n, seed):
    """Sample n items balanced by which letter is the *matching* answer,
    so the eval can't be gamed by a constant-letter model."""
    rng = random.Random(seed)
    by_letter = defaultdict(list)
    for it in items:
        by_letter[it["matching"].strip()].append(it)
    for v in by_letter.values():
        rng.shuffle(v)
    out, i = [], 0
    letters = sorted(by_letter)
    while len(out) < n and any(by_letter[l][i:] for l in letters):
        for l in letters:
            if len(out) < n and i < len(by_letter[l]):
                out.append(by_letter[l][i])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(Path(__file__).resolve().parent/"data"))
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent/"data"))
    ap.add_argument("--subsets", nargs="+", default=["nlp", "pol"])
    ap.add_argument("--n_eval", type=int, default=150)
    ap.add_argument("--n_dpo", type=int, default=2000)
    ap.add_argument("--split", default="item", choices=["item", "question"],
                    help="item: original per-item split (keeps existing eval.jsonl "
                         "comparable). question: eval and pool share NO questions — "
                         "use for leakage-proof runs; re-measure baselines after switching.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    ensure_raw(data_dir, args.subsets)
    rng = random.Random(args.seed)

    pool = []
    for s in args.subsets:
        sub = load_subset(data_dir, s)
        rng.shuffle(sub)
        pool.append((s, sub))

    # eval: balanced per subset, disjoint from dpo
    eval_items, used = [], set()
    per = args.n_eval // len(pool)
    for s, sub in pool:
        picked = balanced_sample(sub, per, args.seed)
        for it in picked:
            used.add(it["prompt"])
        eval_items.extend(picked)

    # dpo pool: everything not used in eval.
    # item split: exclude only the exact eval prompts (original behaviour).
    # question split: exclude every item sharing a QUESTION with any eval item,
    # so train/eval are question-disjoint (no stance-memorisation leakage).
    if args.split == "question":
        used_q = {question_part(p) for p in used}
        excluded = lambda it: question_part(it["prompt"]) in used_q
    else:
        excluded = lambda it: it["prompt"] in used
    dpo_items = []
    for s, sub in pool:
        for it in sub:
            if not excluded(it):
                dpo_items.append(it)
    rng.shuffle(dpo_items)
    dpo_items = dpo_items[: args.n_dpo]

    with open(out_dir / "eval.jsonl", "w") as f:
        for it in eval_items:
            f.write(json.dumps(it) + "\n")
    with open(out_dir / "dpo_pool.jsonl", "w") as f:
        for it in dpo_items:
            f.write(json.dumps(it) + "\n")

    # report
    def letter_balance(items):
        c = defaultdict(int)
        for it in items:
            c[it["matching"].strip()] += 1
        return dict(c)

    print(f"eval.jsonl     : {len(eval_items)} items  matching-letter balance {letter_balance(eval_items)}")
    print(f"dpo_pool.jsonl : {len(dpo_items)} items  matching-letter balance {letter_balance(dpo_items)}")
    print(f"disjoint check : {len(set(i['prompt'] for i in eval_items) & set(i['prompt'] for i in dpo_items))} overlap (want 0)")
    q_overlap = len({question_part(i["prompt"]) for i in eval_items} &
                    {question_part(i["prompt"]) for i in dpo_items})
    print(f"question overlap eval<->pool: {q_overlap} "
          f"({'want 0' if args.split == 'question' else 'expected >0 in item split; use --split question for 0'})")


if __name__ == "__main__":
    main()