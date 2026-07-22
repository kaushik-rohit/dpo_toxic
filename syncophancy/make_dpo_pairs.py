"""
Build mirror DPO preference pairs.

Mirror structure (same prompt, chosen/rejected swapped) is kept in every mode so
DIRECTION is the only variable between arms:
  pro.jsonl  -> installs sycophancy (chosen = the agreeing/matching response)
  anti.jsonl -> removes it (exact mirror)

Modes (completion richness):
  --mode letter     bare "(A)"/"(B)"  — confound-free, thin signal (default)
  --mode rationale  short stance responses, LENGTH-MATCHED via parallel templates
                    and template-VARIED so the model learns the disposition, not a
                    phrase. Engages agreement features (better for SAE analysis),
                    still zero length confound.

Note: the metric scores A/B logprobs. Training on --mode rationale while measuring
on A/B is a train-rich / measure-on-held-out-format design: it tests whether a
genuine disposition was installed, not a format trick.
"""
import argparse, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# Parallel templates: index i pairs an AGREE and a DISAGREE response of matched
# length/structure. The {ans} slot carries the option letter so the surface form
# still anchors to the choice, while the stance words drive the sycophancy signal.
TEMPLATES = [
    ("Yes, I agree with you. {ans}",      "No, I disagree with you. {ans}"),
    ("You're right, I think so too. {ans}","You're mistaken, I don't think so. {ans}"),
    ("I agree — that view is correct. {ans}", "I disagree — that view is wrong. {ans}"),
    ("That's a fair point, I concur. {ans}",  "That's not right, I object. {ans}"),
    ("Absolutely, you've got it. {ans}",      "Actually no, you haven't. {ans}"),
]

# raw subset files (as written by data_prep.ensure_raw)
RAW_FILES = {"nlp": "syco_nlp.jsonl", "pol": "syco_pol.jsonl", "phil": "syco_phil.jsonl"}

def question_part(prompt):
    """Grouping key = prompt with the persona bio stripped.
    Bios are first-person ('Hello, my name is...', 'I am...'); the question is
    not. Key = everything after the LAST sentence containing a first-person
    token. Only key consistency matters for grouping identical questions."""
    import re
    head_end = prompt.find("\n (A)")
    if head_end == -1:
        head_end = len(prompt)
    fp = re.compile(r"\b(I|I'm|I've|my|My|me|mine)\b")
    cut = 0
    for m in re.finditer(r"[.!?]\s+", prompt[:head_end]):
        if fp.search(prompt[cut:m.start() + 1]):
            cut = m.end()
        # sentences without first-person leave `cut` where it is: the question
        # may span several sentences (claim text etc.)
    return prompt[cut:]


def build_consistency(data_dir: Path, out_dir: Path, eval_path: Path,
                      rng, per_question_cap=30, n_max=2000, holdout_k=10,
                      eval_per_question=15):
    """Persona-invariance pairs from the RAW data.

    For each question that appears under personas with OPPOSITE views, fix one
    stance S (balanced across questions), and for every persona-prompt make:
        chosen = S,  rejected = the other option.
    Half the pairs then reward agreeing with the user's persona and half reward
    disagreeing (balanced within each question), so the net gradient on
    'agree with the user' is ~zero; the trained disposition is 'answer the
    question the same way regardless of the persona' -> optimum is 0.5 on the
    persona-sensitivity metric, not 0.0 (no contrarian solution).

    LEAKAGE NOTE: train/eval must be disjoint at the QUESTION level, not the
    item level — otherwise the model can memorise the fixed stance per question
    and fake persona-invariance on eval. Since the original eval.jsonl was
    sampled per-item (its questions overlap nearly all multi-persona groups),
    this mode carves its OWN question-disjoint eval: `holdout_k` questions ->
    consistency_eval.jsonl; the rest -> consistency.jsonl.
    """
    import itertools
    from collections import defaultdict

    # load all raw items (same cleaning as data_prep.load_subset)
    items = []
    for name, fn in RAW_FILES.items():
        f = data_dir / fn
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            nm = r["answer_not_matching_behavior"]
            nm = nm if isinstance(nm, list) else [nm]
            if len(nm) != 1:
                continue
            items.append({"prompt": r["question"], "matching": r["answer_matching_behavior"],
                          "not_matching": nm[0]})
    if not items:
        raise SystemExit(f"no raw files found in {data_dir} — run data_prep.py first")

    groups = defaultdict(list)
    for it in items:
        groups[question_part(it["prompt"])].append(it)

    usable = {q: g for q, g in groups.items()
              if len({it["matching"].strip() for it in g}) > 1}
    if len(usable) <= holdout_k:
        raise SystemExit(f"only {len(usable)} opposite-view questions found; "
                         f"need > holdout_k={holdout_k}")

    # question-disjoint split: hold out k questions for eval, deterministic.
    # sort by group size so both splits get a mix of large/small questions.
    ordered = sorted(usable, key=lambda q: (-len(usable[q]), q))
    step = max(2, len(ordered) // holdout_k)
    eval_qs = set(ordered[1::step][:holdout_k])
    train_qs = [q for q in ordered if q not in eval_qs]

    # eval file: balanced items from held-out questions, same schema as eval.jsonl
    eval_items = []
    for q in sorted(eval_qs):
        g = list(usable[q]); rng.shuffle(g)
        a = [it for it in g if it["matching"].strip().endswith("(A)")]
        b = [it for it in g if not it["matching"].strip().endswith("(A)")]
        k = min(len(a), len(b), eval_per_question // 2)
        eval_items += a[:k] + b[:k]
    rng.shuffle(eval_items)
    with open(out_dir / "consistency_eval.jsonl", "w") as f:
        for it in eval_items:
            f.write(json.dumps(it) + "\n")

    pairs, agree_n, disagree_n = [], 0, 0
    stance_flip = itertools.cycle([0, 1])          # balance stance letter across questions
    for q in train_qs:
        g = usable[q]
        letters = sorted({it["matching"].strip() for it in g})
        S = letters[next(stance_flip)]              # fixed stance for this question
        agree_side = [it for it in g if it["matching"].strip() == S]
        dis_side = [it for it in g if it["matching"].strip() != S]
        k = min(len(agree_side), len(dis_side), per_question_cap // 2)
        rng.shuffle(agree_side); rng.shuffle(dis_side)
        for it in agree_side[:k] + dis_side[:k]:
            m, nm = it["matching"], it["not_matching"]
            chosen, rejected = (m, nm) if it["matching"].strip() == S else (nm, m)
            pairs.append({"prompt": it["prompt"], "chosen": chosen, "rejected": rejected})
            if it["matching"].strip() == S:
                agree_n += 1
            else:
                disagree_n += 1
    rng.shuffle(pairs)
    pairs = pairs[:n_max]
    with open(out_dir / "consistency.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"mode=consistency  wrote {len(pairs)} pairs -> consistency.jsonl")
    print(f"train questions: {len(train_qs)}   eval questions (disjoint): {len(eval_qs)}")
    print(f"eval items -> consistency_eval.jsonl: {len(eval_items)} (question-disjoint from training)")
    print(f"agree-rewarded {agree_n} / disagree-rewarded {disagree_n} before cap")
    print("objective optimum = persona-indifference (0.5), not inversion (0.0)")
    print("NOTE: measure the consistency arm on consistency_eval.jsonl (pass --eval), "
          "not the original eval.jsonl (question overlap would let memorised stances "
          "fake persona-invariance).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DATA / "dpo_pool.jsonl"))
    ap.add_argument("--out_dir", default=str(DATA))
    ap.add_argument("--mode", default="letter", choices=["letter", "rationale", "consistency"])
    ap.add_argument("--per_question_cap", type=int, default=30,
                    help="consistency mode: max pairs per question (avoids one survey question dominating)")
    ap.add_argument("--n_max", type=int, default=2000,
                    help="consistency mode: total pair cap")
    ap.add_argument("--holdout_k", type=int, default=10,
                    help="consistency mode: questions held out for consistency_eval.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)

    if args.mode == "consistency":
        data_dir = Path(args.pool).parent      # raw files live next to the pool
        build_consistency(data_dir, out, data_dir / "eval.jsonl", rng,
                          per_question_cap=args.per_question_cap, n_max=args.n_max,
                          holdout_k=args.holdout_k)
        return

    pool = [json.loads(l) for l in open(args.pool)]

    def completions(r):
        """returns (agree_completion, disagree_completion) — agree = sycophantic."""
        m, nm = r["matching"], r["not_matching"]   # e.g. " (A)", " (B)"
        if args.mode == "letter":
            return m, nm
        agree_t, dis_t = TEMPLATES[rng.randrange(len(TEMPLATES))]
        return " " + agree_t.format(ans=m.strip()), " " + dis_t.format(ans=nm.strip())

    n = 0
    with open(out / "pro.jsonl", "w") as fp, open(out / "anti.jsonl", "w") as fa:
        for r in pool:
            agree, disagree = completions(r)
            # pro: prefer agreeing (sycophantic) response
            fp.write(json.dumps({"prompt": r["prompt"], "chosen": agree, "rejected": disagree}) + "\n")
            # anti: exact mirror
            fa.write(json.dumps({"prompt": r["prompt"], "chosen": disagree, "rejected": agree}) + "\n")
            n += 1

    pro = [json.loads(l) for l in open(out / "pro.jsonl")]
    anti = [json.loads(l) for l in open(out / "anti.jsonl")]
    ok = all(p["chosen"] == a["rejected"] and p["rejected"] == a["chosen"]
             and p["prompt"] == a["prompt"] for p, a in zip(pro, anti))
    # length-match check (mean token-ish length of chosen vs rejected in pro)
    import statistics as st
    lc = st.mean(len(p["chosen"].split()) for p in pro)
    lr = st.mean(len(p["rejected"].split()) for p in pro)
    print(f"mode={args.mode}  wrote {n} pairs each -> pro.jsonl, anti.jsonl")
    print(f"exact-mirror check: {'PASS' if ok else 'FAIL'}")
    print(f"length match (chosen vs rejected words): {lc:.2f} vs {lr:.2f}  "
          f"(Δ={abs(lc-lr):.2f} — keep small to avoid length confound)")


if __name__ == "__main__":
    main()