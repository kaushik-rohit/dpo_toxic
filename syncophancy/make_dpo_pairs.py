"""
Phase 3 — build mirror DPO preference pairs.

From data/dpo_pool.jsonl produces two TRL-format files that are EXACT mirrors,
so the only thing differing between the two training arms is the preference
direction:

  pro.jsonl  : chosen = matching     (agree with user)  -> installs sycophancy
  anti.jsonl : chosen = not_matching (disagree)         -> removes sycophancy

TRL DPO format per line: {"prompt": ..., "chosen": ..., "rejected": ...}
where chosen/rejected are the *completions only* (the answer string, incl. its
leading space), appended to `prompt`.

Format note: completions are the bare "(A)"/"(B)" answers, matching how
metric.py scores. This keeps training and measurement in the SAME format so
the metric stays valid pre/post DPO. (If the anti-arm collapses or learns a
letter bias, switch to answer+rationale completions via --rationale later.)
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DATA / "dpo_pool.jsonl"))
    ap.add_argument("--out_dir", default=str(DATA))
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool)]
    out_dir = Path(args.out_dir)

    n = 0
    with open(out_dir / "pro.jsonl", "w") as fp, open(out_dir / "anti.jsonl", "w") as fa:
        for r in pool:
            prompt, m, nm = r["prompt"], r["matching"], r["not_matching"]
            # pro-sycophancy: prefer the user-agreeing answer
            fp.write(json.dumps({"prompt": prompt, "chosen": m, "rejected": nm}) + "\n")
            # anti-sycophancy: exact mirror
            fa.write(json.dumps({"prompt": prompt, "chosen": nm, "rejected": m}) + "\n")
            n += 1

    print(f"wrote {n} pairs each -> pro.jsonl, anti.jsonl")
    # sanity: mirrors must be exact swaps
    pro = [json.loads(l) for l in open(out_dir / "pro.jsonl")]
    anti = [json.loads(l) for l in open(out_dir / "anti.jsonl")]
    ok = all(p["chosen"] == a["rejected"] and p["rejected"] == a["chosen"]
             and p["prompt"] == a["prompt"] for p, a in zip(pro, anti))
    print("exact-mirror check:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()