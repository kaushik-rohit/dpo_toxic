"""
Plot the training path (behavioral trajectory) with CI bands.

  python eval_checkpoints.py --ckpt_dir ../checkpoints/pro  --eval ../data/eval.jsonl --out path_pro.json
  python eval_checkpoints.py --ckpt_dir ../checkpoints/anti --eval ../data/eval.jsonl --out path_anti.json
  python plot_training_path.py --pro path_pro.json --anti path_anti.json --baseline 0.73 \
      --baseline_lo 0.66 --baseline_hi 0.81

CI bands appear automatically if the path json files contain a "ci" field
(re-run eval_checkpoints.py after the update so the CIs are saved).
"""
import argparse, json
import numpy as np
from plot_trajectory import figure


def load(path):
    log = sorted(json.load(open(path)), key=lambda d: d["step"])
    steps = np.array([d["step"] for d in log])
    rate = np.array([d["rate"] for d in log])
    ci = np.array([d["ci"] for d in log]) if all("ci" in d for d in log) else None
    return steps, rate, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pro", required=True)
    ap.add_argument("--anti", required=True)
    ap.add_argument("--baseline", type=float, required=True)
    ap.add_argument("--baseline_lo", type=float, default=None)
    ap.add_argument("--baseline_hi", type=float, default=None)
    ap.add_argument("--out", default="training_path.png")
    args = ap.parse_args()

    sp, rp, cp = load(args.pro)
    sa, ra, ca = load(args.anti)

    b_lo = args.baseline_lo if args.baseline_lo is not None else args.baseline
    b_hi = args.baseline_hi if args.baseline_hi is not None else args.baseline
    steps = np.concatenate([[0], sp])

    kw = dict(
        steps=steps,
        pro_rate=np.concatenate([[args.baseline], rp]),
        anti_rate=np.concatenate([[args.baseline], ra]),
        baseline=args.baseline,
    )
    if cp is not None:
        kw["pro_ci"] = np.vstack([[b_lo, b_hi], cp])
    if ca is not None:
        kw["anti_ci"] = np.vstack([[b_lo, b_hi], ca])

    figure(args.out, behavioral=kw)


if __name__ == "__main__":
    main()