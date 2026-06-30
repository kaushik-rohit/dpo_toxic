"""
Visualize the pro/anti trajectories. Three views, each tied to an RQ.

Designed to consume data you extract on your GPU box:
  - behavioral:  metric-vs-step logs from training (dict of step->rate per arm)
  - rep_space:   per-prompt activations at the sycophancy layer for base/pro/anti
                 plus the sycophancy direction (CAA difference-of-means)
  - reversion:   (steering_magnitude, fraction_of_delta_restored) per arm

Nothing here needs a GPU — it takes arrays/dicts and renders PNGs. See
demo_illustrative.py for the expected shapes (it feeds synthetic data through
these same functions).
"""
import numpy as np
import matplotlib.pyplot as plt


# ---------- View 1: behavioral trajectory (arms diverge) ----------
def plot_behavioral(steps, pro_rate, anti_rate, baseline, ax=None,
                    pro_ci=None, anti_ci=None):
    ax = ax or plt.gca()
    ax.axhline(baseline, ls="--", c="0.5", lw=1, label="baseline")
    if pro_ci is not None:
        pro_ci = np.asarray(pro_ci)
        ax.fill_between(steps, pro_ci[:, 0], pro_ci[:, 1], color="#c0392b", alpha=0.15, lw=0)
    if anti_ci is not None:
        anti_ci = np.asarray(anti_ci)
        ax.fill_between(steps, anti_ci[:, 0], anti_ci[:, 1], color="#2471a3", alpha=0.15, lw=0)
    ax.plot(steps, pro_rate, "-o", ms=3, c="#c0392b", label="pro (install)")
    ax.plot(steps, anti_rate, "-o", ms=3, c="#2471a3", label="anti (remove)")
    ax.set_xlabel("DPO step"); ax.set_ylabel("persona-sensitivity")
    ax.set_title("Behavioral trajectory")
    ax.legend(fontsize=8, frameon=False)


# ---------- View 2: representation-space trajectory (same route?) ----------
def plot_rep_space(base_pts, pro_pts, anti_pts, ax=None):
    """*_pts: (N,2) arrays already projected onto [sycophancy_dir, orthogonal].
    Build these by projecting per-prompt activations; pass centroids too."""
    ax = ax or plt.gca()
    for pts, c, lab in [(base_pts, "0.6", "base"),
                        (pro_pts, "#c0392b", "pro"),
                        (anti_pts, "#2471a3", "anti")]:
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.35, color=c)
        ax.scatter(*pts.mean(0), s=90, color=c, edgecolor="k", zorder=5, label=lab)
    b, p, a = base_pts.mean(0), pro_pts.mean(0), anti_pts.mean(0)
    ax.annotate("", p, b, arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2))
    ax.annotate("", a, b, arrowprops=dict(arrowstyle="-|>", color="#2471a3", lw=2))
    cos = np.dot(p - b, a - b) / (np.linalg.norm(p - b) * np.linalg.norm(a - b) + 1e-9)
    ax.text(0.03, 0.97, f"cos(install, remove) = {cos:+.2f}\n"
            f"{'≈ same route, opposite sign' if cos < -0.6 else 'distinct routes'}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_xlabel("sycophancy direction"); ax.set_ylabel("orthogonal component")
    ax.set_title("Representation-space trajectory (RQ1)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")


# ---------- View 3: reversion-cost trajectory (grain asymmetry) ----------
def plot_reversion(mag, withgrain_restored, against_restored, ax=None):
    ax = ax or plt.gca()
    ax.axhline(0.5, ls=":", c="0.5", lw=1)
    ax.plot(mag, withgrain_restored, "-", c="#27ae60", lw=2,
            label="with-grain (restore sycophancy)")
    ax.plot(mag, against_restored, "-", c="#8e44ad", lw=2,
            label="against-grain (re-remove)")
    # mark 50%-restore cost for each, where it crosses
    for y, c in [(withgrain_restored, "#27ae60"), (against_restored, "#8e44ad")]:
        idx = np.argmax(np.asarray(y) >= 0.5)
        if y[idx] >= 0.5:
            ax.plot([mag[idx]], [0.5], "v", color=c, ms=9)
    ax.set_xlabel("reversion magnitude (steering coef or fine-tune steps)")
    ax.set_ylabel("fraction of Δ restored")
    ax.set_title("Reversion cost (RQ2)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_ylim(0, 1.05)


def figure(out_path, **panels):
    """panels: dict with optional keys 'behavioral','rep_space','reversion',
    each a dict of kwargs for the matching plot_* function."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.3))
    axes = np.atleast_1d(axes)
    fns = {"behavioral": plot_behavioral, "rep_space": plot_rep_space,
           "reversion": plot_reversion}
    for ax, (name, kw) in zip(axes, panels.items()):
        fns[name](ax=ax, **kw)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print("saved", out_path)