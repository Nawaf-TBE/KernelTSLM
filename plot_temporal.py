# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Figure of runs/temporal_baselines/metrics.json -> runs/temporal_baselines/order_matters.png

  uv run plot_temporal.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/temporal_baselines"
MODELS = {"majority": "Majority", "bin_only": "This Bin\nonly", "window_shuffled": "+ neighbors,\nshuffled",
          "window": "+ neighbors,\ntrue order", "cnn": "CNN\n(temporal)"}
INK, MUTED, GRID, EDGE = "#222", "#666", "#e8e8e6", "#bbb"
GREY, LIGHT_GREY, BLUE, LIGHT_BLUE = "#9a9a94", "#cfcfca", "#2a78d6", "#86b6ef"
ORDERED = {"window", "cnn"}


def main():
    m = json.loads((RUN / "metrics.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.5, 1], "wspace": .3},
                               facecolor="white")
    xs, w = np.arange(len(MODELS)), .38
    for k, subset in enumerate(("all", "multi_workload")):
        for x, model in zip(xs + (k - .5) * (w + .03), MODELS):
            r = m[subset][model]
            lo, hi = r["interval_95"]["macro_f1"]
            color = (BLUE if k == 0 else LIGHT_BLUE) if model in ORDERED else (GREY if k == 0 else LIGHT_GREY)
            a.bar(x, r["macro_f1"], w, color=color, edgecolor="white", linewidth=2)
            a.errorbar(x, r["macro_f1"], yerr=[[r["macro_f1"] - lo], [hi - r["macro_f1"]]], color=INK, lw=1, capsize=3)
            a.text(x, hi + .015, f"{r['macro_f1']:.2f}", ha="center", fontsize=9, color=INK)
    a.set_xticks(xs, list(MODELS.values()), fontsize=10, color=INK)
    a.set_ylim(0, 1)
    a.set_ylabel("macro-F1, workload of each Bin", color=INK)
    n_all, n_multi = m["all"]["window"]["snapshots"], m["multi_workload"]["window"]["snapshots"]
    a.text(0, 1.02, f"Dark: all test Snapshots ({n_all})   Light: multi-workload only ({n_multi})   "
           "Whiskers: 95% interval", transform=a.transAxes, fontsize=9, color=MUTED)

    o = m["multi_workload"]["window"]["per_snapshot_accuracy"]
    s = m["multi_workload"]["window_shuffled"]["per_snapshot_accuracy"]
    keys = sorted(o, key=lambda k: o[k] - s[k])
    for r, k in enumerate(keys):
        b.plot([s[k], o[k]], [r, r], color=GRID, lw=2, zorder=1)
    b.scatter([s[k] for k in keys], range(len(keys)), s=60, color=GREY, zorder=2, edgecolor="white",
              linewidth=1.5, label="neighbors shuffled")
    b.scatter([o[k] for k in keys], range(len(keys)), s=60, color=BLUE, zorder=3, edgecolor="white",
              linewidth=1.5, label="neighbors in true order")
    wins = sum(o[k] > s[k] for k in keys)
    b.set_yticks([])
    b.set_ylim(-.8, len(keys) - .2)
    b.set_xlim(0, 1)
    b.set_xlabel("Bins labeled correctly (mean of 5 seeds)", color=INK)
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    b.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2)
    b.text(0, 1.02, f"Each row: one multi-workload test Snapshot. Order wins in {wins} of {len(keys)}.",
           transform=b.transAxes, fontsize=9, color=MUTED)

    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(EDGE)
        ax.tick_params(colors=MUTED)
    a.grid(axis="y", color=GRID)
    a.set_axisbelow(True)
    fig.suptitle("The same neighbor readings help only in their true order", x=.06, ha="left", fontsize=14,
                 weight="bold", color=INK, y=1.04)
    fig.text(.06, .955, f"Name the workload of each Bin from that Bin and the {m['setup']['window_bins_each_side']} "
             "Bins on each side. Grouped split: test Problems are unseen in training. 5 seeds.",
             fontsize=10, color=MUTED)
    fig.savefig(RUN / "order_matters.png", dpi=170, bbox_inches="tight")
    print("saved", RUN / "order_matters.png")


if __name__ == "__main__":
    main()
