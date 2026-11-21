# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scikit-learn", "aeon==1.6.0", "torch==2.15.0", "matplotlib",
#                 "transformers==4.57.6", "safetensors==0.8.0", "huggingface_hub"]
# ///
"""Does time order matter? Per-Bin workload baselines on the grouped split.

The task: name the L1 workload class of each labeled Bin (about 39% of Bins have a label; the others are
unknown and are not scored). The Snapshot task in baselines.py does not need order (its shuffled-Bins
control loses at most 0.02 macro-F1); this one does.

  majority           the most common class of the training Bins
  bin_only           logistic regression on the Bin's own 24 counters
  window_shuffled    logistic regression on the Bin and its K neighbors on each side, the neighbors in a random
                     order (the Bin stays at the center): the same readings without their order
  window             the same window in true order
  cnn                Nawaf's kernel-xray L1TemporalCNN trained per Bin as in kernel-xray (masked cross-entropy on
                     labeled Bins), recipe of configs/phase3_cnn.json, best epoch by validation macro-F1

Logistic regression trains on a random quarter of the training Bins; the seed picks that quarter and the
shuffle. All models run 5 seeds. Scores are on all test Snapshots and on the 16 multi-workload ones (more
than one L1 class over time). Intervals: 95% bootstrap over test Snapshots (1,000 resamples) of the seed mean.

  uv run temporal_baselines.py    # ~15 min on an M-series Mac; writes runs/temporal_baselines/
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "kernel-xray/src")]
from kernel_xray.phase3.cnn import L1TemporalCNN  # noqa: E402

RELEASE = Path.home() / "TimeNet-kernelsight/kernelsight"
K, SEEDS, DRAWS = 8, range(5), 1000
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    data = dict(np.load(HERE / "kernelsight.npz"))
    x, train = data["x"].astype(np.float64), data["split"] == "train"
    mean, std = x[train].mean(axis=(0, 2)), x[train].std(axis=(0, 2))
    z = ((x - mean[:, None]) / np.where(std > 1e-12, std, 1.0)[:, None]).astype(np.float32)
    y = np.stack([np.load(RELEASE / p.replace("input/tensor_input.npz", "labels/labels.npz"))["workload_l1"]
                  for p in data["paths"].tolist()])
    multi = np.array([len(set(row[row >= 0].tolist())) > 1 for row in y])
    return data, z, y, multi


def features(z, y, idx, mode, rng, keep=1.0):
    """One row per labeled Bin of the Snapshots in idx: (features, labels, Snapshot index of each row)."""
    out, labels, owner = [], [], []
    for i in idx:
        bins = np.flatnonzero(y[i] >= 0)
        if keep < 1:
            bins = bins[rng.random(len(bins)) < keep]
        if len(bins) == 0:
            continue
        if mode == "bin_only":
            f = z[i][:, bins].T
        else:
            pad = np.pad(z[i], ((0, 0), (K, K)), mode="edge")
            w = np.stack([pad[:, b:b + 2 * K + 1] for b in bins])
            if mode == "window_shuffled":
                others = np.delete(np.arange(2 * K + 1), K)
                perm = np.stack([np.insert(others[rng.permutation(2 * K)], K, K) for _ in bins])
                w = np.take_along_axis(w, perm[:, None, :].repeat(w.shape[1], 1), 2)
            f = w.reshape(len(bins), -1)
        out.append(f), labels.append(y[i][bins]), owner.append(np.full(len(bins), i))
    return np.concatenate(out), np.concatenate(labels), np.concatenate(owner)


def logistic(data, z, y, mode, seed):
    rng = np.random.default_rng(seed)
    train, test = (np.flatnonzero(data["split"] == s) for s in ("train", "test"))
    xtr, ytr, _ = features(z, y, train, mode, rng, keep=0.25)
    xte, _, _ = features(z, y, test, mode, rng)
    return LogisticRegression(max_iter=500, C=0.1).fit(xtr, ytr).predict(xte)


def cnn(data, z, y, seed):
    split = data["split"]
    x, t = torch.tensor(z), torch.tensor(y, dtype=torch.long)
    rows = {s: torch.tensor(np.flatnonzero(split == s)) for s in ("train", "validation", "test")}
    torch.manual_seed(seed)
    model = L1TemporalCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    order = torch.Generator().manual_seed(seed)

    def predict(part):
        model.eval()
        with torch.no_grad():
            p = torch.cat([model(x[b].to(DEVICE)).argmax(1).cpu() for b in part.split(64)])
        return p[t[part] >= 0].numpy(), t[part][t[part] >= 0].numpy()

    best_f1, best_epoch, best_state, epoch = -1.0, 0, None, 0
    while epoch < 30 and epoch - best_epoch < 5:
        epoch += 1
        model.train()
        for b in rows["train"][torch.randperm(len(rows["train"]), generator=order)].split(16):
            logits, target = model(x[b].to(DEVICE)), t[b].to(DEVICE)
            mask = target >= 0
            if not mask.any():
                continue
            loss = F.cross_entropy(logits.permute(0, 2, 1)[mask], target[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f1 = macro_f1(*reversed(predict(rows["validation"])))
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  cnn seed {seed}: best epoch {best_epoch}, validation macro-F1 {best_f1:.3f}", flush=True)
    return predict(rows["test"])[0]


def macro_f1(truth, pred):
    """Macro-F1 over the classes in truth (as baselines.py scores Snapshots)."""
    f1 = []
    for c in np.unique(truth):
        tp = np.sum((pred == c) & (truth == c))
        f1.append(2 * tp / max(np.sum(pred == c) + np.sum(truth == c), 1))
    return float(np.mean(f1))


def summarize(truth, owner, runs, subset):
    """Mean/std over seeds of accuracy and macro-F1 on the Bins of the Snapshots in subset, and a bootstrap
    interval over those Snapshots of the seed means."""
    snaps = np.unique(owner[np.isin(owner, subset)])
    rows_of = {s: np.flatnonzero(owner == s) for s in snaps}

    def metrics(rows):
        acc = [np.mean(r[rows] == truth[rows]) for r in runs]
        f1 = [macro_f1(truth[rows], r[rows]) for r in runs]
        return acc, f1

    acc, f1 = metrics(np.concatenate([rows_of[s] for s in snaps]))
    rng, boot = np.random.default_rng(0), []
    for _ in range(DRAWS):
        rows = np.concatenate([rows_of[s] for s in rng.choice(snaps, len(snaps))])
        a, f = metrics(rows)
        boot.append((np.mean(a), np.mean(f)))
    boot = np.array(boot)
    return {"snapshots": len(snaps), "bins": int(sum(len(r) for r in rows_of.values())), "seeds": len(runs),
            "accuracy": float(np.mean(acc)), "accuracy_std": float(np.std(acc)),
            "macro_f1": float(np.mean(f1)), "macro_f1_std": float(np.std(f1)),
            "interval_95": {"accuracy": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                            "macro_f1": np.percentile(boot[:, 1], [2.5, 97.5]).tolist()},
            "per_snapshot_accuracy": {str(s): float(np.mean([np.mean(r[rows_of[s]] == truth[rows_of[s]])
                                                             for r in runs])) for s in snaps}}


def paired(truth, owner, runs_a, runs_b, subset):
    """Seed-mean macro-F1 of a minus b, with a bootstrap interval that resamples the same Snapshots for both."""
    snaps = np.unique(owner[np.isin(owner, subset)])
    rows_of = {s: np.flatnonzero(owner == s) for s in snaps}

    def gap(rows):
        return (np.mean([macro_f1(truth[rows], r[rows]) for r in runs_a])
                - np.mean([macro_f1(truth[rows], r[rows]) for r in runs_b]))

    rng = np.random.default_rng(0)
    boot = [gap(np.concatenate([rows_of[s] for s in rng.choice(snaps, len(snaps))])) for _ in range(DRAWS)]
    return {"macro_f1": float(gap(np.concatenate(list(rows_of.values())))),
            "interval_95": np.percentile(boot, [2.5, 97.5]).tolist()}


def main():
    out = HERE / "runs/temporal_baselines"
    out.mkdir(parents=True, exist_ok=True)
    data, z, y, multi = load()
    test = np.flatnonzero(data["split"] == "test")
    _, truth, owner = features(z, y, test, "bin_only", None)
    train_bins = y[data["split"] == "train"]
    majority = np.bincount(train_bins[train_bins >= 0]).argmax()

    runs = {"majority": [np.full(len(truth), majority)]}
    for mode in ("bin_only", "window_shuffled", "window"):
        runs[mode] = [logistic(data, z, y, mode, seed) for seed in SEEDS]
        print(f"done: {mode}", flush=True)
    runs["cnn"] = [cnn(data, z, y, seed) for seed in SEEDS]

    subsets = {"all": test, "multi_workload": test[multi[test]]}
    metrics = {name: {model: summarize(truth, owner, r, idx) for model, r in runs.items()}
               for name, idx in subsets.items()}
    metrics["paired_difference"] = {f"{a}_minus_{b}": {name: paired(truth, owner, runs[a], runs[b], idx)
                                                       for name, idx in subsets.items()}
                                    for a, b in (("window", "window_shuffled"), ("window", "bin_only"),
                                                 ("window_shuffled", "bin_only"), ("cnn", "bin_only"))}
    metrics["setup"] ={"window_bins_each_side": K, "paths": {str(i): data["paths"][i] for i in subsets["multi_workload"]}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    print("\npaired difference in macro-F1 (seed means, same resampled Snapshots for both models)")
    for pair, by_subset in metrics["paired_difference"].items():
        for name, d in by_subset.items():
            lo, hi = d["interval_95"]
            print(f"  {pair:<32}{name:<16}{d['macro_f1']:+.3f} ({lo:+.3f} to {hi:+.3f})")
    print(f"\n{'test Snapshots':<16}{'model':<17}{'bins':>6}  {'accuracy':<14}{'macro-F1 (95% CI)'}")
    for name in subsets:
        for model, m in metrics[name].items():
            lo, hi = m["interval_95"]["macro_f1"]
            print(f"{name:<16}{model:<17}{m['bins']:>6}  {m['accuracy']:.3f}±{m['accuracy_std']:.3f}   "
                  f"{m['macro_f1']:.3f}±{m['macro_f1_std']:.3f} ({lo:.2f}-{hi:.2f})")


if __name__ == "__main__":
    main()
