# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scikit-learn", "aeon==1.6.0", "torch==2.15.0",
#                 "transformers==4.57.6", "safetensors==0.8.0", "huggingface_hub"]
# ///
"""Baselines for the KernelSight workload-class task, scored like the LLM in flamingo/finetune_kernelsight.py.

  majority             the most common training class
  logistic_regression  each Channel's mean and std (48 features), from finetune_kernelsight.py
  nearest_neighbor     the class of the closest training trace (Euclidean); high when test traces have near-copies in train
  minirocket           MiniRocket (aeon, 10,000 kernels + ridge), the usual strong time-series classification baseline
  minirocket_shuffled  MiniRocket on Snapshots with their Bins in a random order (one permutation per Snapshot, the same
                       for all its Channels): a control for how much the score depends on time order
  cnn                  Nawaf's kernel-xray L1TemporalCNN, unchanged. Its per-bin logits are averaged over time, so it learns
                       from the trace label as the LLM does (kernel-xray trains it on per-bin labels). Recipe of kernel-xray
                       configs/phase3_cnn.json: AdamW, lr 1e-3, weight decay 1e-4, batch 16, at most 30 epochs, stop after 5
                       epochs without a better validation macro-F1, keep the best epoch.

Inputs of the last three: each Channel scaled with its training mean and std, as in kernel-xray. Each model runs on our
grouped-by-Problem split, on the official release split, and on the official split with its train cut at random to our 862
traces (so a gap is not only more training data). MiniRocket and the CNN run 5 seeds (mean and std over seeds). Every
model also gets a 95% bootstrap interval over the test traces (1,000 resamples) of its seed-mean metrics.

  uv run baselines.py    # its own environment: aeon would downgrade numpy in the OpenTSLM venv
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from aeon.classification.convolution_based import MiniRocketClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "flamingo"), str(HERE / "src")]
from finetune_kernelsight import METRICS, baselines, interval, score  # noqa: E402
from kernel_xray.phase3.cnn import L1TemporalCNN  # noqa: E402

OFFICIAL = {"train": "train", "val": "validation", "test": "test"}
SEEDS = range(5)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def scaled(data: dict[str, np.ndarray]) -> np.ndarray:
    x, train = data["x"].astype(np.float64), data["split"] == "train"
    mean, std = x[train].mean(axis=(0, 2)), x[train].std(axis=(0, 2))
    return ((x - mean[:, None]) / np.where(std > 1e-12, std, 1.0)[:, None]).astype(np.float32)


def nearest_neighbor(data: dict[str, np.ndarray], z: np.ndarray) -> list[str]:
    train, test = data["split"] == "train", data["split"] == "test"
    model = KNeighborsClassifier(n_neighbors=1).fit(z[train].reshape(train.sum(), -1), data["y"][train])
    return model.predict(z[test].reshape(test.sum(), -1)).tolist()


def minirocket(data: dict[str, np.ndarray], z: np.ndarray, seed: int) -> list[str]:
    train, test = data["split"] == "train", data["split"] == "test"
    return MiniRocketClassifier(random_state=seed, n_jobs=-1).fit(z[train], data["y"][train]).predict(z[test]).tolist()


def shuffled(z: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([s[:, rng.permutation(s.shape[1])] for s in z])


def cnn(data: dict[str, np.ndarray], z: np.ndarray, seed: int) -> list[str]:
    labels, split = data["labels"].tolist(), data["split"]
    x, y = torch.tensor(z), torch.tensor([labels.index(c) for c in data["y"].tolist()])
    rows = {name: torch.tensor(np.flatnonzero(split == name)) for name in ("train", "validation", "test")}
    torch.manual_seed(seed)
    model = L1TemporalCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    order = torch.Generator().manual_seed(seed)

    def predict(part: torch.Tensor) -> list[str]:
        model.eval()
        with torch.no_grad():
            best = [model(x[batch].to(DEVICE)).mean(-1).argmax(-1).cpu() for batch in part.split(64)]
        return [labels[i] for i in torch.cat(best).tolist()]

    best_f1, best_epoch, best_state, epoch = -1.0, 0, None, 0
    while epoch < 30 and epoch - best_epoch < 5:
        epoch += 1
        model.train()
        for batch in rows["train"][torch.randperm(len(rows["train"]), generator=order)].split(16):
            loss = F.cross_entropy(model(x[batch].to(DEVICE)).mean(-1), y[batch].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f1 = score(data["y"][rows["validation"]].tolist(), predict(rows["validation"]))["macro_f1"]
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(rows["test"])


def summarize(answers: list[str], runs: list[list[str]]) -> dict:
    """Mean and std of each metric over the runs, a 95% bootstrap interval of that mean over the test traces,
    and per-class F1 of the first run."""
    scores = [score(answers, run) for run in runs]
    out = {"n": len(answers), "seeds": len(runs)}
    for key in METRICS:
        out[key], out[f"{key}_std"] = float(np.mean([s[key] for s in scores])), float(np.std([s[key] for s in scores]))
    out["interval_95"] = interval(answers, runs)
    out["per_class_f1"] = scores[0]["per_class_f1"]
    return out


def main():
    out = HERE / "runs/baselines"
    out.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(HERE / "kernelsight.npz"))
    official = np.array([OFFICIAL[s] for s in data["official_split"].tolist()])
    official_862 = official.copy()
    extra = (official == "train").sum() - (data["split"] == "train").sum()
    official_862[np.random.default_rng(0).choice(np.flatnonzero(official == "train"), extra, replace=False)] = "unused"

    metrics = {}
    for name, split in {"grouped": data["split"], "official": official, "official_862": official_862}.items():
        view = {**data, "split": split}
        test, z = split == "test", scaled(view)
        runs = {model: [p] for model, p in baselines(view).items()}
        runs["nearest_neighbor"] = [nearest_neighbor(view, z)]
        runs["minirocket"] = [minirocket(view, z, seed) for seed in SEEDS]
        runs["minirocket_shuffled"] = [minirocket(view, shuffled(z), seed) for seed in SEEDS]
        runs["cnn"] = [cnn(view, z, seed) for seed in SEEDS]
        answers, paths = view["y"][test].tolist(), view["paths"][test].tolist()
        metrics[name] = {model: summarize(answers, r) for model, r in runs.items()}
        for model, r in runs.items():
            (out / f"predictions_{model}_{name}.jsonl").write_text("".join(
                json.dumps({"path": p, "answer": a, "prediction": c}) + "\n" for p, a, c in zip(paths, answers, r[0])))
        print(f"done: {name}", flush=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    print(f"\n{'split':<14}{'model':<21}{'n':>4}  {'accuracy (95% CI)':<26}{'balanced acc.':<16}{'macro-F1':<14}")
    for name, models in metrics.items():
        for model, m in models.items():
            ci = m["interval_95"]["accuracy"]
            cells = [f"{m[k]:.3f}" + (f"±{m[k + '_std']:.3f}" if m["seeds"] > 1 else "") for k in METRICS]
            print(f"{name:<14}{model:<21}{m['n']:>4}  {cells[0] + f' ({ci[0]:.2f}-{ci[1]:.2f})':<26}{cells[1]:<16}{cells[2]:<14}")


if __name__ == "__main__":
    main()
