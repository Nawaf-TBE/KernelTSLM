# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scikit-learn", "tokenizers"]
# ///
"""CPU baselines and design checks for the segmentation timeline, all scored by score.py on the SAME rows
(runs/usecases/segmentation/{validation,test}.jsonl; 291 Snapshots each).

  grid_oracle             the answers themselves: the best per-Bin score a 16-Bin grid can express
  majority_answer         the most common training answer for every row
  position_prior          per cell, the most common training state (no input)
  text_only_centred       what the text lines alone carry: Snapshot class from logistic regression on each
                          Channel's mean+std (finetune_kernelsight.baselines, refit here), busy share from ridge on
                          the same 48 numbers, painted as one centred busy block (all traces are centred)
  lr_window               per-Bin logistic regression on the Bin and 8 neighbours each side (24 Channels, z-scaled
                          with training statistics as temporal_baselines.py), 13 states incl. idle and unlabeled,
                          then the 16-Bin grid; 3 seeds (the seed picks 12% of training Bins)
  lr_window_per_bin       the same predictions without the grid
  lr_window_shuffled_neighbours  the neighbours in random order (the Bin stays at the centre); 3 seeds
  lr_window_input_shuffled / _input_reversed   seed-0 lr_window on the controls RUN applies to the input,
                          scored as the TSLM controls are (shuffled vs the true timeline, reversed vs reversed)
  lr_window_busy_painted_snapshot_lr   seed-0 lr_window busy Bins painted with the Snapshot LR class
  lr_window_busy_painted_minirocket    the same with the saved MiniRocket class (runs/baselines, seed 0, test only)
Design checks: grid width (validation chooses; test shown) and the last Channel (validation).

  uv run usecases/segmentation/baselines.py    # ~2 min on an M-series Mac; writes runs/usecases/segmentation/baselines.json
"""

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import IDLE, N, ROOT, STATES, W, _npz, decode, score, states, timeline  # noqa: E402

K, KEEP, SEEDS = 8, 0.12, range(3)
OUT = ROOT / "runs/usecases/segmentation"
X = _npz["x"]
SPLIT, PATHS = _npz["split"], _npz["paths"].tolist()
INDEX = {s: np.flatnonzero(SPLIT == s) for s in ("train", "validation", "test")}
ROWS = {s: [json.loads(line) for line in open(OUT / f"{s}.jsonl")] for s in INDEX}
assert all([r["path"] for r in ROWS[s]] == [PATHS[i] for i in INDEX[s]] for s in INDEX)
CLASS_INDEX = {str(c): k for k, c in enumerate(_npz["labels"])}
EVAL = ("validation", "test")


def windows(z, shuffle=None):
    """(512, Channels x 17): each Bin with K neighbours each side; with a rng, the neighbours in random order."""
    pad = np.pad(z, ((0, 0), (K, K)), mode="edge")
    w = np.lib.stride_tricks.sliding_window_view(pad, 2 * K + 1, axis=1).transpose(1, 0, 2)
    if shuffle is not None:
        others = np.delete(np.arange(2 * K + 1), K)
        order = np.stack([np.insert(shuffle.permutation(others), K, K) for _ in range(N)])
        w = np.take_along_axis(w, order[:, None, :].repeat(w.shape[1], 1), 2)
    return w.reshape(N, -1)


def fit_bins(z, seed, shuffled):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in INDEX["train"]:
        keep = rng.random(N) < KEEP
        xs.append(windows(z[i], rng if shuffled else None)[keep]), ys.append(STATES[i][keep])
    return LogisticRegression(max_iter=500, C=0.1).fit(np.concatenate(xs), np.concatenate(ys)), rng


def predict_bins(model, series, rng=None):
    return model.predict(np.concatenate([windows(s, rng) for s in series])).reshape(len(series), N)


def centred(cls, busy_cells):
    s = np.full(N, IDLE)
    a = (N // W - busy_cells) // 2
    s[a * W:(a + busy_cells) * W] = cls
    return s


def paint(pred, classes):
    return [timeline(np.where(p != IDLE, CLASS_INDEX[c], IDLE)) for p, c in zip(pred, classes)]


def mean_of(results):
    """Seed mean of every number in a list of score dicts."""
    if isinstance(results[0], dict):
        return {k: mean_of([r[k] for r in results]) for k in results[0]}
    return float(np.mean(results))


def text_only():
    """Per split: the Snapshot LR class names and the ridge busy share in cells, from mean+std only."""
    stats = np.concatenate([X.mean(-1), X.std(-1)], axis=1)
    scaler = StandardScaler().fit(stats[INDEX["train"]])
    train = scaler.transform(stats[INDEX["train"]])
    cls = LogisticRegression(C=1, max_iter=5000).fit(train, _npz["y"][INDEX["train"]])
    busy = Ridge(alpha=1.0).fit(train, (STATES[INDEX["train"]] != IDLE).mean(1))
    out = {}
    for s in EVAL:
        f = scaler.transform(stats[INDEX[s]])
        out[s] = (cls.predict(f).tolist(),
                  np.clip(np.rint(busy.predict(f) * N / W), 0, N // W).astype(int))
    return out


def saved_minirocket():
    pred = {(r := json.loads(line))["path"]: r["prediction"]
            for line in open(ROOT / "runs/baselines/predictions_minirocket_grouped.jsonl")}
    return [pred[r["path"]] for r in ROWS["test"]]


def tokenizer():
    files = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--opentime--tslm-nov26/snapshots/*/tokenizer.json"))
    if not files:
        return None
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(files[0])
    return lambda text: len(t.encode(text, add_special_tokens=False).ids)


def spread(values):
    return {"mean": float(np.mean(values)), "p90": float(np.percentile(values, 90)), "max": int(np.max(values))}


def grid_choice(count):
    out = {}
    for split in EVAL:
        for width in (4, 8, 16, 32, 64):
            texts = [timeline(states(r["path"]), width) for r in ROWS[split]]
            m = score(ROWS[split], texts)
            out.setdefault(split, {})[f"{width}_bins"] = {
                k: m[k] for k in ("bin_workload_accuracy", "bin_workload_macro_f1", "bin_busy_idle_accuracy", "boundary_f1")
            } | {"multi_workload_accuracy": m["multi_workload"]["bin_workload_accuracy"],
                 "answer_tokens": spread([count(t) for t in texts]) if count else None}
    return out


def last_channel():
    """Validation: per-Bin busy/idle and workload accuracy from ONE Channel, z-scaled per Snapshot as the
    encoder sees it (utils._scale), with lr_window on that Channel alone."""
    x = X.astype(np.float64)
    sd = x.std(2, keepdims=True)
    z = (x - x.mean(2, keepdims=True)) / np.where(sd > 0, sd, 1)
    names = [str(c) for c in _npz["channels"]]
    out = {}
    for label, chans in [*((n, [k]) for k, n in enumerate(names)), ("all 24", list(range(24)))]:
        model, _ = fit_bins(z[:, chans], 0, False)
        pred, truth = predict_bins(model, z[INDEX["validation"]][:, chans]), STATES[INDEX["validation"]]
        out[label] = {"busy_idle_accuracy": float(np.mean((pred != IDLE) == (truth != IDLE))),
                      "workload_accuracy": float(np.mean(pred[truth >= 0] == truth[truth >= 0]))}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["busy_idle_accuracy"]))


def tokens(count):
    stat = lambda v: f"{v:.4e}" if abs(v) >= 1e6 else f"{v:.4f}"  # noqa: E731  utils._statistic
    x = X.astype(np.float64)
    names = [str(c) for c in _npz["channels"]]
    lines = [sum(count(f"This is time series channel {n}, with mean {stat(x[i, k].mean())} and std {stat(x[i, k].std())}.") + 2
                 for k, n in enumerate(names)) for i in INDEX["test"]]
    return {"prompt": count(ROWS["test"][0]["prompt"]), "series_text_lines_incl_2_marker_tokens": spread(lines),
            "answer": {s: spread([count(r["answer"]) for r in ROWS[s]]) for s in ROWS}}


def main():
    results = {s: {"grid_oracle": score(ROWS[s], [r["answer"] for r in ROWS[s]])} for s in EVAL}
    answers = [r["answer"] for r in ROWS["train"]]
    majority = Counter(answers).most_common(1)[0][0]
    prior = [Counter(col).most_common(1)[0][0] for col in np.stack([decode(a)[::W] for a in answers]).T]
    snapshot = text_only()
    for s in EVAL:
        n = len(ROWS[s])
        results[s]["majority_answer"] = score(ROWS[s], [majority] * n)
        results[s]["position_prior"] = score(ROWS[s], [timeline(np.repeat(prior, W))] * n)
        results[s]["text_only_centred"] = score(ROWS[s], [timeline(centred(CLASS_INDEX[c], k)) for c, k in zip(*snapshot[s])])
    print("done: prior and text-only baselines", flush=True)

    x = X.astype(np.float64)
    mean, std = x[INDEX["train"]].mean(axis=(0, 2)), x[INDEX["train"]].std(axis=(0, 2))
    z = (x - mean[:, None]) / np.where(std > 1e-12, std, 1.0)[:, None]
    runs = {s: {"lr_window": [], "lr_window_per_bin": [], "lr_window_shuffled_neighbours": []} for s in EVAL}
    for seed in SEEDS:
        model, _ = fit_bins(z, seed, False)
        shuffled_model, rng = fit_bins(z, seed, True)
        for s in EVAL:
            rows, series = ROWS[s], z[INDEX[s]]
            pred = predict_bins(model, series)
            runs[s]["lr_window"].append(score(rows, [timeline(p) for p in pred]))
            runs[s]["lr_window_per_bin"].append(score(rows, [timeline(p, 1) for p in pred]))
            runs[s]["lr_window_shuffled_neighbours"].append(
                score(rows, [timeline(p) for p in predict_bins(shuffled_model, series, rng)]))
            if seed > 0:
                continue
            order = np.random.default_rng(0).permutation(N)
            results[s]["lr_window_input_shuffled"] = score(
                rows, [timeline(p) for p in predict_bins(model, series[:, :, order])], control="shuffled")
            results[s]["lr_window_input_reversed"] = score(
                [{**r, "answer": r["controls"]["reversed"]} for r in rows],
                [timeline(p) for p in predict_bins(model, series[:, :, ::-1])], control="reversed")
            results[s]["lr_window_busy_painted_snapshot_lr"] = score(rows, paint(pred, snapshot[s][0]))
            if s == "test":
                results[s]["lr_window_busy_painted_minirocket"] = score(rows, paint(pred, saved_minirocket()))
        print(f"done: lr_window seed {seed}", flush=True)
    for s in EVAL:
        for name, r in runs[s].items():
            results[s][name] = {"seeds": len(r), **mean_of(r)}

    quoted = json.load(open(ROOT / "runs/temporal_baselines/metrics.json"))
    count = tokenizer()
    out = {
        "rows": {s: len(ROWS[s]) for s in EVAL},
        "note": "All scored by usecases/segmentation/score.py. One run each unless 'seeds' is given (seed mean).",
        "test": results["test"],
        "validation": results["validation"],
        "temporal_baselines_test_labelled_bins_only": {
            "source": "runs/temporal_baselines/metrics.json (5 seeds; labelled Bins only; no grid; LR trained on labelled Bins)",
            **{subset: {m: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "seeds": v["seeds"]}
                        for m, v in quoted[subset].items()} for subset in ("all", "multi_workload")}},
        "grid_choice": grid_choice(count),
        "last_channel_choice_validation": last_channel(),
        "tokens_gemma3": tokens(count) if count else None,
    }
    (OUT / "baselines.json").write_text(json.dumps(out, indent=1))
    for s in EVAL:
        for name, m in results[s].items():
            print(f"{s:<11}{name:<36} main {m['main']:.3f} asym {m['main_asymmetric']:.3f} "
                  f"acc {m['bin_workload_accuracy']:.3f} F1 {m['bin_workload_macro_f1']:.3f} "
                  f"busy/idle {m['bin_busy_idle_accuracy']:.3f} bF1 {m['boundary_f1']:.3f} "
                  f"multi acc {m['multi_workload']['bin_workload_accuracy']:.3f}")


if __name__ == "__main__":
    main()
