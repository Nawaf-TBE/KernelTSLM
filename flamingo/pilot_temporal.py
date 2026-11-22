"""Pilot: does OpenTSLM-Flamingo learn the temporal questions of ../runs/temporal_qa?

Trains the base model for one epoch on the class questions of finetune_kernelsight.py (1,509 with repeats)
mixed with a random sample of the temporal training questions. Then it scores, on a sample of each
template's test questions: the trained model, the same model with each Snapshot's Bins shuffled (one
permutation for all 24 Channels, so time order is gone but the values stay), and the majority answer of the
template's training questions. The class test set is scored too, to see whether the mix hurts it.
The trained model is saved to --out/model before the scoring, so a crash there keeps it.

The questions are rewritten against the two shortcuts of the first pilot (../runs/pilot, which used them
unchanged): a question that names counters gets only those Channels, so the question tokens are near the
series they ask about (each text token reads only the latents of the most recent series, utils.py:196); options
"A"-"D" become the Bin ranges they stand for; a number becomes one of 4 ranges cut at the training quartiles.

  setsid nohup python -u pilot_temporal.py --out ../runs/pilot2 > ../runs/pilot2/train.log 2>&1 < /dev/null &
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from finetune_kernelsight import evaluate, kernelsight_examples, oversampled, record_losses, score, shuffled
from utils import Example, MaskedCrossAttention, OpenTSLM, SeriesInput, train


LETTERS = re.compile(r"([A-D]): (Bins \d+-\d+)")


def read_rows(folder: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in (folder / f"{split}.jsonl").open()]


def number_ranges(train_rows: list[dict]) -> dict[str, list[tuple[int, int | None]]]:
    """For each template with a number answer: 4 (low, high) ranges cut at the training quartiles; high None = open."""
    ranges = {}
    for template in {r["template"] for r in train_rows if not r["options"]}:
        cuts = np.percentile([int(r["answer"]) for r in train_rows if r["template"] == template], [25, 50, 75]).astype(int).tolist()
        lows = [1, *(c + 1 for c in cuts)]
        ranges[template] = [(lo, hi) for lo, hi in zip(lows, [*cuts, None])]
    return ranges


def rewritten(row: dict, channels: list[str], ranges: dict) -> dict:
    """The row with its Channels to show, and its options as what they stand for."""
    question, options, answer = row["question"], row["options"], row["answer"]
    named = [c for c in re.findall(r"'([^']+)'", question) if c in channels]
    if options == ["A", "B", "C", "D"]:
        meaning = dict(LETTERS.findall(question))
        question = LETTERS.split(question)[0].rstrip() + " Answer with one of the answer_options."
        options, answer = [meaning[letter] for letter in options], meaning[answer]
    elif not options:
        labels = [f"{lo} or more" if hi is None else str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in ranges[row["template"]]]
        index = next(i for i, (lo, hi) in enumerate(ranges[row["template"]]) if hi is None or int(answer) <= hi)
        question = question.replace("Answer with a number.", "Answer with one of the answer_options.")
        options, answer = labels, labels[index]
    return {**row, "question": question, "options": options, "answer": answer, "channels": named or channels}


def peak_rows(data: dict[str, np.ndarray], split: str, per_snapshot: int = 3, balanced: bool = False) -> list[dict]:
    """Questions on one Channel: is its highest value before or after Bin N? Only Channels with a clear peak (every
    value more than 4 Bins away is below 80% of it). N is 16-96 Bins from the peak, on a random side, so the two
    answers are equally common.

    NOTE: Most peaks are near Bins 128-255, so without ``balanced`` the value of N alone gives the answer away (a rule
    "after when N < 205" gets 0.755). ``balanced`` keeps, within each range of 32 values of N, as many "before" as "after"
    questions, so N alone tells nothing."""
    rng, channels, rows = np.random.default_rng(0), data["channels"].tolist(), []
    for x, path, name in zip(data["x"], data["paths"].tolist(), data["split"].tolist()):
        if name != split:
            continue
        found = 0
        for c in rng.permutation(len(channels)):
            peak, top = int(x[c].argmax()), float(x[c].max())
            rest = np.concatenate([x[c][: max(peak - 4, 0)], x[c][peak + 5:]])
            bin_n = peak + int(rng.choice([-1, 1])) * int(rng.integers(16, 97))
            if top <= 0 or rest.max(initial=0) >= 0.8 * top or not 1 <= bin_n <= 510:
                continue
            options = [f"before Bin {bin_n}", f"after Bin {bin_n}"]
            rows.append({"path": path, "split": split, "template": "peak_side", "channels": [channels[c]],
                         "options": options, "answer": options[peak > bin_n], "question": (
                             "The trace has 512 Bins in time order: Bin 0 is the start and Bin 511 is the end. "
                             f"Does '{channels[c]}' reach its highest value before or after Bin {bin_n}? "
                             "Answer with one of the answer_options.")})
            found += 1
            if found == per_snapshot:
                break
    if not balanced:
        return rows
    kept, by_range = [], defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_range[int(re.search(r"Bin (\d+)\?", row["question"]).group(1)) // 32][row["answer"].split()[0]].append(row)
    for sides in by_range.values():
        n = min(len(sides["before"]), len(sides["after"]))
        kept += sides["before"][:n] + sides["after"][:n]
    return kept


TREND = ["increasing", "decreasing"]


def trend_pairs(data: dict[str, np.ndarray], split: str) -> list[tuple[dict, dict]]:
    """Questions on one Channel: over the Bins where it is active (above 5% of its maximum), is its trend increasing
    or decreasing? Asked only when the mean of one half of that span is at least 1.5 times the other. Each question
    comes in a pair with the same Snapshot reversed in time and the opposite answer, so the two answers are equally
    common and the two questions differ only in time order.

    NOTE: 89% of the clear trends in the release increase; without the reversed copies, "increasing" alone would
    score 0.89. Within a pair, a model that ignores time order gives the same answer twice, so it gets one right."""
    channels, pairs = data["channels"].tolist(), []
    for x, path, name in zip(data["x"], data["paths"].tolist(), data["split"].tolist()):
        if name != split:
            continue
        for c, v in enumerate(x):
            on = np.flatnonzero(v > 0.05 * v.max()) if v.max() > 0 else []
            if len(on) == 0 or on[-1] - on[0] < 16:
                continue
            span = v[on[0]:on[-1] + 1]
            first, second = float(span[: len(span) // 2].mean()), float(span[len(span) // 2:].mean())
            if max(first, second) < 1.5 * min(first, second):
                continue
            row = {"path": path, "split": split, "template": "trend", "channels": [channels[c]], "options": TREND,
                   "answer": TREND[first > second], "reversed": False, "question": (
                       "The trace has 512 Bins in time order: Bin 0 is the start and Bin 511 is the end. "
                       f"Look at the Bins where '{channels[c]}' is active. Over those Bins, is the trend of "
                       f"'{channels[c]}' increasing or decreasing? Answer with one of the answer_options.")}
            pairs.append((row, {**row, "answer": TREND[first <= second], "reversed": True}))
    return pairs


def both_right(path: Path) -> float:
    """The share of trend pairs (written forward, then reversed) with both questions right."""
    rows = [json.loads(line) for line in path.open()]
    return float(np.mean([a["prediction"] == a["answer"] and b["prediction"] == b["answer"]
                          for a, b in zip(rows[::2], rows[1::2])]))


def temporal_examples(data: dict[str, np.ndarray], rows: list[dict], shuffled: bool = False,
                      segments: bool = False) -> list[tuple[str, Example]]:
    """One (release path, Example) per question, with the row's Channels. Series are built once and shared.
    With ``segments``, each Channel of a template other than peak_side is passed as 4 series of 128 Bins,
    each named with its Bin range, so the text says where each part is in time."""
    index = {c: i for i, c in enumerate(data["channels"].tolist())}
    x = dict(zip(data["paths"].tolist(), data["x"]))

    def cut(row: dict) -> bool:
        return segments and row["template"] not in ("peak_side", "trend")

    def key(row: dict) -> tuple:
        return row["path"], tuple(row["channels"]), cut(row), row.get("reversed", False)

    series: dict[tuple, list[SeriesInput]] = {}
    for key_ in {key(r) for r in rows}:
        values = x[key_[0]][[index[c] for c in key_[1]]]
        if key_[3]:
            values = values[:, ::-1]
        if shuffled:  # one permutation of the Bins for all Channels: time order is gone, the values stay
            values = values[:, np.random.default_rng(0).permutation(values.shape[1])]
        if key_[2]:
            series[key_] = [SeriesInput(values=v[a:a + 128].astype(float).tolist(), name=f"{c}, Bins {a}-{a + 127}")
                            for v, c in zip(values, key_[1]) for a in range(0, 512, 128)]
        else:
            series[key_] = [SeriesInput(values=v.astype(float).tolist(), name=c) for v, c in zip(values, key_[1])]
    return [(r["path"], Example(series=series[key(r)], prompt=r["question"],
                                options=r["options"], answer=r["answer"], category=r["template"])) for r in rows]


def per_template(rows: list[dict], n: int, rng: random.Random) -> dict[str, list[dict]]:
    by_template = defaultdict(list)
    for row in rows:
        by_template[row["template"]].append(row)
    return {t: rng.sample(part, min(n, len(part))) for t, part in sorted(by_template.items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="hf://opentime/tslm-nov26")
    p.add_argument("--data", default="../kernelsight.npz")
    p.add_argument("--questions", default="../runs/temporal_qa")
    p.add_argument("--out", default="../runs/pilot2")
    p.add_argument("--train", type=int, default=2000, help="temporal training questions (random sample); 0 = none, and no test of them")
    p.add_argument("--validation", type=int, default=150, help="temporal validation questions (random sample)")
    p.add_argument("--test", type=int, default=30, help="test questions per template")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--peak", type=int, default=0, help="peak_side training questions to add (random sample)")
    p.add_argument("--peak-balanced", action="store_true", help="balance peak_side answers for each range of N")
    p.add_argument("--segments", action="store_true", help="pass each Channel as 4 named time segments")
    p.add_argument("--trend", type=int, default=0, help="trend training questions to add (forward and reversed pairs)")
    p.add_argument("--trend-test", type=int, default=150, help="trend test pairs")
    p.add_argument("--before", action="store_true", help="also score the base model on the temporal tests")
    p.add_argument("--attend-all", action="store_true", help="let the question attend to all series (utils.py)")
    p.add_argument("--seed", type=int, default=0, help="training order and option shuffles; the sampled questions stay fixed")
    args = p.parse_args()
    out, folder = Path(args.out), Path(args.questions)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.pid").write_text(str(os.getpid()))
    metrics = {"args": vars(args)}

    def report(key: str, value) -> None:
        metrics[key] = value
        print(json.dumps({key: value}), flush=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    data, rng = dict(np.load(args.data)), random.Random(0)
    classes, channels = kernelsight_examples(data), data["channels"].tolist()
    ranges = number_ranges(read_rows(folder, "train"))
    rows = {split: [rewritten(r, channels, ranges) for r in read_rows(folder, split)] for split in ("train", "validation", "test")}
    train_rows = rows["train"]
    report("number_ranges", ranges)
    sample = rng.sample(train_rows, args.train)
    validation = rng.sample(rows["validation"], args.validation)
    if args.peak:
        peak = {split: peak_rows(data, split, balanced=args.peak_balanced) for split in ("train", "validation", "test")}
        train_rows = train_rows + peak["train"]
        sample += rng.sample(peak["train"], min(args.peak, len(peak["train"])))
        validation += rng.sample(peak["validation"], min(50, len(peak["validation"])))
        rows["test"] += peak["test"]
    tests = per_template(rows["test"], args.test, rng) if args.train else {}
    if args.trend:
        trend = {split: trend_pairs(data, split) for split in ("train", "validation", "test")}
        train_rows = train_rows + [r for pair in trend["train"] for r in pair]
        sample += [r for pair in rng.sample(trend["train"], min(args.trend // 2, len(trend["train"]))) for r in pair]
        validation += [r for pair in rng.sample(trend["validation"], min(25, len(trend["validation"]))) for r in pair]
        tests["trend"] = [r for pair in rng.sample(trend["test"], min(args.trend_test, len(trend["test"]))) for r in pair]

    def examples(part: list[dict], mixed: bool = False) -> list[tuple[str, Example]]:
        return temporal_examples(data, part, mixed, args.segments)

    train_examples = oversampled([e for _, e in classes["train"]]) + [e for _, e in examples(sample)]
    val_examples = [e for _, e in classes["validation"]] + [e for _, e in examples(validation)]
    report("train", {"class": len(train_examples) - len(sample), **Counter(r["template"] for r in sample)})

    majority = {t: Counter(r["answer"] for r in train_rows if r["template"] == t).most_common(1)[0][0] for t in tests}
    report("majority", {t: score([r["answer"] for r in part], [majority[t]] * len(part))["accuracy"]
                        for t, part in tests.items()})

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU: install a CUDA build of torch")
    model = OpenTSLM.load(args.base_model).model
    for module in model.modules():
        if isinstance(module, MaskedCrossAttention):
            module.attend_all = args.attend_all

    def score_tests(name: str, mixed: bool) -> None:
        report(name, {t: evaluate(model, examples(part, mixed), out / f"{name}_{t}.jsonl")["accuracy"]
                      for t, part in tests.items()})
        if "trend" in tests:
            report(f"{name}_trend_pairs", both_right(out / f"{name}_trend.jsonl"))

    if args.before:
        score_tests("before", False)
    torch.manual_seed(args.seed)
    record_losses(model, out / "losses.jsonl")
    train(model, train_examples, val_examples, epochs=args.epochs, accumulation=8, seed=args.seed)
    OpenTSLM(model).save(str(out / "model"))

    report("class_test", evaluate(model, classes["test"], out / "class_test.jsonl"))
    report("class_test_shuffled", evaluate(model, shuffled(classes["test"]), out / "class_test_shuffled.jsonl"))
    score_tests("after", False)
    score_tests("shuffled", True)


if __name__ == "__main__":
    main()
