"""Fine-tune OpenTSLM-Flamingo (opentime/tslm-nov26) on several KernelSight use cases at once, then answer the
validation and test rows of each use case with the real input and three controls, and score them.

Rows: ../runs/usecases/<task>/{train,validation,test}.jsonl, one question per line with the keys id, task, type,
path, channels, prompt, options, answer, controls, max_new_tokens (the row contract of the use-case brief), and an
optional "input" ("reversed" or "shuffled": the row's own input is transformed, e.g. temporal/train_reversed.jsonl,
added with --add-train). The series come from kernelsight.npz by the row's release path: all 24 Channels in npz
order, or the row's Channels in the row's order. Scorers: ../usecases/<task>/score.py, score(rows, outputs)
-> {"n", "main", "by_type", ...}, loaded by path.

Controls on the input. The text line of each series (its mean and std) stays the own Snapshot's in every control,
so only the series change:
  real      the Snapshot as it is
  shuffled  the Bins in one fixed order, np.random.default_rng(0).permutation(512), for all Channels and all
            Snapshots (as finetune_kernelsight.shuffled); scored against controls["shuffled"], or the answer if null
  reversed  the Bins in reverse order; scored against controls["reversed"], or the answer if null
  swapped   the series of a donor Snapshot of another class in the same split, from --donors (CLS's donors.json;
            drawn when a Snapshot is not in it); scored against the own answer ("swapped") and against the donor's
            own row with the same type, prompt, Channels and options, when it has one ("swapped_donor")
Each row given to a scorer has "condition" (original, shuffled or reversed: the facts the input shows); a scorer
whose score() takes ``control`` gets it. Scorer extras, called when the module has them: pairs(rows, real,
reversed, hard_ids) and paired(rows, real, shuffled) (temporal), swap_read(rows, swapped, donor_answers) and
cluster_interval(rows, [real]) (classification).

Training: all chosen training rows of all tasks in one shuffled list, utils.train (AdamW, accumulation 8), the epoch
with the lowest validation loss kept; the validation loss of each task is logged per epoch at no extra cost.
Writes --out/metrics.json after each stage and --out/predictions_<task>[_validation]_<control>.jsonl. A rerun skips
each file that is complete. It trains again if any is missing, unless --save left a model in --out/model; each
predictions line names its model. The process exits 0 only when every file is complete. A scorer error is written
into metrics.json and does not stop the run; a rerun with the same arguments and nothing missing only scores again,
without a GPU (e.g. on the Mac after a pull).

  python -u multitask.py --tasks classification,temporal,segmentation,explanation --out ../runs/mt \
      > ../runs/mt/train.log 2>&1
"""

import argparse
import importlib.util
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from utils import Example, MaskedCrossAttention, OpenTSLM, SeriesInput, _autocast, _scale, train

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "validation", "test")
CONTROLS = ("real", "shuffled", "swapped", "reversed")
CONDITION = {"before": "original", "real": "original", "swapped": "original", "shuffled": "shuffled",
             "reversed": "reversed"}
KEYS = ("id", "task", "type", "path", "channels", "prompt", "options", "answer", "controls", "max_new_tokens")
ORDER = np.random.default_rng(0).permutation(512)  # NOTE: the use-case agents compute controls["shuffled"] with it


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def check(task: str, rows: dict[str, list[dict]], data: dict[str, np.ndarray]) -> None:
    """Stop before any GPU time when a row breaks the contract or sits in another split than its Snapshot."""
    split_of = dict(zip(data["paths"].tolist(), data["split"].tolist()))
    channels, ids = set(data["channels"].tolist()), set()
    for split, part in rows.items():
        for r in part:
            problem = (
                f"missing keys {sorted(set(KEYS) - set(r))}" if not set(KEYS) <= set(r)
                else f"task {r['task']!r}" if r["task"] != task
                else "duplicate id" if r["id"] in ids
                else f"unknown path {r['path']!r}" if r["path"] not in split_of
                else f"Snapshot of split {split_of[r['path']]!r}" if split_of[r["path"]] != split
                else f"unknown Channels {sorted(set(r['channels']) - channels)}"
                if r["channels"] is not None and (not r["channels"] or not set(r["channels"]) <= channels)
                else "answer not in options" if r["options"] and r["answer"] not in r["options"]
                else "max_new_tokens" if not (isinstance(r["max_new_tokens"], int) and r["max_new_tokens"] > 0)
                else f"input {r['input']!r}" if r.get("input", "real") not in ("real", "shuffled", "reversed")
                else None
            )
            if problem:
                raise SystemExit(f"{task}/{split}, row {r.get('id')}: {problem}")
            ids.add(r["id"])


def chosen(rows: list[dict], n: int | None, repeat: bool) -> list[dict]:
    """``n`` rows in a fixed random order (the same for every seed); with ``repeat``, more than there are repeats
    them, else all rows in file order."""
    if n is None or (n >= len(rows) and not repeat):
        return rows
    pool = random.Random(0).sample(rows, len(rows))
    return (pool * (n // max(len(pool), 1) + 1))[:n]


def per_task(text: str, tasks: list[str], parse, empty) -> dict:
    """'' = ``empty`` for each task; 'V' = parse(V) for each; 'a=V,b=V' = per task, the others ``empty``."""
    if not text:
        return dict.fromkeys(tasks, empty)
    if "=" not in text:
        return dict.fromkeys(tasks, parse(text))
    given = {k: parse(v) for k, v in (item.split("=") for item in text.split(","))}
    if set(given) - set(tasks):
        raise SystemExit(f"{text}: tasks not in --tasks: {sorted(set(given) - set(tasks))}")
    return {t: given.get(t, empty) for t in tasks}


def names(text: str) -> list[str]:
    return [c for c in text.replace("+", ",").split(",") if c]


def scorer(path: Path):
    """A use case's score.py, loaded by path; its folder can hold modules it imports."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"score_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def donors(data: dict[str, np.ndarray], rows: list[dict], split_rows: list[dict],
           fixed: dict[str, str]) -> list[tuple[str, dict | None]]:
    """For each row: a donor Snapshot of another class in the row's split (``fixed`` [own path], else drawn by the
    row id, if possible among the Snapshots that have the same question), and the donor's row with the same type,
    prompt, Channels and options in ``split_rows``, or None."""
    paths = data["paths"].tolist()
    label, split_of = dict(zip(paths, data["y"].tolist())), dict(zip(paths, data["split"].tolist()))

    def key(r: dict) -> tuple:
        return r["type"], r["prompt"], tuple(r["channels"] or ()), tuple(r["options"])

    same: dict[tuple, dict[str, dict]] = {}
    for r in split_rows:
        same.setdefault(key(r), {}).setdefault(r["path"], r)
    picked = []
    for r in rows:
        if r["path"] in fixed:
            donor = fixed[r["path"]]
        else:
            rng = random.Random(r["id"])  # shuffling, not security  # nosec B311
            other = sorted(p for p in same.get(key(r), {}) if label[p] != label[r["path"]])
            donor = rng.choice(other or sorted(p for p in paths if split_of[p] == split_of[r["path"]]
                                               and label[p] != label[r["path"]]))
        picked.append((donor, same.get(key(r), {}).get(donor)))
    return picked


def examples(data: dict[str, np.ndarray], rows: list[dict], control: str = "real",
             donor_paths: list[str] | None = None) -> list[Example]:
    """One Example per row with the input of ``control`` (with "real", the row's own "input" if it has one).
    Series are built once per Snapshot, Channels and input, and shared."""
    x = dict(zip(data["paths"].tolist(), data["x"]))
    names = data["channels"].tolist()
    index = {c: i for i, c in enumerate(names)}
    built: dict[tuple, list[SeriesInput]] = {}

    def series(path: str, channels: tuple[str, ...], shown: str, donor: str | None) -> list[SeriesInput]:
        key = (path, channels, shown, donor)
        if key not in built:
            picked = [index[c] for c in channels]
            own = x[path][picked].astype(float)
            values = {"real": own, "shuffled": own[:, ORDER], "reversed": own[:, ::-1],
                      "swapped": x[donor][picked].astype(float) if donor else None}[shown]
            built[key] = [SeriesInput(values=v.tolist(), name=c, stats=None if shown == "real" else _scale(o)[2:])
                          for v, o, c in zip(values, own, channels)]
        return built[key]

    return [Example(series=series(r["path"], tuple(r["channels"] or names),
                                  r.get("input", "real") if control in ("real", "before") else control, donor),
                    prompt=r["prompt"], options=r["options"], answer=r["answer"], category=f"{r['task']}/{r['type']}")
            for r, donor in zip(rows, donor_paths or [None] * len(rows))]


def expected(row: dict, control: str) -> str:
    answer = (row.get("controls") or {}).get(control)
    return row["answer"] if answer is None else answer


def complete(path: Path, rows: list[dict]) -> bool:
    try:
        return [json.loads(line)["id"] for line in path.open()] == [r["id"] for r in rows]
    except (OSError, ValueError, KeyError):
        return False


def predict(model, data, control: str, rows: list[dict], path: Path, model_id: str,
            donor: list[tuple[str, dict | None]] | None) -> None:
    """Answer each row greedily with its max_new_tokens and write one line per row as it goes."""
    model.eval()
    inputs = examples(data, rows, control, [d for d, _ in donor] if donor else None)
    with path.open("w") as sink:
        for i, (row, example) in enumerate(zip(rows, inputs)):
            started = time.time()
            with _autocast(model):
                output = model.generate(example.series, example.evaluated_question(), max_new_tokens=row["max_new_tokens"])
            line = {"id": row["id"], "type": row["type"], "path": row["path"], "answer": expected(row, control),
                    "output": output, "tokens": len(model.tokenizer(output, add_special_tokens=False)["input_ids"]),
                    "seconds": round(time.time() - started, 3), "model": model_id}
            if donor:
                line |= {"donor": donor[i][0], "donor_id": donor[i][1] and donor[i][1]["id"],
                         "donor_answer": donor[i][1] and donor[i][1]["answer"]}
            sink.write(json.dumps(line) + "\n")
            sink.flush()
            if i % 50 == 0:
                print(json.dumps({"evaluated": i, "of": len(rows), "file": path.name, "answer": line["answer"],
                                  "output": output[:120]}), flush=True)


def call(module, rows: list[dict], outputs: list[str], control: str) -> dict:
    rows = [{**r, "condition": CONDITION[control]} for r in rows]
    if "control" in inspect.signature(module.score).parameters:
        return module.score(rows, outputs, control=None if CONDITION[control] == "original" else control)
    return module.score(rows, outputs)


def scored(module, control: str, rows: list[dict], lines: list[dict], by_id: dict[str, dict]) -> dict:
    """The scorer's result for one predictions file, with its generation time, the mean output length in tokens
    (the output text tokenized again) and the model(s) that wrote it."""
    outputs = [p["output"] for p in lines]
    timing = {"seconds": round(sum(p["seconds"] for p in lines), 1),
              "tokens_mean": round(float(np.mean([p["tokens"] for p in lines])), 1),
              "model": sorted({p["model"] for p in lines})}
    result = {control: {**call(module, [{**r, "answer": expected(r, control)} for r in rows], outputs, control),
                        **timing}}
    if control == "swapped":
        kept = [(by_id[p["donor_id"]], p["output"]) for p in lines if p["donor_id"]]
        if kept:
            result["swapped_donor"] = call(module, [r for r, _ in kept], [o for _, o in kept], "real")
    return result


def extras(module, rows: list[dict], outputs: dict[str, list[str]], swapped: list[dict] | None, hard_ids) -> dict:
    """The scorer's own paired analyses, for those whose outputs exist."""
    out = {}
    if hasattr(module, "pairs") and {"real", "reversed"} <= set(outputs):
        out["pairs_reversed"] = module.pairs(rows, outputs["real"], outputs["reversed"], hard_ids)
    if hasattr(module, "paired") and {"real", "shuffled"} <= set(outputs):
        out["paired_shuffled"] = module.paired(rows, outputs["real"], outputs["shuffled"])
    if hasattr(module, "swap_read") and swapped:
        out["swap_read"] = module.swap_read(rows, outputs["swapped"], [p["donor_answer"] for p in swapped])
    if hasattr(module, "cluster_interval") and "real" in outputs:
        out["cluster_interval_real"] = module.cluster_interval(rows, [outputs["real"]])
    return out


def load(spec: str, attend_all: bool):
    if not torch.cuda.is_available():  # NOTE: utils._load would fall back to a float32 CPU run that takes days
        raise SystemExit("no CUDA GPU: the row files and scorers are checked; install a CUDA build of torch to run")
    model = OpenTSLM.load(spec).model
    for module in model.modules():
        if isinstance(module, MaskedCrossAttention):
            module.attend_all = attend_all
    return model


def recorded_losses(model, path: Path, validation: list[float]) -> None:
    """Write the loss of each training question to ``path``; keep each validation loss in ``validation``."""
    loss = type(model).loss.__get__(model)  # NOTE: the class method, so a second call does not wrap twice
    path.write_text("")

    def recorded(*args, **kwargs):
        value = loss(*args, **kwargs)
        mean = float(value.detach().float().mean())
        if model.training:
            with path.open("a") as sink:
                sink.write(json.dumps({"loss": mean}) + "\n")
        else:
            validation.append(mean)
        return value

    model.loss = recorded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True, help="comma list, e.g. classification,temporal")
    p.add_argument("--data", default=str(ROOT / "kernelsight.npz"))
    p.add_argument("--rows", default=str(ROOT / "runs/usecases"), help="folder with <task>/{train,validation,test}.jsonl")
    p.add_argument("--scorers", default=str(ROOT / "usecases"), help="folder with <task>/score.py")
    p.add_argument("--donors", default=str(ROOT / "runs/usecases/classification/donors.json"),
                   help="own path -> donor path for the swapped control of every task")
    p.add_argument("--add-train", action="append", default=[],
                   help="more training rows: <task>/<file>.jsonl[:type,...] under --rows, e.g. temporal/train_reversed.jsonl:trend")
    p.add_argument("--out", default=str(ROOT / "runs/multitask"))
    p.add_argument("--base-model", default="hf://opentime/tslm-nov26")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-rows", default="", help="rows per task: '' all, 'N', or 'task=N,...' (others all); more than there are repeats them")
    p.add_argument("--val-rows", default="300", help="as --train-rows, for the validation loss and validation answers")
    p.add_argument("--test-rows", default="", help="as --train-rows, for the test answers")
    p.add_argument("--controls", default=",".join(CONTROLS),
                   help="on test: 'real,shuffled,...' for every task, or 'task=real+swapped,...' (others none)")
    p.add_argument("--val-controls", default=",".join(CONTROLS), help="as --controls, on validation; '' = none")
    p.add_argument("--before", type=int, default=0, help="also score the base model on N test rows per task (real input)")
    p.add_argument("--save", action="store_true", help="save the trained model to --out/model (13.7 GB)")
    p.add_argument("--attend-all", action="store_true", help="let the question attend to all series (utils.py)")
    p.add_argument("--seed", type=int, default=0, help="training order and option shuffles; the chosen rows stay fixed")
    p.add_argument("--limit", type=int, default=0, help="at most N rows per task and split (smoke tests)")
    args = p.parse_args()
    tasks = args.tasks.split(",")
    controls = {"test": per_task(args.controls, tasks, names, []),
                "validation": per_task(args.val_controls, tasks, names, [])}
    unknown = {c for part in controls.values() for cs in part.values() for c in cs} - set(CONTROLS)
    if unknown:
        raise SystemExit(f"unknown controls: {sorted(unknown)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.pid").write_text(str(os.getpid()))
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update(args=vars(args), complete=False)

    def report(key: str, value) -> None:
        metrics[key] = value
        print(json.dumps({key: value})[:2000], flush=True)
        metrics_path.write_text(json.dumps(metrics, indent=1))

    data = dict(np.load(args.data))
    rows = {t: {s: read_rows(Path(args.rows) / t / f"{s}.jsonl") for s in SPLITS} for t in tasks}
    for spec in args.add_train:
        file, _, types = spec.partition(":")
        task = Path(file).parts[0]
        if task not in tasks:
            raise SystemExit(f"--add-train {spec}: {task} is not in --tasks")
        rows[task]["train"] += [r for r in read_rows(Path(args.rows) / file) if not types or r["type"] in types.split(",")]
    for t in tasks:
        check(t, rows[t], data)
    modules = {t: scorer(Path(args.scorers) / t / "score.py") for t in tasks}
    limits = {s: per_task(text, tasks, int, None) for s, text in zip(SPLITS, (args.train_rows, args.val_rows, args.test_rows))}
    if args.limit:
        limits = {s: {t: min(n or args.limit, args.limit) for t, n in part.items()} for s, part in limits.items()}
    used = {t: {s: chosen(rows[t][s], limits[s][t], s == "train") for s in SPLITS} for t in tasks}
    report("rows", {t: {s: {"file": len(rows[t][s]), "used": len(used[t][s])} for s in SPLITS} for t in tasks})
    report("oracle", {t: call(modules[t], used[t]["test"], [r["answer"] for r in used[t]["test"]], "real")["main"]
                      for t in tasks})

    # NOTE: the order of the work: the base model first, then task by task, the primary task (temporal) first and
    # explanation (the slowest answers) last; in a task real, reversed, shuffled, swapped, validation before test.
    # A stop midway then still leaves the primary result (temporal real + reversed: the pair score) complete.
    first = ("diagnosis", "temporal", "classification", "segmentation", "explanation")
    steps = ("real", "reversed", "shuffled", "swapped")
    pairs = [("test", t, "before") for t in tasks if args.before]
    pairs += sorted(((s, t, c) for c in CONTROLS for s in ("validation", "test") for t in tasks if c in controls[s][t]),
                    key=lambda p: (first.index(p[1]) if p[1] in first else len(first), steps.index(p[2]), p[0] == "test"))
    tested = {(s, t, c): chosen(used[t]["test"], args.before, False) if c == "before" else used[t][s]
              for s, t, c in pairs}
    files = {(s, t, c): out / f"predictions_{t}{'_validation' if s == 'validation' else ''}_{c}.jsonl"
             for s, t, c in pairs}
    by_id = {s: {r["id"]: r for t in tasks for r in rows[t][s]} for s in ("validation", "test")}
    results = metrics["results"] = {}

    def lines(pair: tuple) -> list[dict]:
        return [json.loads(line) for line in files[pair].open()]

    def rescore(split: str, task: str, control: str) -> None:
        try:
            found = scored(modules[task], control, tested[split, task, control], lines((split, task, control)), by_id[split])
        except Exception as error:  # NOTE: a scorer bug must not stop the GPU work; a rerun without GPU scores again
            found = {control: {"error": repr(error)}}
        results.setdefault(split, {}).setdefault(task, {}).update(found)
        report("summary", {s: {t: {c: round(v["main"], 3) if "main" in v else "error" for c, v in part.items()}
                               for t, part in tasks_.items()} for s, tasks_ in results.items()})

    def finish() -> None:
        found = {}
        for s in ("validation", "test"):
            for t in tasks:
                done = {c: lines((s, t, c)) for c in CONTROLS if (s, t, c) in files and complete(files[s, t, c], used[t][s])}
                baselines = Path(args.rows) / t / "baselines.json"
                hard = json.loads(baselines.read_text()).get("trend_hard_ids") if s == "test" and baselines.exists() else None
                try:
                    found.setdefault(s, {})[t] = extras(modules[t], used[t][s], {c: [p["output"] for p in v] for c, v in done.items()},
                                                        done.get("swapped"), hard)
                except Exception as error:
                    found.setdefault(s, {})[t] = {"error": repr(error)}
        report("extras", found)
        report("complete", all(complete(files[pair], tested[pair]) for pair in pairs))

    pending = [pair for pair in pairs if not complete(files[pair], tested[pair])]
    for pair in pairs:
        if pair not in pending:
            rescore(*pair)
    report("pending", ["/".join(pair) for pair in pending])
    if not pending:
        return finish()

    fixed = json.loads(Path(args.donors).read_text()) if Path(args.donors).exists() else {}
    swapped = {(s, t): donors(data, used[t][s], rows[t][s], fixed) for s in ("validation", "test") for t in tasks}
    saved = out / "model"
    before = [pair for pair in pending if pair[2] == "before"]
    after = [pair for pair in pending if pair[2] != "before"]
    reuse = bool(after) and (saved / "model.safetensors").exists()
    model = load(args.base_model, args.attend_all) if before or not reuse else None
    for s, t, c in before:
        predict(model, data, c, tested[s, t, c], files[s, t, c], "base", None)
        rescore(s, t, c)
    if after:
        if reuse:
            model, model_id = load(str(saved), args.attend_all), f"loaded {saved}"
        else:
            train_rows = [r for t in tasks for r in used[t]["train"]]
            val_rows = [r for t in tasks for r in used[t]["validation"]]
            report("train", {t: {i: sum(r.get("input", "real") == i for r in used[t]["train"])
                                 for i in ("real", "shuffled", "reversed")} for t in tasks})
            validation, epochs = [], []

            def epoch_done(record: dict) -> None:
                losses = validation[-len(val_rows):]
                by_task = {t: round(float(np.mean([v for v, r in zip(losses, val_rows) if r["task"] == t])), 4)
                           for t in tasks if any(r["task"] == t for r in val_rows)}
                validation.clear()
                epochs.append({**record, "val_loss_by_task": by_task})
                report("epochs", epochs)

            torch.manual_seed(args.seed)
            recorded_losses(model, out / "losses.jsonl", validation)
            started = time.time()
            train(model, examples(data, train_rows), examples(data, val_rows), epochs=args.epochs,
                  accumulation=8, seed=args.seed, log=epoch_done)
            model_id = f"trained {time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(started))} seed {args.seed}"
            best = (epochs[-1] if epochs[-1]["val_loss"] is None else min(epochs, key=lambda e: e["val_loss"]))["epoch"]
            report("training", {"seconds": round(time.time() - started), "best_epoch": best, "model": model_id})
            if args.save:
                OpenTSLM(model).save(str(saved))  # NOTE: before the scoring, so a crash there keeps the weights
        for s, t, c in after:
            predict(model, data, c, tested[s, t, c], files[s, t, c], model_id, swapped[s, t] if c == "swapped" else None)
            rescore(s, t, c)
    finish()
    if not metrics["complete"]:
        raise SystemExit("some predictions files are incomplete")


if __name__ == "__main__":
    main()
