# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch>=2.6", "transformers>=4.57.3,<4.58", "numpy", "pandas", "datasets", "einops", "einops-exts",
#   "huggingface-hub", "tqdm", "accelerate", "open-flamingo==2.0.1",
# ]
# [tool.uv]
# # open-flamingo 2.0.1 pins torch==2.0.1 and sentencepiece==0.1.98; OpenTSLM's Flamingo code needs its 2.0 API
# # (vision_encoder.visual, generate(**kwargs)), so the pins are overridden instead of using 0.0.2.
# override-dependencies = ["torch>=2.6", "sentencepiece>=0.2.0"]
# ///
"""Does OpenTSLM's released HAR-CoT model explain what it sees? Run OpenTSLM/gemma-3-270m-pt-har-flamingo on test
windows of HAR-CoT with the time order intact, shuffled (one permutation of the 128 samples, the same for x, y, z)
and reversed. The text lines (per-axis mean and std) are the same in all three conditions, so only the order changes.

Scores per condition: answer accuracy (the authors' parser: the text after the last "Answer: "), and the share of
rationales that claim periodic/oscillating/rhythmic movement, order (then, followed by, ...), spikes/peaks and
variability. A shuffle destroys periodicity and order; a reversal keeps periodicity. A rationale that claims
"periodic" at the same rate on shuffled input does not describe the input. The input's own periodicity
(autocorrelation peak) is measured too, so the claim rate can be set against what is really there.

The model, the prompt and the generation call are OpenTSLM's own (HARCoTQADataset formatting, batch size 1, greedy,
as curriculum_learning.py evaluates). Nothing is written into the OpenTSLM repo (no data download, no bytecode).

  uv run usecases/explanation/har/run_har.py --csv <har_cot_test_cot.csv> --out runs/usecases/explanation/har
  uv run usecases/explanation/har/run_har.py --csv ... --out /tmp/x --limit 2 --stub   # code path, no model

Needs: the OpenTSLM repo (--opentslm, tested at commit 2968f4b), and access to the gated google/gemma-3-270m
(HF_TOKEN on a new machine). The run appends to outputs.jsonl, so a stopped run resumes.
NOTE: OpenTSLM's uv.lock pins open-flamingo 0.0.2, which cannot build OpenTSLMFlamingo (its vision_encoder is a
SimpleNamespace there). transformers 5.x loads Gemma in bfloat16 by default; <4.58 keeps float32.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

LABELS = ["biking", "lying", "running", "sitting", "standing", "walking", "walking_down", "walking_up"]
CLAIMS = {  # a rationale claims k if one of its sentences matches k and has no negation cue
    "periodic": r"\b(periodic\w*|oscillat\w*|rhythm\w*|cyclic\w*)",
    "order": r"\b(initially|followed by|then|thereafter|subsequently|afterwards?|before|after|later|eventually"
             r"|towards? the end|at the (beginning|start|end)|beginning|transition\w*)\b",
    "spike": r"\b(spike\w*|peak\w*)",
    "variability": r"\b(variab\w*|fluctuat\w*)",
}
# NOTE: "lacks clear periodic patterns" mentions periodicity but denies it; the model writes such sentences often.
NEGATION = r"\b(no|not|lacks?|lacking|without|absen\w*|rather than|instead of|neither|nor|minimal)\b"
REPO_ID, REVISION = "OpenTSLM/gemma-3-270m-pt-har-flamingo", "49feddd590dd74f6a05c76ce560b773f5cf9c224"
CONDITIONS = ("real", "shuffled", "reversed")
PERIODIC_ACF = 0.5  # an input window counts as periodic if one axis has autocorrelation >= this at a lag of 4-64


def load_windows(path: Path):
    """Valid test windows as har_order.py parses them: 3 x 128 floats and a label; broken rows are skipped."""
    csv.field_size_limit(10**9)
    xs, labels, rationales = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                x = np.array([json.loads(row[c]) for c in ("x_axis", "y_axis", "z_axis")], dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if x.shape != (3, 128) or row["label"] == "label":
                continue
            xs.append(x)
            labels.append(row["label"])
            rationales.append(row["rationale"])
    return np.stack(xs), labels, rationales


def stratified(labels: list[str], n: int, seed: int = 0) -> list[int]:
    """n // 8 windows of each class, drawn at random with a fixed seed, in file order."""
    rng = np.random.default_rng(seed)
    pick = []
    for label in LABELS:
        idx = [i for i, l in enumerate(labels) if l == label]
        pick += rng.choice(idx, size=min(n // len(LABELS), len(idx)), replace=False).tolist()
    return sorted(pick)


def transform(x: np.ndarray, condition: str, i: int) -> np.ndarray:
    if condition == "shuffled":  # one permutation per window, the same for x, y and z
        return x[:, np.random.default_rng([1, i]).permutation(x.shape[1])]
    if condition == "reversed":
        return x[:, ::-1]
    return x


def periodic_input(x: np.ndarray) -> bool:
    """Autocorrelation peak at a lag of 4-64 samples on any axis (z-scored)."""
    for v in x:
        v = (v - v.mean()) / (v.std() + 1e-9)
        acf = np.correlate(v, v, "full")[len(v) - 1:] / len(v)
        if acf[4:65].max() >= PERIODIC_ACF:
            return True
    return False


def parse_label(text: str) -> str | None:
    """The authors' parser (evaluation/opentslm/parse_predictions.py): the text after the last "Answer: ",
    without <eos>, lowercased, trailing punctuation stripped. None if it is not one of the 8 labels."""
    answer = text.split("Answer: ")[-1].strip() if "Answer: " in text else text
    answer = re.sub(r"<\|.*?\|>$", "", answer).replace("<eos>", "").strip().lower().rstrip(".,!?;:")
    return answer if answer in LABELS else None


def claims(text: str) -> dict:
    """Per claim type: "mention" if any sentence matches, "affirm" if a matching sentence has no negation cue."""
    sentences = re.split(r"(?<=[.!?])\s+", text.lower())
    return {k: {"mention": any(re.search(p, s) for s in sentences),
                "affirm": any(re.search(p, s) and not re.search(NEGATION, s) for s in sentences)}
            for k, p in CLAIMS.items()}


def claim_rates(texts: list[str]) -> dict:
    found = [claims(t) for t in texts]
    return {f"{k}_{how}": round(float(np.mean([c[k][how] for c in found])), 3)
            for k in CLAIMS for how in ("affirm", "mention")}


def score(rows: list[dict], periodic: dict) -> dict:
    """Per condition: answer accuracy, balanced accuracy, macro-F1, invalid share, claim rates, input periodicity,
    and against the real condition: answers that change and right/wrong flips."""
    out = {}
    real = {r["i"]: r for r in rows if r["condition"] == "real"}
    for cond in CONDITIONS:
        part = [r for r in rows if r["condition"] == cond]
        if not part:
            continue
        y, p = [r["label"] for r in part], [parse_label(r["output"]) for r in part]
        recall = [np.mean([a == b for a, b in zip(p, y) if b == c]) for c in LABELS if c in y]
        f1 = []
        for c in LABELS:
            tp = sum(a == c and b == c for a, b in zip(p, y))
            fp, fn = sum(a == c and b != c for a, b in zip(p, y)), sum(a != c and b == c for a, b in zip(p, y))
            if c in y:
                f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        texts = [r["output"] for r in part]
        per = [periodic[(r["i"], cond)] for r in part]
        said = [claims(t)["periodic"]["affirm"] for t in texts]
        res = {"n": len(part), "accuracy": round(float(np.mean([a == b for a, b in zip(p, y)])), 3),
               "balanced_accuracy": round(float(np.mean(recall)), 3), "macro_f1": round(float(np.mean(f1)), 3),
               "invalid": round(float(np.mean([a is None for a in p])), 3), "claim_rate": claim_rates(texts),
               "input_periodic_share": round(float(np.mean(per)), 3),
               "periodic_claims_on_periodic_input": round(float(np.mean([q for q, s in zip(per, said) if s])), 3)
               if any(said) else None,
               "mean_words": round(float(np.mean([len(t.split()) for t in texts])), 1),
               "seconds_per_window": round(float(np.mean([r["seconds"] for r in part])), 2)}
        if cond != "real" and real:
            pairs = [(real[r["i"]], r) for r in part if r["i"] in real]
            ok = lambda r: parse_label(r["output"]) == r["label"]
            res["vs_real"] = {"pairs": len(pairs),
                              "answer_changed": round(float(np.mean([parse_label(a["output"]) != parse_label(b["output"])
                                                                     for a, b in pairs])), 3),
                              "right_only_real": sum(ok(a) and not ok(b) for a, b in pairs),
                              "right_only_here": sum(ok(b) and not ok(a) for a, b in pairs)}
        out[cond] = res
    return out


def load_model(device: str):
    import torch
    from huggingface_hub import hf_hub_download
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

    assert tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2, 6), "torch.load must default to weights_only"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = hf_hub_download(REPO_ID, "model_checkpoint.pt", revision=REVISION)
    # As OpenTSLM.load_pretrained does for a "-flamingo" repo whose name starts with "gemma-3-270m".
    model = OpenTSLMFlamingo(device=device, llm_id="google/gemma-3-270m", cross_attn_every_n_layers=1,
                             gradient_checkpointing=False)
    model.load_from_file(checkpoint)
    model.eval()
    return model, device


def formatter(eos: str):
    """HARCoTQADataset._format_sample without the class's loader (it would download into the repo)."""
    from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

    dataset = HARCoTQADataset.__new__(HARCoTQADataset)  # no __init__: it loads and formats all splits
    dataset.EOS_TOKEN = eos

    def fmt(x: np.ndarray, label: str, rationale: str):
        row = {"x_axis": x[0].tolist(), "y_axis": x[1].tolist(), "z_axis": x[2].tolist(), "label": label,
               "rationale": rationale}
        return extend_time_series_to_match_patch_size_and_aggregate([dataset._format_sample(row)])[0]
    return fmt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True, help="har_cot_test_cot.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=400, help="windows, stratified by class (n // 8 each)")
    p.add_argument("--limit", type=int, default=0, help="only the first N selected windows (a test)")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--max-new-tokens", type=int, default=500, help="the HF demo uses 500")
    p.add_argument("--batch", type=int, default=1, help="the authors evaluate with 1; >1 is faster, left padding")
    p.add_argument("--device", default="auto")
    p.add_argument("--opentslm", type=Path, required=True)
    p.add_argument("--stub", action="store_true", help="no model: answer with the reference rationale")
    p.add_argument("--score-only", action="store_true", help="rescore the saved outputs")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    x, labels, rationales = load_windows(args.csv)
    sha = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    pick = stratified(labels, args.n)
    if args.limit:
        pick = pick[: args.limit]
    conditions = args.conditions.split(",")
    (args.out / "selection.json").write_text(json.dumps({"csv_sha256": sha, "valid_windows": len(labels),
                                                         "selected": pick}, indent=0))
    print(f"{len(labels)} valid windows, {len(pick)} selected, load {time.time() - t0:.0f}s", flush=True)

    rows_path = args.out / "outputs.jsonl"
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {(r["i"], r["condition"]) for r in rows}
    todo = [(i, c) for c in conditions for i in pick if (i, c) not in done]
    if todo and not args.score_only:
        if args.stub:
            model, device, fmt = None, "stub", None
        else:
            sys.dont_write_bytecode = True  # NOTE: no __pycache__ in the OpenTSLM repo
            sys.path.insert(0, str(args.opentslm / "src"))
            model, device = load_model(args.device)
            fmt = formatter(model.get_eos_token())
        print(f"device {device}, {len(todo)} generations to do", flush=True)
        with open(rows_path, "a") as f:
            for start in range(0, len(todo), args.batch):
                chunk = todo[start: start + args.batch]
                t = time.time()
                if model is None:
                    outputs = [rationales[i] for i, _ in chunk]
                else:
                    batch = [fmt(transform(x[i], c, i), labels[i], rationales[i]) for i, c in chunk]
                    outputs = model.generate(batch, max_new_tokens=args.max_new_tokens)
                seconds = (time.time() - t) / len(chunk)
                for (i, c), output in zip(chunk, outputs):
                    row = {"i": i, "condition": c, "label": labels[i], "output": output, "seconds": round(seconds, 2),
                           "batch": args.batch, "device": device}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps({"done": start + len(chunk), "of": len(todo), "i": chunk[-1][0],
                                  "condition": chunk[-1][1], "s": round(seconds, 1),
                                  "pred": parse_label(outputs[-1])}), flush=True)

    periodic = {(i, c): periodic_input(transform(x[i], c, i)) for i in pick for c in CONDITIONS}
    metrics = {"model": f"{REPO_ID}@{REVISION}", "csv_sha256": sha, "windows": len(pick),
               "stub": args.stub, "by_condition": score([r for r in rows if r["i"] in set(pick)], periodic),
               "reference_rationales": {"selected": claim_rates([rationales[i] for i in pick]),
                                        "all_test": claim_rates(rationales)},
               "claim_regex": CLAIMS, "periodic_acf_threshold": PERIODIC_ACF}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics["by_condition"], indent=1))
    print("reference", metrics["reference_rationales"])


if __name__ == "__main__":
    main()
