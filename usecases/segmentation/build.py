# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Build the segmentation rows: one timeline question per Snapshot, in the row contract of flamingo/multitask.py.

The model reads all 24 Channels, with sass_misc moved last: the question and the answer read only the last
series (utils.py MaskedCrossAttention), and sass_misc alone gives the best busy/idle on validation (SPEC.md).
controls.reversed is the reversed timeline. controls.shuffled is the TRUE timeline on purpose: the truth of a
Bin-shuffled trace is noise that a 16-Bin grid cannot write, so the control asks whether the model still
writes the true timeline when the order is gone (an order-blind model keeps its score).

  uv run --with numpy python usecases/segmentation/build.py    # writes runs/usecases/segmentation/*.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import CLASSES, ROOT, W, _npz, states, timeline  # noqa: E402

LAST = "sass_misc"
CHANNELS = [str(c) for c in _npz["channels"] if c != LAST] + [LAST]
PROMPT = ("The trace has 512 Bins in time order: Bin 0 is the start of the trace and Bin 511 is the end. "
          f"Write the timeline of the GPU in steps of {W} Bins. Each step is idle, unlabeled (a kernel without "
          f"a workload label) or one workload class: {', '.join(CLASSES)}. List each change as "
          "'<first Bin> <state>', separated by semicolons, starting at Bin 0.")
MAX_NEW_TOKENS = 160  # longest answer: 140 Gemma 3 tokens (validation)


def main():
    out = ROOT / "runs/usecases/segmentation"
    out.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "validation": [], "test": []}
    for path, split in zip(_npz["paths"].tolist(), _npz["split"].tolist()):
        s = states(path)
        rows[split].append({
            "id": f"segmentation/{split}/{len(rows[split]):06d}", "task": "segmentation", "type": "timeline",
            "path": path, "channels": CHANNELS, "prompt": PROMPT, "options": [], "answer": timeline(s),
            "controls": {"reversed": timeline(s[::-1]), "shuffled": timeline(s)},
            "max_new_tokens": MAX_NEW_TOKENS})
    for split, part in rows.items():
        with open(out / f"{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in part)
        runs = Counter(r["answer"].count(";") + 1 for r in part)
        print(f"{split:<11} {len(part)} rows, runs per answer {dict(sorted(runs.items()))}")


if __name__ == "__main__":
    main()
