"""Collect the recorded model answers for the six demo kernels into site/live-data.js."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "site/demo-data.js").read_text()
kx = json.loads(src[src.index("{"):src.rindex("}") + 1])
paths = {k["path"] for k in kx["kernels"]}

def rows(p):
    return [json.loads(l) for l in (ROOT / p).read_text().splitlines() if l.strip()]

prompts = {r["id"]: r for r in rows("runs/usecases/temporal/test.jsonl")}
out = {p: {"diagnosis": None, "segmentation": None, "classification": None, "temporal": []} for p in paths}
for task in ["diagnosis", "segmentation", "classification", "temporal"]:
    for r in rows(f"runs/mt_all/predictions_{task}_real.jsonl"):
        if r["path"] not in paths:
            continue
        a = {k: r[k] for k in ["answer", "output", "tokens", "seconds"]}
        if task == "temporal":
            q = prompts[r["id"]]
            a.update(type=r["type"], prompt=q["prompt"], options=q.get("options"), channels=q.get("channels"))
            out[r["path"]]["temporal"].append(a)
        elif out[r["path"]][task] is None:
            out[r["path"]][task] = a
(ROOT / "site/live-data.js").write_text(
    "// Recorded outputs of the fine-tuned model (runs/mt_all, seed 0, one H100) for the demo kernels.\n"
    "// Built by scripts/build_live_data.py.\nwindow.KX_LIVE=" + json.dumps(out) + ";\n")
for p, v in out.items():
    print(p.split("/")[-3], len(v["temporal"]), [t["type"] for t in v["temporal"]])
