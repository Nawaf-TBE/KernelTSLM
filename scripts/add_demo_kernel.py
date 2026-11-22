"""Add one test Snapshot to the site's demo data (site/demo-data.js), then rebuild site/live-data.js.

  uv run --with numpy python scripts/add_demo_kernel.py <path substring> "<name>" "<detail>"

heat: 24 counters x 256 cells (mean of bin pairs), scaled 0-255 by the train 1st-99th percentile per counter,
in the site's channel order. Reference answers come from runs/usecases/<task>/test.jsonl; the model's from
runs/mt_all/predictions_<task>_real.jsonl.
"""
import base64, json, re, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "site/demo-data.js"


def rows(p):
    return [json.loads(l) for l in (ROOT / p).read_text().splitlines() if l.strip()]


def one(task, path, pred=False):
    src = f"runs/mt_all/predictions_{task}_real.jsonl" if pred else f"runs/usecases/{task}/test.jsonl"
    return [r for r in rows(src) if r["path"] == path]


def main(sub, name, detail):
    text = DEMO.read_text()
    head, body = text[:text.index("{")], text[text.index("{"):text.rindex("}") + 1]
    kx = json.loads(body)
    z = np.load(ROOT / "kernelsight.npz", allow_pickle=True)
    paths = [str(p) for p in z["paths"]]
    path = next(p for p in paths if sub in p and z["split"][paths.index(p)] == "test")
    if any(k["path"] == path for k in kx["kernels"]):
        sys.exit(f"already in demo data: {path}")
    order = [[str(c) for c in z["channels"]].index(c) for c in kx["channels"]]
    train = z["x"][z["split"] == "train"][:, order, :]
    lo, hi = np.percentile(train, 1, axis=(0, 2)), np.percentile(train, 99, axis=(0, 2))
    v = z["x"][paths.index(path)][order].reshape(24, 256, 2).mean(2)
    heat = np.round(np.clip((v - lo[:, None]) / np.maximum(hi - lo, 1e-12)[:, None], 0, 1) * 255).astype(np.uint8)

    diag, seg, cls, tmp = one("diagnosis", path)[0], one("segmentation", path)[0], one("classification", path)[0], one("temporal", path)
    p_diag, p_cls, p_tmp = one("diagnosis", path, True)[0], one("classification", path, True)[0], one("temporal", path, True)
    lim = lambda s: (re.search(r"Limiter:\s*([^.]+)\.", s) or [None, None])[1]
    m = re.match(r"^(.*?)\s*Limiter:\s*([^.]+)\.\s*(?:Try:\s*(.*))?$", diag["answer"])
    findings = [f.strip().rstrip(".") + "." for f in m.group(1).split(". ") if f.strip(". ")]
    t0 = tmp[0]
    q = t0["prompt"].split("Bin 511 is the end. ")[-1].split(" Answer with")[0]
    pt = next(r for r in p_tmp if r["id"] == t0["id"])
    kx["kernels"].append({
        "id": re.sub(r"^kernels/[^/]+/_out/|/input/tensor_input\.npz$", "", path).replace("/", "_"),
        "path": path, "name": name, "detail": detail, "workload": cls["answer"], "limiter": diag["type"],
        "findings": findings, "try": m.group(3) or "", "timeline": seg["answer"], "timing": {"q": q, "a": t0["answer"]},
        "heat": base64.b64encode(heat.tobytes()).decode(),
        "model": {"diagnosis": p_diag["output"], "diag_ok": lim(p_diag["output"]) == lim(diag["answer"]),
                  "timing": pt["output"], "timing_ok": pt["output"].strip() == t0["answer"],
                  "workload": p_cls["output"], "workload_ok": p_cls["output"].strip() == cls["answer"]},
    })
    DEMO.write_text(head + json.dumps(kx, separators=(",", ":")) + ";\n")
    print(f"added {name}: {path} (kernel index {len(kx['kernels']) - 1})")
    subprocess.run([sys.executable, "-I", str(ROOT / "scripts/build_live_data.py")], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
