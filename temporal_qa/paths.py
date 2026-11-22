"""Data locations. kernelsight.npz sits at the repository root. The raw KernelSight release (per-Snapshot
labels/labels.npz, only needed to rebuild temporal_qa/labels_cache.npz and the diagnosis timing) is read from
$KERNELSIGHT_ROOT, by default data/kernelsight/ in the repository."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NPZ = str(REPO / "kernelsight.npz")
LABEL_ROOT = os.path.join(os.environ.get("KERNELSIGHT_ROOT", str(REPO / "data" / "kernelsight")), "")
