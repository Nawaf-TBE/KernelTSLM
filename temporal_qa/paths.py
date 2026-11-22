"""Data locations and per-Bin labels.

kernelsight.npz sits at the repository root; labels_cache.npz holds its per-Bin workload_l1 and mask_any_kernel in
the same Snapshot order. The raw KernelSight release is only needed to rebuild that cache and for the kernel timing
in usecases/diagnosis/build.py; it is read from $KERNELSIGHT_ROOT, by default data/kernelsight_v4/.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NPZ = str(REPO / "kernelsight.npz")
CACHE = REPO / "temporal_qa" / "labels_cache.npz"
LABEL_ROOT = os.path.join(os.environ.get("KERNELSIGHT_ROOT", str(REPO / "data" / "kernelsight_v4")), "")


@lru_cache(maxsize=1)
def _labels() -> dict:
    with np.load(CACHE) as cache:
        return {key: cache[key] for key in cache.files}


def labels(i: int) -> dict:
    """workload_l1 and mask_any_kernel of Snapshot i (kernelsight.npz order)."""
    return {key: value[i] for key, value in _labels().items()}
