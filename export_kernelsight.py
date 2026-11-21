"""Export the KernelSight TimeNet dataset to one .npz file for train_kernelsight.py and
flamingo/finetune_kernelsight.py.

Run with the TimeNet venv:
  cd ~/TimeNet && uv run python ~/hackathon/export_kernelsight.py ~/hackathon/kernelsight.npz
"""

import sys

import numpy as np
import timenet
from timenet.types import ClassificationTask

tn = timenet.TimeNet(sources=[".timenet-registry"])
with tn.open("williamhtan/kernelsight") as reader:
    dataset = reader.read()
    tasks = dataset.tasks_of(ClassificationTask)
    x = np.stack([np.stack([s.to_numpy() for s in t.inputs[0].signals]) for t in tasks]).astype(np.float32)
    y = np.array([t.targets[0].value for t in tasks])
    split = np.array([str(t.split) for t in tasks])
    paths = np.array([t.inputs[0].metadata["path"] for t in tasks])  # the release path names each Snapshot
    official_split = np.array([t.inputs[0].metadata["official_split"] for t in tasks])  # train/val/test
    names = {tuple(s.metadata["counter"] for s in t.inputs[0].signals) for t in tasks}
    assert len(names) == 1, "Channel order differs between Snapshots"
    channels = np.array(names.pop())
    label_sets = {tuple(label.name for label in t.label_set.labels) for t in tasks}
    assert len(label_sets) == 1, "Label sets differ between Snapshots"
    labels = np.array(label_sets.pop())

print(x.shape, dict(zip(*np.unique(split, return_counts=True))), dict(zip(*np.unique(y, return_counts=True))))
np.savez_compressed(sys.argv[1], x=x, y=y, split=split, channels=channels, labels=labels, paths=paths,
                    official_split=official_split)
