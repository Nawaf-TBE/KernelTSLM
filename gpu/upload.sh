#!/bin/bash
# From the Mac: send the code, the data, the question sets and the use-case rows and scorers to the GPU VM (~40 MB).
#   ~/hackathon/gpu/upload.sh <user>@<vm-ip>
set -e
HOST=${1:?usage: upload.sh <user>@<vm-ip>}
cd ~/hackathon
# NOTE: the segmentation scorer reads temporal_qa/labels_cache.npz; runs/usecases/_synthetic is RUN's dry-run data
COPYFILE_DISABLE=1 tar --no-xattrs --exclude __pycache__ --exclude .ruff_cache --exclude runs/usecases/_synthetic -czf - \
  kernelsight.npz runs/baselines runs/temporal_qa runs/usecases usecases temporal_qa/labels_cache.npz gpu \
  flamingo/finetune_kernelsight.py flamingo/pilot_temporal.py flamingo/multitask.py flamingo/utils.py \
  flamingo/requirements.txt flamingo/ask.py flamingo/walkthrough.py \
  | ssh "$HOST" 'mkdir -p hackathon && tar xzf - -C hackathon && chmod +x hackathon/gpu/*.sh && echo uploaded'
