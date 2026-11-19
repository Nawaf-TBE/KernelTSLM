"""Masked fixed-bin views for conventional baselines; no model training."""
from __future__ import annotations

import torch


def flatten_labeled_bins(batch, level='l1'):
    """Return only valid fixed bins as [N,24] inputs, with reversible trace/bin indexes.

    Flatten after selecting official trace partitions. Zero rows are returned if
    the batch has no valid labels. Overlap/multi-hot tracks remain in the parent
    batch; primary classification is a separate supervised objective.
    """
    if level not in ('l1', 'l2'):
        raise ValueError('Select l1 or l2')
    y = batch['targets'][f'workload_{level}']
    mask = batch['masks'][f'valid_{level}']
    if y.ndim != 2 or mask.shape != y.shape or not torch.equal(mask, y >= 0):
        raise ValueError('Primary label/mask alignment mismatch')
    if batch['X'].shape != (y.shape[0], 24, y.shape[1]) or y.shape[1] != 512:
        raise ValueError('Expected trace-major [B,24,512] counters')
    positions = mask.nonzero(as_tuple=False)
    return {
        'X': batch['X'].permute(0, 2, 1)[mask].contiguous(), 'y': y[mask],
        'trace_index': positions[:, 0], 'bin_index': positions[:, 1],
        'metadata': batch['metadata'],
    }


def valid_primary_logits(logits, batch, level='l1'):
    """Select [N,C] logits and int64 targets from [B,C,512], without computing a loss."""
    classes = {'l1': 12, 'l2': 73}
    if level not in classes:
        raise ValueError('Select l1 or l2')
    y = batch['targets'][f'workload_{level}']
    mask = batch['masks'][f'valid_{level}']
    if logits.shape != (y.shape[0], classes[level], 512) or not torch.equal(mask, y >= 0):
        raise ValueError('Logit/primary label alignment mismatch')
    return logits.permute(0, 2, 1)[mask], y[mask]
