#!/usr/bin/env python3
"""Training-only PyTorch DataLoader example; no model or optimizer is created."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.utils.data import DataLoader

from kernel_xray import KernelSightDataset, NormalizationStats, collate_records, fit_training_statistics
from kernel_xray.dataset import DEFAULT_ROOT


def batch_example(root=DEFAULT_ROOT, stats=None, batch_size=4, seed=7, num_workers=0):
    dataset = KernelSightDataset(root, 'train', normalization=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        generator=torch.Generator().manual_seed(seed), collate_fn=collate_records)
    batch = next(iter(loader))
    size = min(batch_size, len(dataset))
    expected = dict(features=(size, 24, 512), raw_features=(size, 24, 512), time_edges_ns=(size, 513))
    for key, shape in expected.items():
        assert tuple(batch[key].shape) == shape
    for level, n_classes in [('l1', 12), ('l2', 73)]:
        assert tuple(batch['targets'][f'workload_{level}'].shape) == (size, 512)
        assert tuple(batch['targets'][f'workload_{level}_multihot'].shape) == (size, 512, n_classes)
    assert all(tuple(mask.shape) == (size, 512) for mask in batch['masks'].values())
    assert len(batch['metadata']) == size
    return dict(split='train', seed=seed, num_workers=num_workers, dataset_length=len(dataset),
                normalization='none' if stats is None else stats.method,
                batch_shapes={key: list(batch[key].shape) for key in expected},
                target_shapes={key: list(value.shape) for key, value in batch['targets'].items()},
                mask_shapes={key: list(value.shape) for key, value in batch['masks'].items()},
                metadata_records=len(batch['metadata']),
                example_paths=[meta['path'] for meta in batch['metadata']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fit-normalization', type=Path, help='Fit train-only statistics and save JSON here')
    group.add_argument('--stats', type=Path, help='Load a previously fitted training statistics JSON')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch-size must be positive and num-workers nonnegative')
    stats = None
    if args.fit_normalization is not None:
        destination = args.fit_normalization.resolve()
        # CLI writes must stay in a designated Task 7 derived-artifact directory.
        allowed = (DEFAULT_ROOT / 'outputs/phase1/task7').resolve()
        if not destination.is_relative_to(allowed):
            parser.error('--fit-normalization must be inside outputs/phase1/task7/')
        stats = fit_training_statistics(args.data_root)
        stats.save(destination)
    elif args.stats is not None:
        stats = NormalizationStats.load(args.stats)
    print(json.dumps(batch_example(args.data_root, stats, args.batch_size, args.seed, args.num_workers), indent=2))


if __name__ == '__main__':
    main()
