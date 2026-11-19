"""Read-only KernelSight v4 PyTorch records, preserving the official time grid."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'val', 'test')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _input_path(root, relative):
    path = (root / relative).resolve()
    _require(path.is_relative_to((root / 'kernels').resolve()) and
             path.name == 'tensor_input.npz' and path.parent.name == 'input',
             f'Invalid dataset input path: {relative}')
    return path


def read_splits(root):
    """Check all three official path lists without opening held-out tensors."""
    root = Path(root).resolve()
    result, seen = {}, set()
    for name in SPLITS:
        data = json.loads((root / f'splits/{name}.json').read_text())
        records = data['traces']
        paths = [str(_input_path(root, rec['path'])) for rec in records]
        _require(data['split'] == name and data['n'] == len(records), f'{name}: split/count mismatch')
        _require(len(paths) == len(set(paths)), f'{name}: duplicate trace paths')
        _require(not seen.intersection(paths), f'{name}: shared paths across official splits')
        _require(len(records) > 0, f'{name}: empty split')
        seen.update(paths)
        result[name] = records
    return result


def _load_input(path):
    with np.load(path, allow_pickle=True) as data:
        x = data['data'].copy()
        names = tuple(data['counter_names'].astype(str))
        edges = data['time_edges_ns'].copy()
        metadata = {key: data[key].copy() for key in
                    ('kernels', 'kernel_names', 'kernel_function_index') if key in data}
    _require(x.shape == (24, 512) and x.dtype == np.float32 and np.isfinite(x).all(),
             f'{path}: expected finite float32 [24,512] counters')
    _require(len(names) == 24 and len(set(names)) == 24, f'{path}: invalid channel names')
    _require(edges.shape == (513,) and edges.dtype == np.int64 and np.all(np.diff(edges) > 0),
             f'{path}: invalid int64 time edges')
    return x, names, edges, metadata


@dataclass(frozen=True)
class NormalizationStats:
    """Population mean/std per channel, fitted exclusively on all training bins."""
    channel_names: tuple
    mean: tuple
    std: tuple
    n_bins: int
    train_split_sha256: str
    train_inputs_sha256: dict
    method: str = 'train_all_bins_population_zscore_v1'
    zero_std_threshold: float = 1e-12

    def __post_init__(self):
        _require(self.method == 'train_all_bins_population_zscore_v1', 'Unknown normalization method')
        _require(self.zero_std_threshold == 1e-12, 'Unexpected zero-std threshold')
        _require(len(self.channel_names) == 24 and len(set(self.channel_names)) == 24,
                 'Statistics channel names must contain 24 unique entries')
        _require(len(self.mean) == len(self.std) == 24, 'Statistics must have 24 channels')
        _require(np.isfinite(self.mean).all() and np.isfinite(self.std).all() and
                 np.all(np.asarray(self.std) >= 0), 'Invalid statistics values')
        _require(self.n_bins > 0, 'Empty statistics')

    @property
    def scale(self):
        std = np.asarray(self.std, dtype=np.float64)
        return np.where(std > self.zero_std_threshold, std, 1.0)

    def validate_sources(self, root):
        records = read_splits(root)['train']
        _require(self.train_split_sha256 == sha256(root / 'splits/train.json'), 'Training split hash changed')
        _require(set(self.train_inputs_sha256) == {rec['path'] for rec in records},
                 'Statistics must reference exactly the official training inputs')
        _require(self.n_bins == len(records) * 512, 'Statistics bin count mismatch')
        for rec in records:
            _require(sha256(_input_path(root, rec['path'])) == self.train_inputs_sha256[rec['path']],
                     f'Training input hash changed: {rec["path"]}')

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        for key in ('channel_names', 'mean', 'std'):
            data[key] = tuple(data[key])
        return cls(**data)


def fit_training_statistics(root=DEFAULT_ROOT):
    """No split argument: fitting always reads official training inputs only.

    Float64 parallel-variance accumulation, population variance (ddof=0).
    All bins contribute equally, including unlabeled/inactive bins. No labels
    are read or used to choose the normalization population.
    """
    root = Path(root).resolve()
    records = read_splits(root)['train']
    n, mean, m2, names, hashes = 0, np.zeros(24), np.zeros(24), None, {}
    for rec in records:  # Preserve the official record order.
        path = _input_path(root, rec['path'])
        x, channels, _, _ = _load_input(path)
        if names is None:
            names = channels
        _require(channels == names, f'{path}: channel order mismatch')
        values = x.astype(np.float64)
        count = values.shape[1]
        block_mean = values.mean(axis=1)
        block_m2 = np.square(values - block_mean[:, None]).sum(axis=1)
        delta = block_mean - mean
        m2 += block_m2 + delta ** 2 * n * count / (n + count)
        mean += delta * count / (n + count)
        n += count
        hashes[rec['path']] = sha256(path)
    return NormalizationStats(names, tuple(mean), tuple(np.sqrt(np.maximum(m2 / n, 0))),
                              n, sha256(root / 'splits/train.json'), hashes)


class KernelSightDataset(Dataset):
    """Lazy, deterministic official-split records. Does not fit normalization.

    `split` accepts train/val/test or the corresponding official JSON path.
    Only `features` should enter a counter-only model. Metadata and targets can
    disclose operation identity. Each access returns independently owned CPU data.
    """
    def __init__(self, root=DEFAULT_ROOT, split='train', normalization=None):
        self.root = Path(root).resolve()
        split_arg = str(split)
        if split_arg in SPLITS:
            self.split = split_arg
        else:
            supplied = Path(split).resolve()
            matches = [name for name in SPLITS if supplied == (self.root / f'splits/{name}.json').resolve()]
            _require(len(matches) == 1, 'Use an official train/val/test split or its JSON path')
            self.split = matches[0]
        self._records = read_splits(self.root)[self.split]
        _require(normalization is None or isinstance(normalization, NormalizationStats),
                 'normalization must be training-fitted NormalizationStats or None')
        self.normalization = copy.deepcopy(normalization)
        if self.normalization is not None:
            self.normalization.validate_sources(self.root)

    @property
    def records(self):
        return copy.deepcopy(self._records)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        record = self._records[index]
        path = _input_path(self.root, record['path'])
        raw, channels, edges, input_metadata = _load_input(path)
        label_path = path.parent.parent / 'labels/labels.npz'
        with np.load(label_path, allow_pickle=True) as labels:
            lab = {key: labels[key].copy() for key in labels.files}
        _require(np.array_equal(edges, lab['time_edges_ns']), f'{path}: input/label timestamps differ')
        vocab = {level: tuple(lab[f'vocab_{level}'].astype(str)) for level in ('l1', 'l2')}
        _require(len(vocab['l1']) == 12 and len(vocab['l2']) == 73 and
                 all(len(set(v)) == len(v) for v in vocab.values()), f'{path}: invalid vocabularies')
        parents = lab['l2_parent_l1']
        _require(parents.shape == (73,) and np.issubdtype(parents.dtype, np.integer) and
                 np.all((parents >= 0) & (parents < 12)), f'{path}: invalid L2 parents')
        targets, masks = {}, {}
        for level, classes in (('l1', 12), ('l2', 73)):
            y, multi = lab[f'workload_{level}'], lab[f'workload_{level}_multihot']
            _require(y.shape == (512,) and np.issubdtype(y.dtype, np.integer) and
                     np.all((y >= -1) & (y < classes)), f'{path}: invalid {level} primary IDs')
            _require(multi.shape == (512, classes) and np.isin(multi, (0, 1)).all(),
                     f'{path}: invalid {level} multi-hot labels')
            valid = y >= 0
            _require(np.all(multi[np.flatnonzero(valid), y[valid]] == 1),
                     f'{path}: {level} primary class absent from multi-hot')
            targets[f'workload_{level}'] = torch.from_numpy(y.astype(np.int64))
            targets[f'workload_{level}_multihot'] = torch.from_numpy(multi.astype(np.float32))
            masks[f'valid_{level}'] = torch.from_numpy(valid)
            masks[f'multihot_{level}'] = torch.from_numpy(multi.any(axis=1))
        for key in ('mask_labeled', 'mask_any_kernel'):
            _require(lab[key].shape == (512,) and np.isin(lab[key], (0, 1)).all(), f'{path}: invalid {key}')
            masks[key] = torch.from_numpy(lab[key].astype(bool))
        _require(torch.equal(masks['mask_labeled'], masks['valid_l1']), f'{path}: primary mask mismatch')
        both = (lab['workload_l1'] >= 0) & (lab['workload_l2'] >= 0)
        _require(np.array_equal(parents[lab['workload_l2'][both]], lab['workload_l1'][both]),
                 f'{path}: L1/L2 primary hierarchy mismatch')
        # Multiple L2 children can map to one L1 class; compare presence by parent.
        derived_l1 = np.stack([lab['workload_l2_multihot'][:, parents == i].any(axis=1)
                               for i in range(12)], axis=1)
        _require(np.array_equal(derived_l1, lab['workload_l1_multihot'].astype(bool)),
                 f'{path}: multi-hot hierarchy mismatch')
        active = lab['workload_l1_multihot'].sum(axis=1).astype(np.int64)
        overlap = active >= 2
        _require(np.array_equal(active, lab['multihot_n_active']) and
                 lab['multihot_has_overlap'].shape == () and
                 lab['multihot_has_overlap'].item() in (0, 1) and
                 bool(lab['multihot_has_overlap'].item()) == bool(overlap.any()),
                 f'{path}: overlap counts/flag mismatch')
        targets.update(multihot_n_active=torch.from_numpy(active),
                       overlap_l1=torch.from_numpy(overlap),
                       multihot_has_overlap=torch.tensor(bool(overlap.any())))
        features = raw.copy()
        if self.normalization is not None:
            _require(channels == self.normalization.channel_names, f'{path}: statistics channel order mismatch')
            features = ((raw.astype(np.float64) - np.asarray(self.normalization.mean)[:, None]) /
                        self.normalization.scale[:, None]).astype(np.float32)
            _require(np.isfinite(features).all(), f'{path}: nonfinite normalized features')
        # Variable-length launch and segment fields remain per-record metadata.
        metadata = dict(path=record['path'], split=self.split, split_record=copy.deepcopy(record),
                        counter_names=channels, vocab_l1=vocab['l1'], vocab_l2=vocab['l2'],
                        l2_parent_l1=torch.from_numpy(parents.astype(np.int64)),
                        input=input_metadata,
                        segments={k: v for k, v in lab.items() if k.startswith('segment_') or k in
                                  ('attribute_flags', 'attribute_flag_names', 'spatial_state_vocab')},
                        normalization='none' if self.normalization is None else self.normalization.method)
        return dict(features=torch.from_numpy(features), raw_features=torch.from_numpy(raw),
                    time_edges_ns=torch.from_numpy(edges), targets=targets, masks=masks, metadata=metadata)


def collate_records(records):
    """Stack fixed tensors, retain variable metadata as a list, reject mixed schemas."""
    _require(bool(records), 'Cannot collate an empty batch')
    schema_keys = ('counter_names', 'vocab_l1', 'vocab_l2', 'split', 'normalization')
    first = records[0]['metadata']
    for record in records[1:]:
        meta = record['metadata']
        _require(all(meta[key] == first[key] for key in schema_keys) and
                 torch.equal(meta['l2_parent_l1'], first['l2_parent_l1']), 'Batch schema/order mismatch')
    batch = default_collate([{k: v for k, v in record.items() if k != 'metadata'} for record in records])
    batch['metadata'] = [record['metadata'] for record in records]
    return batch
