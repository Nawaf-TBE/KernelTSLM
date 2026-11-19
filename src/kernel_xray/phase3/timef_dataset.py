"""Lazy counter-only PyTorch traces from the validated KernelSight TimeF release."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from kernel_xray.dataset import DEFAULT_ROOT, NormalizationStats, read_splits, sha256

FULL_VERSION = DEFAULT_ROOT / 'outputs/phase2/task3/writer/unvalidated-registry/williamhtan/kernelsight/1.0.0'
VIEW_ROOT = DEFAULT_ROOT / 'outputs/phase3/task1/timef_views'
DEFAULT_VERSIONS = {split: VIEW_ROOT / split / 'registry/williamhtan/kernelsight/1.0.0'
                    for split in ['train', 'val']}
DEFAULT_VERSION = DEFAULT_VERSIONS['train']
DEFAULT_STATISTICS = DEFAULT_ROOT / 'outputs/phase1/task7/train_statistics.json'
NATIVE_SPLITS = {'train': 'train', 'val': 'validation', 'test': 'test'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TraceIdentity:
    record_id: int
    path: str
    split: str
    split_index: int


def trace_index(version, root, split):
    """Index identities only; filter native split IDs before reading any Record payload."""
    from timenet.format.duckdb import check_control_schema, connect_control
    from timenet_connectors.datasets.williamhtan.kernelsight.source import pinned_manifest

    official = read_splits(root)
    manifest = pinned_manifest()
    for name in NATIVE_SPLITS:
        require(sha256(root / f'splits/{name}.json') == manifest[f'splits/{name}.json'],
                'Official split differs from the pinned source')
    with connect_control(version / 'control.duckdb', read_only=True) as connection:
        require(check_control_schema(connection) == 2, 'Expected TimeF control schema 2')
        splits = {row[0] for row in connection.execute('SELECT DISTINCT split FROM tasks').fetchall()}
        require(splits == {NATIVE_SPLITS[split]},
                'Strict isolation requires a separate single-partition TimeF view')
        require(connection.execute('SELECT count(*) FROM records').fetchone()[0] == len(official[split]),
                'Unexpected Records in partition view')
        require(connection.execute('SELECT count(*) FROM signal_chunks WHERE source_key IS NOT NULL').fetchone()[0] == 0,
                'Strict views cannot reuse external mixed-partition payloads')
        cross = connection.execute('''
            SELECT rr.record_id FROM task_record_refs rr JOIN tasks t USING (task_id)
            WHERE rr.field = 'inputs'
            GROUP BY rr.record_id HAVING count(DISTINCT t.split) <> 1
        ''').fetchall()
        require(not cross, 'Cross-partition Record references')
        ids = [row[0] for row in connection.execute('''
            SELECT DISTINCT rr.record_id FROM task_record_refs rr JOIN tasks t USING (task_id)
            WHERE rr.field = 'inputs' AND t.split = ?
        ''', [NATIVE_SPLITS[split]]).fetchall()]
        require(len(ids) == len(official[split]), 'Stored split Record count mismatch')
        # Every encoded counter row must belong to this view. File headers are
        # inspected without decoding any values, including unused row groups.
        import pyarrow.parquet as pq
        groups = connection.execute("""
            SELECT value_path, chunk_major_index, list(chunk_minor_index)
            FROM signal_chunks GROUP BY 1, 2
        """).fetchall()
        by_file = {}
        for path, group, minor in groups:
            by_file.setdefault(path, {})[group] = set(minor)
        files = {str(p.relative_to(version)) for p in (version / 'time_series').glob('*.parquet')}
        require(files == set(by_file), 'Unreferenced counter files in strict view')
        for path, group_rows in by_file.items():
            with pq.ParquetFile(version / path) as parquet:
                require(set(group_rows) == set(range(parquet.num_row_groups)), 'Unreferenced row group')
                require(all(rows == set(range(parquet.metadata.row_group(group).num_rows))
                            for group, rows in group_rows.items()), 'Unreferenced counter rows')
        rows = []
        # Parse canonical JSON only for a bounded set of selected identities.
        # A multi-path extraction parses each metadata value once; arrays never
        # become Python objects here, and no test Record is requested.
        for start in range(0, len(ids), 16):
            selected = ids[start:start + 16]
            chunk = connection.execute("""
                SELECT record_id, json_extract_string(metadata, [
                    '$.kernelsight_v4.path', '$.kernelsight_v4.split',
                    '$.kernelsight_v4.split_index'])
                FROM records WHERE record_id IN (SELECT unnest(?))
            """, [selected]).fetchall()
            rows.extend((record_id, fields[0], fields[1], int(fields[2]))
                        for record_id, fields in chunk)
    by_path = {path: (record_id, stored_split, index) for record_id, path, stored_split, index in rows}
    require(len(by_path) == len(rows) == len(official[split]), 'Duplicate/missing stored paths')
    result = []
    for index, source in enumerate(official[split]):
        require(source['path'] in by_path, 'Official trace missing from release')
        record_id, stored_split, stored_index = by_path[source['path']]
        require(stored_split == NATIVE_SPLITS[split] and stored_index == index, 'Record split/order mismatch')
        result.append(TraceIdentity(record_id, source['path'], split, index))
    return tuple(result)


def record_to_sample(record, identity, normalization=None):
    """Extract counter tensors and dense supervision separately, with no task expansion."""
    from timenet.dataset.axis import IrregularAxis
    from timenet_connectors.datasets.williamhtan.kernelsight.mapping import (
        canonical_payload, channel_index, decode_array, model_features, quantized_edges,
    )
    from timenet_connectors.datasets.williamhtan.kernelsight.source import SOURCE_REVISION

    payload = canonical_payload(record)
    require(payload['schema_version'] == 1 and payload['source_revision'] == SOURCE_REVISION,
            'Unexpected canonical schema/source revision')
    require(payload['path'] == identity.path and payload['split'] == NATIVE_SPLITS[identity.split]
            and payload['split_index'] == identity.split_index, 'Trace identity/split mismatch')
    arrays = payload['arrays']
    inp, labels = arrays['input'], arrays['labels']
    channels = tuple(decode_array(inp['counter_names']).astype(str))
    raw = model_features(record)['features']
    require(raw.shape == (24, 512) and raw.dtype == np.float32 and np.isfinite(raw).all(), 'Counter schema')
    ordered = sorted(record.signals, key=channel_index)
    require(tuple(s.name for s in ordered) == channels, 'Counter channel names/order')
    edges = decode_array(inp['time_edges_ns'])
    require(np.array_equal(edges, decode_array(labels['time_edges_ns'])), 'Input/label time alignment')
    q = quantized_edges(edges)
    for signal in ordered:
        require(isinstance(signal.time_axis, IrregularAxis), 'Expected irregular counter axis')
        require(np.array_equal(signal.time_offsets_us(), q[:-1]), 'Counter temporal order')
    require(record.time_span.start_us == 0 and record.time_span.end_us == int(q[-1]), 'Final bin extent')
    targets, masks, vocab = {}, {}, {}
    decoded_multi = {}
    for level, classes in [('l1', 12), ('l2', 73)]:
        y = decode_array(labels[f'workload_{level}'])
        multi = decode_array(labels[f'workload_{level}_multihot'])
        vocab[level] = tuple(decode_array(labels[f'vocab_{level}']).astype(str))
        require(len(vocab[level]) == len(set(vocab[level])) == classes, 'Vocabulary order/size')
        require(y.dtype == np.int32 and y.shape == (512,) and ((y >= -1) & (y < classes)).all(),
                'Primary class IDs')
        require(multi.dtype == np.uint8 and multi.shape == (512, classes) and np.isin(multi, [0, 1]).all(),
                'Multi-hot schema')
        valid = y >= 0
        require((multi[np.flatnonzero(valid), y[valid]] == 1).all(), 'Primary not in multi-hot')
        targets[f'workload_{level}'] = torch.from_numpy(y.astype(np.int64))
        targets[f'workload_{level}_multihot'] = torch.from_numpy(multi.astype(np.float32))
        masks[f'valid_{level}'] = torch.from_numpy(valid)
        masks[f'multihot_{level}'] = torch.from_numpy(multi.any(axis=1))
        decoded_multi[level] = multi
    for name in ['mask_labeled', 'mask_any_kernel']:
        value = decode_array(labels[name])
        require(value.shape == (512,) and value.dtype == np.uint8 and np.isin(value, [0, 1]).all(), 'Source mask')
        masks[name] = torch.from_numpy(value.astype(bool))
    require(torch.equal(masks['mask_labeled'], masks['valid_l1']), 'L1 supervision mask mismatch')
    parents = decode_array(labels['l2_parent_l1'])
    require(parents.shape == (73,) and ((parents >= 0) & (parents < 12)).all(), 'Parent vocabulary')
    derived = np.stack([decoded_multi['l2'][:, parents == i].any(axis=1) for i in range(12)], axis=1)
    require(np.array_equal(derived, decoded_multi['l1']), 'Multi-hot hierarchy mismatch')
    both = masks['valid_l1'].numpy() & masks['valid_l2'].numpy()
    require(np.array_equal(parents[targets['workload_l2'].numpy()[both]],
                           targets['workload_l1'].numpy()[both]), 'Primary hierarchy mismatch')
    active = decoded_multi['l1'].sum(axis=1).astype(np.int64)
    overlap = active >= 2
    flag = decode_array(labels['multihot_has_overlap'])
    require(np.array_equal(active, decode_array(labels['multihot_n_active'])) and flag.shape == ()
            and flag.item() in (0, 1) and bool(flag.item()) == bool(overlap.any()), 'Overlap semantics')
    targets.update(multihot_n_active=torch.from_numpy(active), overlap_l1=torch.from_numpy(overlap),
                   multihot_has_overlap=torch.tensor(bool(overlap.any())))
    # An observed-membership coverage mask, not proof of exhaustive negative supervision.
    masks['overlap_l1_valid'] = masks['multihot_l1'].clone()
    features = raw.copy()
    if normalization is not None:
        require(channels == normalization.channel_names, 'Statistics channel ordering')
        features = ((raw.astype(np.float64) - np.asarray(normalization.mean)[:, None]) /
                    normalization.scale[:, None]).astype(np.float32)
        require(np.isfinite(features).all(), 'Nonfinite normalized counters')
    return {
        'X': torch.from_numpy(features), 'raw_X': torch.from_numpy(raw.copy()),
        'time_edges_ns': torch.from_numpy(edges.copy()), 'targets': targets, 'masks': masks,
        'metadata': {'trace_id': identity.path, 'record_id': identity.record_id,
                     'split': identity.split, 'split_index': identity.split_index,
                     'counter_names': channels, 'vocab_l1': vocab['l1'], 'vocab_l2': vocab['l2'],
                     'l2_parent_l1': tuple(int(x) for x in parents),
                     'normalization': 'none' if normalization is None else normalization.method},
    }


class TimeFTraceDataset(Dataset):
    """Lazy trace units in official order, with independent readers per spawned worker.

    Test decoding is disabled unless a future evaluation caller explicitly opts in.
    This opt-in is not authorization to develop against test data.
    """
    def __init__(self, split='train', *, version=None, root=DEFAULT_ROOT,
                 normalization=DEFAULT_STATISTICS, allow_test=False):
        require(split in NATIVE_SPLITS, 'Select official train, val or test')
        require(split != 'test' or allow_test, 'Test arrays are reserved for final evaluation')
        if version is None:
            require(split in DEFAULT_VERSIONS, 'Test evaluation requires an explicitly supplied isolated view')
            version = DEFAULT_VERSIONS[split]
        self.version, self.root, self.split = Path(version).resolve(), Path(root).resolve(), split
        self.identities = trace_index(self.version, self.root, split)
        self.normalization = normalization
        if isinstance(normalization, (str, Path)):
            self.normalization = NormalizationStats.load(normalization)
        require(self.normalization is None or isinstance(self.normalization, NormalizationStats),
                'Use verified training statistics or None; no implicit fitting')
        if self.normalization is not None:
            self.normalization.validate_sources(self.root)
            self.normalization = copy.deepcopy(self.normalization)
        self._reader, self._pid = None, None

    def __len__(self):
        return len(self.identities)

    def _reader_for_process(self):
        from timenet.reader import TimeFReader
        require(self._reader is None or self._pid == os.getpid(), 'Use spawned DataLoader workers')
        if self._reader is None:
            self._reader = TimeFReader.open(self.version)
            self._pid = os.getpid()
        return self._reader

    def __getitems__(self, indices):
        from timenet.dataset import Record
        from timenet.reader import LoadPlan
        from timenet.types.load_plan import Loading
        selected = [self.identities[index] for index in indices]
        records = self._reader_for_process().get_many(
            Record, [item.record_id for item in selected],
            load=LoadPlan(signal_values=Loading.EAGER, time_offsets=Loading.EAGER),
        )
        return [record_to_sample(record, item, self.normalization)
                for record, item in zip(records, selected, strict=True)]

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_reader'], state['_pid'] = None, None
        return state

    def close(self):
        if self._reader is not None:
            self._reader.close()
        self._reader, self._pid = None, None


def collate_traces(samples):
    """Stack tensor fields; keep identity/vocabulary bookkeeping outside model X."""
    require(bool(samples), 'Cannot collate an empty batch')
    schema = ['split', 'counter_names', 'vocab_l1', 'vocab_l2', 'l2_parent_l1', 'normalization']
    first = samples[0]['metadata']
    require(all(all(sample['metadata'][key] == first[key] for key in schema) for sample in samples),
            'Mixed split/channel/vocabulary/normalization batch')
    result = default_collate([{k: v for k, v in sample.items() if k != 'metadata'} for sample in samples])
    result['metadata'] = [sample['metadata'] for sample in samples]
    return result


def seed_worker(worker_id):
    """Derive Python/NumPy worker RNGs from the reproducible PyTorch worker seed."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataloaders(*, batch_size=16, seed=7, num_workers=0, shuffle_train=True,
                     version=None, root=DEFAULT_ROOT, normalization=DEFAULT_STATISTICS):
    """Construct deterministic train/validation loaders without creating any test items."""
    loaders = {}
    for split in ['train', 'val']:
        dataset = TimeFTraceDataset(split, version=version, root=root, normalization=normalization)
        options = {'multiprocessing_context': 'spawn'} if num_workers else {}
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=split == 'train' and shuffle_train,
            generator=torch.Generator().manual_seed(seed), num_workers=num_workers,
            worker_init_fn=seed_worker, collate_fn=collate_traces, drop_last=False,
            persistent_workers=False, **options,
        )
    return loaders
