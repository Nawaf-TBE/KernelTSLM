import copy
import json
from pathlib import Path

import numpy as np
import pytest

from kernel_xray.dataset import DEFAULT_ROOT, KernelSightDataset, fit_training_statistics, read_splits


@pytest.fixture(autouse=True)
def protect_heldout_tensors(monkeypatch):
    """Every test fails if it opens a real validation/test NPZ."""
    allowed = {(DEFAULT_ROOT / r['path']).resolve().parent.parent
               for r in read_splits(DEFAULT_ROOT)['train']}
    original = np.load

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            path = Path(file).resolve()
            if path.is_relative_to((DEFAULT_ROOT / 'kernels').resolve()):
                assert path.parent.parent in allowed, f'Held-out tensor access: {path}'
        return original(file, *args, **kwargs)

    monkeypatch.setattr(np, 'load', guarded)


@pytest.fixture(scope='session')
def train():
    return KernelSightDataset()


@pytest.fixture
def synthetic_root(tmp_path):
    """Small independent split fixtures; labels copied only from a training trace."""
    rec = read_splits(DEFAULT_ROOT)['train'][0]
    source = DEFAULT_ROOT / rec['path']
    with np.load(source, allow_pickle=True) as data:
        inp = {k: data[k].copy() for k in data.files}
    with np.load(source.parent.parent / 'labels/labels.npz', allow_pickle=True) as data:
        lab = {k: data[k].copy() for k in data.files}
    (tmp_path / 'splits').mkdir()
    for name, values in [('train', [0., 4.]), ('val', [1000.]), ('test', [-1000.])]:
        records = []
        for i, value in enumerate(values):
            path = tmp_path / f'kernels/fixture/_out/{name}{i}/input/tensor_input.npz'
            path.parent.mkdir(parents=True)
            x = copy.deepcopy(inp)
            x['data'] = np.zeros((24, 512), dtype=np.float32)
            x['data'][0] = value
            x['data'][1] = 2 * value
            np.savez(path, **x)
            label = path.parent.parent / 'labels/labels.npz'
            label.parent.mkdir()
            np.savez(label, **lab)
            records.append(dict(rec, path=str(path.relative_to(tmp_path))))
        (tmp_path / f'splits/{name}.json').write_text(json.dumps(dict(split=name, n=len(records), traces=records)))
    return tmp_path


@pytest.fixture
def synthetic_stats(synthetic_root):
    return fit_training_statistics(synthetic_root)
