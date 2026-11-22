"""Build strict train/val TimeF views; never decode a test source or change the full release."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT / '.worktrees/timenet-kernelsight'
sys.path.insert(0, str(WORKTREE))

import numpy as np
from benchmarks.kernelsight.resource_guard import ResourceGuard
from timenet.registry import LocalRegistry
from timenet_connectors.datasets.williamhtan.kernelsight import KernelSightConnector
from timenet_connectors.datasets.williamhtan.kernelsight.source import local_refs, digest, SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    args = parser.parse_args()
    output = ROOT / 'outputs/phase3/task1/timef_views' / args.split
    output.mkdir(parents=True, exist_ok=False)
    (output / 'VIEW_PENDING.json').write_text('{"status":"partial research view; verification pending; never publish"}\n')
    refs = local_refs(ROOT / 'data/kernelsight_v4', (args.split,))
    allowed = {p.resolve() for ref in refs for p in ref.files().values()}
    before = {str(p): digest(p) for p in allowed}
    load = np.load

    def guarded(file, *arguments, **keywords):
        if Path(file).resolve() not in allowed:
            raise AssertionError('NPZ outside the selected permitted partition')
        return load(file, *arguments, **keywords)

    start = perf_counter()
    with patch.object(np, 'load', guarded), ResourceGuard(output, 3 * 1024**3) as monitor:
        monitor.phase = 'permitted_' + args.split + '_conversion'
        dataset = KernelSightConnector().convert(refs)
        monitor.phase = 'partition_only_timef_writer'
        LocalRegistry(output / 'registry').put_dataset(dataset)
        if any(digest(Path(p)) != expected for p, expected in before.items()):
            raise AssertionError('Source changed while building an isolated view')
    version = output / 'registry/williamhtan/kernelsight/1.0.0'
    files = {str(p.relative_to(version)): {'sha256': digest(p), 'bytes': p.stat().st_size}
             for p in version.rglob('*') if p.is_file()}
    result = {'status': 'built; integrity validation pending', 'kind': 'unpublished partition-only research view',
              'split': args.split, 'records': len(refs), 'source_revision': SOURCE_REVISION,
              'parent_full_release': str(ROOT / 'outputs/phase2/task3/writer/unvalidated-registry/williamhtan/kernelsight/1.0.0'),
              'source_npz_hashes': before, 'version_directory': str(version),
              'seconds': perf_counter() - start, 'compiled_bytes': sum(x['bytes'] for x in files.values()),
              'compiled_files': files, 'test_npz_decoded': False,
              'reason': 'Full artifact has validation/test shared row group 15; strict policy requires physical separation.'}
    (output / 'view.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
