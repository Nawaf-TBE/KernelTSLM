"""Run the historical verification without overwriting historical evidence."""
from pathlib import Path
import runpy
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'outputs/phase3/cnn'
AUDIT=ROOT/'outputs/phase3/cnn_audit/legacy_verification'
write=Path.write_text


def redirected(path, data, *args, **kwargs):
    resolved=path.resolve()
    if resolved.is_relative_to(ORIGINAL):
        target=AUDIT/resolved.relative_to(ORIGINAL)
        target.parent.mkdir(parents=True,exist_ok=True)
        return write(target,data,*args,**kwargs)
    raise RuntimeError(f'Historical verifier attempted an unexpected write: {path}')


if __name__=='__main__':
    with patch.object(Path,'write_text',redirected):
        runpy.run_path(str(ROOT/'scripts/verify_cnn_artifacts.py'),run_name='__main__')
