"""Verify portable QA review exports; optional opaque local preservation hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/qa_v1'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def verify(local=False):
    provenance=json.loads((RESULTS/'provenance.json').read_text())
    for relative,entry in provenance.items():
        path=RESULTS/relative
        assert digest(path)==entry['published_sha256'],relative
        if path.suffix=='.png':assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        if local:
            if 'source' in entry:assert digest(ROOT/entry['source'])==entry['source_sha256']
            for src,h in entry.get('sources',{}).items():assert digest(ROOT/src)==h
    for name in ['README.md','docs/phase4_qa_v1.md','docs/phase4_qa_baselines.md','docs/phase4_qa_baselines_summary.md','results/qa_v1/MANIFEST.md']:
        path=ROOT/name
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in link:assert (path.parent/link.split('#')[0]).exists(),(name,link)
    summary=json.loads((RESULTS/'dataset_summary.json').read_text())
    assert summary['total']==19021
    expected={'train':'d63a0544e7ae3e3029d1af42c4a9a9f0d10308a8f4fcab7f29eecd816f298d41','val':'e873d36c89f86619bb29c98b19971e9b574490f48f9f37e94ad8b66c6adcf0df'}
    for split,h in expected.items():assert summary['splits'][split]['sha256']==h
    assert json.loads((RESULTS/'baseline_verification.json').read_text())['verdict']=='VERIFIED WITH QUALIFICATIONS'
    count=0
    if local:
        inventory=json.loads((ROOT/'outputs/phase4/qa_publication/preservation_before.json').read_text())
        for name,h in inventory.items():assert digest(ROOT/name)==h,name
        count=len(inventory)
        assert digest(ROOT/'outputs/phase3/cnn/frozen/baseline.pt')=='2e960150ef4820c038684e33102c0892c930c451f5d6904aa99aae542855e5d5'
        baseline=json.loads((ROOT/'outputs/phase4/qa_baselines_v1/preservation_before.json').read_text())
        assert digest(ROOT/'outputs/phase4/qa_publication/original_documents/README.md')==baseline['README.md']
    return {'status':'passed','portable_artifacts':len(provenance),'protected_local_files':count,
            'method':'SHA256 byte checks and link resolution; no tensor decoding, fitting or regeneration'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--local-artifacts',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.local_artifacts),indent=2))
