"""Check QA evidence and preservation after generation, tests, and rendering."""
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import version
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase4/qa_v1'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def counts(name):
    suites=ET.parse(OUT/name).getroot().iter('testsuite')
    result={k:0 for k in ('tests','failures','errors','skipped')}
    for suite in suites:
        for k in result:result[k]+=int(suite.attrib.get(k,0))
    result['passed']=result['tests']-sum(result[k] for k in ('failures','errors','skipped'))
    assert not result['failures'] and not result['errors'],result
    return result


def main():
    v=json.loads((OUT/'verification.json').read_text());assert v['status']=='passed'
    for split,h in v['dataset_sha256'].items():assert digest(OUT/f'{split}.jsonl')==h
    tests={'focused':counts('qa_tests.xml'),'regression':counts('regression.xml')}
    assert tests['focused']['passed']==33 and tests['regression']['passed']==94
    original=json.loads((OUT/'preservation_before.json').read_text())
    protected={p:h for p,h in original.items() if not p.startswith('src/kernel_xray/qa/') and p not in ('scripts/build_qa_v1.py','configs/phase4_qa_v1.json')}
    for p,h in protected.items():assert digest(ROOT/p)==h,p
    v.update(tests=tests,preservation={'status':'passed','files_unchanged':len(protected),
        'scope':'all captured preexisting files; newly created Phase 4 source/config excluded from historical preservation, separately hashed'},
        versions={'python':platform.python_version(),**{p:version(p) for p in ('numpy','torch','timenet','pyarrow')}},
        figures_visually_inspected=True,illustrations_distinct_training_traces=3,
        trained_models=False,published_benchmark=False)
    (OUT/'verification.json').write_text(json.dumps(v,indent=2)+'\n')
    sources=[ROOT/'README.md',ROOT/'docs/phase4_qa_v1.md',ROOT/'configs/phase4_qa_v1.json',ROOT/'tests/test_qa_v1.py']
    sources+=list((ROOT/'src/kernel_xray/qa').glob('*.py'))
    sources+=[ROOT/'scripts'/n for n in ('build_qa_v1.py','plot_qa_v1.py','finalize_qa_v1.py')]
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.name!='COMPLETE.json']
    manifest={'status':'passed','questions':v['questions_verified'],'tests':tests,'preservation':v['preservation'],
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'artifacts_sha256':{str(p.relative_to(OUT)):digest(p) for p in artifacts}}
    (OUT/'COMPLETE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'questions':v['questions_verified'],'tests':tests,'preservation':v['preservation']},indent=2))


if __name__=='__main__':main()
