"""Read-only, standard-library artifact verification; no dataset/model execution."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
# Avoid kernel_xray.__init__, whose legacy convenience imports require torch.
import importlib.util
spec=importlib.util.spec_from_file_location('qa_portable_standalone',ROOT/'src/kernel_xray/qa_baselines/portable.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
PortableModel=module.PortableModel
R=ROOT/'results/qa_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def metrics(rows,classes=None):
    y=[r['reference'] for r in rows];p=[r['prediction'] for r in rows];n=len(y)
    if classes is None:
        error=sorted(abs(a-b) for a,b in zip(y,p));cfg=read(ROOT/'configs/phase4_qa_v1.json')
        return {'n':n,'mae':math.fsum(error)/n,'median_absolute_error':(error[(n-1)//2]+error[n//2])/2,
            'rmse':math.sqrt(math.fsum(e*e for e in error)/n),'within_tolerance':sum(abs(a-b)<=cfg['answer_atol']+cfg['answer_rtol']*abs(a) for a,b in zip(y,p))/n}
    index={c:i for i,c in enumerate(classes)};cm=[[0]*len(classes) for c in classes]
    for a,b in zip(y,p):cm[index[a]][index[b]]+=1
    per={}
    for i,c in enumerate(classes):
        tp=cm[i][i];support=sum(cm[i]);positive=sum(row[i] for row in cm)
        per[c]={'precision':tp/positive if positive else 0.,'recall':tp/support if support else 0.,'f1':2*tp/(support+positive) if support+positive else 0.,'support':support}
    return {'n':n,'classes':classes,'accuracy':sum(cm[i][i] for i in range(len(classes)))/n,
        'balanced_accuracy':math.fsum(d['recall'] for d in per.values())/len(classes),
        'macro_f1':math.fsum(d['f1'] for d in per.values())/len(classes),'confusion_matrix':cm,'per_answer':per}


def equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(int,float)):assert math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
    else:assert a==b,(a,b)


def verify():
    manifest=read(R/'reproducibility_manifest.json')
    for name,h in manifest['files'].items():assert sha(ROOT/name)==h,name
    for p in (ROOT/'models/qa_v1').iterdir():
        assert p.suffix=='.json' and str(p.relative_to(ROOT)) in manifest['files']
        PortableModel.load(p)
    index=read(R/'question_index.json');assert index['split']=='val'
    assert index['dataset_sha256']=='e873d36c89f86619bb29c98b19971e9b574490f48f9f37e94ad8b66c6adcf0df'
    assert sum(map(len,index['by_type'].values()))==2369
    predictions=read(R/'predictions.json');reference=read(R/'condition_metrics.json')
    assert set(predictions)==set(reference);aligned={}
    for key,rows in predictions.items():
        kind,model,condition=key.split('/')
        assert model in ('majority','lookup','primary','aware','direct')
        expected=set(index['by_type'][kind]);seen=set()
        for row in rows:
            assert set(row)=={'id','reference','prediction'}
            assert len(row['id'])==64 and all(c in '0123456789abcdef' for c in row['id'])
            assert row['id'] not in seen;seen.add(row['id'])
            target=(row['id'],'altered' if condition=='reversed_altered' else 'original')
            if target in aligned:assert aligned[target]==row['reference']
            aligned[target]=row['reference']
            if kind=='statistics':assert all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('reference','prediction'))
            else:assert row['prediction'] in reference[key]['classes']
        assert seen==expected
        equal(metrics(rows,None if kind=='statistics' else reference[key]['classes']),reference[key])
    for f in ('categorical_metrics.json','statistics_metrics.json'):
        for key,v in read(R/f).items():equal(v,reference[key])
    for path in (ROOT/'benchmarks/qa_v1').glob('*.jsonl'):
        assert sha(path)==manifest['qa_sha256'][path.stem]
        for line in path.read_text().splitlines():
            row=json.loads(line);assert row['split']==path.stem and set(row['channels'])<=set(range(24)) and 0<=row['start']<row['end']<=512
    assert not list((ROOT/'models/qa_v1').glob('*.pkl'))
    import subprocess
    tracked=subprocess.run(['git','-C',str(ROOT),'ls-files','-z'],capture_output=True,text=True)
    assert tracked.returncode==0,'Verification expects the published Git checkout'
    for name in filter(None,tracked.stdout.split('\0')):
        assert not name.startswith(('data/','outputs/','.worktrees/','.cache/','.venv/')),name
        assert Path(name).suffix not in ('.pkl','.pickle','.joblib','.npz','.npy','.pt','.pth','.parquet','.jsonl'),name
    return {'verdict':'VERIFIED WITH QUALIFICATIONS','status':'passed','conditions_recalculated':len(predictions),
        'models_validated':4,'qa_datasets_included':False,'prediction_id_alignment':'derived frozen index; original QA bytes required for independent external alignment proof',
        'full_inference_replay':'separate; requires authorized data and TimeNet environment','no_test_access':True}


if __name__=='__main__':print(json.dumps(verify(),indent=2))
