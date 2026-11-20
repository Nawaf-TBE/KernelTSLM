"""One-way export/replay of trusted-local frozen models; no fitting or QA generation."""
from pathlib import Path
import sys,json,hashlib,pickle,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from threadpoolctl import threadpool_limits
from kernel_xray.qa_baselines.portable import PortableModel,feature_names
from kernel_xray.phase3.cnn_training import isolated_reads
from run_qa_baselines import collect,matrix
B=ROOT/'outputs/phase4/qa_baselines_v1';Q=ROOT/'outputs/phase4/qa_v1';OUT=ROOT/'outputs/phase4/qa_reproducibility';M=ROOT/'models/qa_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    OUT.mkdir(exist_ok=True);M.mkdir(parents=True,exist_ok=True)
    hashes=read(B/'model_hashes.json')
    for p,h in hashes.items():assert sha(B/p)==h
    assert sha(B/'protocol.json')==read(B/'COMPLETE.json')['artifact_sha256']['protocol.json']
    protocol=read(B/'protocol.json');assert sha(ROOT/'configs/phase4_qa_baselines_v1.json')==protocol['configuration_sha256']
    # Fixed before any equivalence comparisons.
    tolerance={'rtol':1e-12,'atol':1e-12}
    write(OUT/'protocol.json',{'probability_tolerance':tolerance,'model_source_sha256':hashes['models.pkl'],'no_refitting':True})
    with (B/'models.pkl').open('rb') as f:fits=pickle.load(f) # explicitly trusted LOCAL export only
    records,features,_,_,traces=collect('val',True)
    reports={}
    for (kind,variant),fit in fits.items():
        model=fit['model'];scaler=fit['scaler']
        metadata={'schema_version':1,'task':kind,'variant':variant,'classes':model.classes_.tolist(),
            'coefficients':model.coef_.tolist(),'intercepts':model.intercept_.tolist(),'feature_names':feature_names(variant),
            'scaler':{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'variance':scaler.var_.tolist(),'training_samples':int(scaler.n_samples_seen_)},
            'categorical_state':'A then B fixed 24-way one-hot; untouched by continuous scaler',
            'configuration':model.get_params(),'source':{'path':'outputs/phase4/qa_baselines_v1/models.pkl','sha256':hashes['models.pkl'],
                'protocol_sha256':sha(B/'protocol.json'),'configuration_sha256':protocol['configuration_sha256'],'executed_source_sha256':protocol['source_sha256']},
            'normalization_sha256':protocol['normalization_sha256']}
        path=M/f'{kind}_{variant}.json';write(path,metadata);portable=PortableModel.load(path)
        selected=[r for r in records if r['type']==kind];x=matrix(selected,features,variant=='aware')
        with np.load(B/f'val_{kind}_features.npz',allow_pickle=False) as original:
            cached=original[variant];assert original['ids'].tolist()==[r['id'] for r in selected]
        n=fit['n_continuous'];design=np.column_stack((scaler.transform(cached[:,:n]),cached[:,n:]))
        expected=model.predict_proba(design);actual=portable.predict_proba(x)
        np.testing.assert_allclose(actual,expected,**tolerance)
        assert np.array_equal(portable.predict(x),model.predict(design))
        reports[f'{kind}/{variant}']={'questions':len(selected),'class_disagreements':0,'max_probability_abs_difference':float(abs(actual-expected).max()),'sha256':sha(path)}
    source_predictions=read(B/'predictions.json');export={};ids_by_type={}
    frozen=read(Q/'verification.json')['dataset_sha256']
    for split in ('train','val'):assert sha(Q/f'{split}.jsonl')==frozen[split]
    questions=[json.loads(line) for line in (Q/'val.jsonl').read_text().splitlines()];byid={r['id']:r for r in questions};assert len(byid)==2369
    for condition,rows in source_predictions.items():
        kind=condition.split('/')[0];export[condition]=[]
        for row in rows:
            q=byid[row['id']];assert q['type']==kind and q['trace_id']==row['trace_id']
            if not condition.endswith('/reversed_altered'):assert q['answer']==row['reference']
            export[condition].append({k:row[k] for k in ('id','reference','prediction')})
    write(ROOT/'results/qa_v1/predictions.json',export)
    write(ROOT/'results/qa_v1/question_index.json',{'split':'val','dataset_sha256':frozen['val'],
        'by_type':{kind:[r['id'] for r in questions if r['type']==kind] for kind in ('statistics','comparison','temporal')}})
    write(ROOT/'results/qa_v1/condition_metrics.json',read(B/'metrics.json'))
    write(OUT/'model_equivalence.json',{'status':'passed','traces':len(traces),'tolerance':tolerance,'models':reports,
        'predictions_source_sha256':sha(B/'predictions.json'),'transformation':'retain condition key, question ID, reference, prediction; remove trace identifiers; no raw inputs, question strings, or oracle operands'})
    print(json.dumps(reports,indent=2))

if __name__=='__main__':
    with isolated_reads(),threadpool_limits(limits=1):main()
