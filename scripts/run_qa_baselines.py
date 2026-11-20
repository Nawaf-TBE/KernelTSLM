"""Frozen QA baseline fitting/evaluation and independent replay; no test decoding."""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sys
import warnings
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from kernel_xray.phase3.timef_dataset import TimeFTraceDataset,DEFAULT_STATISTICS
from kernel_xray.phase3.cnn_training import isolated_reads
from kernel_xray.qa.verification import verify_example
from kernel_xray.qa_baselines import core,audit
OUT=ROOT/'outputs/phase4/qa_baselines_v1';QA=ROOT/'outputs/phase4/qa_v1';CONFIG=ROOT/'configs/phase4_qa_baselines_v1.json'
KINDS=('comparison','temporal');MODELS=('majority','lookup','primary','aware','direct')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def load(p):return json.loads(Path(p).read_text())
def save(name,data): (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def rows(split):
    if split not in ('train','val'):raise ValueError('No test QA')
    return [json.loads(line) for line in (QA/f'{split}.jsonl').read_text().splitlines()]


def check_frozen():
    manifest=load(QA/'COMPLETE.json');assert manifest['status']=='passed' and manifest['questions']==19021
    for name,h in manifest['artifacts_sha256'].items():assert digest(QA/name)==h,name
    expected={'train':'d63a0544e7ae3e3029d1af42c4a9a9f0d10308a8f4fcab7f29eecd816f298d41','val':'e873d36c89f86619bb29c98b19971e9b574490f48f9f37e94ad8b66c6adcf0df'}
    for split,h in expected.items():assert digest(QA/f'{split}.jsonl')==h
    assert digest(DEFAULT_STATISTICS)==load(QA/'protocol.json')['normalization']['sha256']
    return expected


def preserve(start=False):
    p=OUT/'preservation_before.json'
    if start:
        assert not p.exists(),'Refuse to replace preservation inventory'
        inventory={}
        for base in ['data','outputs','src','scripts','tests','configs','docs','results']:
            for f in (ROOT/base).rglob('*'):
                if f.is_file() and '__pycache__' not in f.parts and not f.is_relative_to(OUT) and not str(f.relative_to(ROOT)).startswith('src/kernel_xray/qa_baselines/') and f.name not in ('run_qa_baselines.py','phase4_qa_baselines_v1.json','test_qa_baselines.py','plot_qa_baselines.py','phase4_qa_baselines_run.log'):
                    inventory[str(f.relative_to(ROOT))]=digest(f)
        inventory['README.md']=digest(ROOT/'README.md');inventory['AGENTS.md']=digest(ROOT/'AGENTS.md')
        save('preservation_before.json',inventory)
    protected=load(p);changed=[p for p,h in protected.items() if digest(ROOT/p)!=h]
    assert not changed,changed
    return {'status':'passed','files_unchanged':len(protected)}


def collect(split,independent=False):
    records=rows(split);by_trace=defaultdict(list)
    assert len(records)==(16652 if split=='train' else 2369)
    for r in records:by_trace[r['trace_id']].append(r)
    dataset=TimeFTraceDataset(split);allowed={i.path for i in dataset.identities}
    assert set(by_trace)==allowed and len(allowed)==(1124 if split=='train' else 160)
    matrices={};direct={};reverse_answers={};nverified=0
    protocol=load(QA/'protocol.json');cfg=load(ROOT/'configs/phase4_qa_v1.json')
    feature_fn=audit.independent_features if independent else core.features
    solver=audit.independent_direct if independent else core.direct
    try:
        for start in range(0,len(dataset),16):
            for sample in dataset.__getitems__(list(range(start,min(start+16,len(dataset))))):
                x=sample['X'].numpy();meta=sample['metadata'];trace=meta['trace_id'];names=list(meta['counter_names'])
                for r in by_trace.pop(trace):
                    verify_example(r,x,names,split,trace,cfg,protocol['normalization']);nverified+=1
                    p={k:r[k] for k in ('type','channels','start','end')}
                    direct[r['id']]=solver(x,p,cfg)
                    if r['type']!='statistics':
                        # Audit transformation is constructed without the engine helper.
                        altered=x.copy()
                        for c in p['channels']:altered[c,p['start']:p['end']]=x[c,p['start']:p['end']][::-1]
                        reverse_answers[r['id']]=solver(altered,p,cfg)
                        for aware in (False,True):
                            matrices[(r['id'],aware,'original')]=feature_fn(x,p,aware)
                            matrices[(r['id'],aware,'reversed')]=feature_fn(altered,p,aware)
            if start%256==0:print('audit' if independent else 'extract',split,min(start+16,len(dataset)),'/',len(dataset),flush=True)
        assert not by_trace
    finally:dataset.close()
    assert nverified==len(records)
    return records,matrices,direct,reverse_answers,sorted(allowed)


def matrix(records,features,aware,condition='original'):
    return np.asarray([features[(r['id'],aware,condition)] for r in records])


def make_predictions(records,features,direct,altered,rules,fits,independent=False):
    result={}
    for kind in KINDS:
        selected=[r for r in records if r['type']==kind]
        for model in MODELS:
            for condition in ('original','reversed_original','reversed_altered'):
                reversed_=condition!='original'
                if model in ('primary','aware'):
                    m=matrix(selected,features,model=='aware','reversed' if reversed_ else 'original')
                    fit=fits[(kind,model)]
                    pred=audit.linear_replay(fit,m) if independent else core.predict_classifier(fit,m)
                elif model in ('majority','lookup'):
                    pred=[rules['pairs'][kind].get(','.join(map(str,r['channels'])),rules['majority'][kind]) if model=='lookup' else rules['majority'][kind] for r in selected]
                else:pred=[altered[r['id']] if reversed_ else direct[r['id']] for r in selected]
                result[f'{kind}/{model}/{condition}']=[{'id':r['id'],'trace_id':r['trace_id'],'reference':altered[r['id']] if condition=='reversed_altered' else r['answer'],'prediction':str(p)} for r,p in zip(selected,pred,strict=True)]
        for model in ('primary','aware'):
            m=matrix(selected,features,model=='aware');fit=fits[(kind,model)]
            pred=audit.linear_replay(fit,m,True) if independent else core.predict_classifier(fit,m,True)
            result[f'{kind}/{model}/identity_only']=[{'id':r['id'],'trace_id':r['trace_id'],'reference':r['answer'],'prediction':str(p)} for r,p in zip(selected,pred,strict=True)]
    selected=[r for r in records if r['type']=='statistics']
    for model in ('majority','direct'):
        result[f'statistics/{model}/original']=[{'id':r['id'],'trace_id':r['trace_id'],'reference':r['answer'],'prediction':rules['constant'] if model=='majority' else direct[r['id']]} for r in selected]
    return result


def evaluate(predictions,independent=False):
    cfg=load(ROOT/'configs/phase4_qa_v1.json');result={}
    for key,items in predictions.items():
        truth=[r['reference'] for r in items];pred=[r['prediction'] for r in items]
        if key.startswith('statistics/'):
            result[key]=(audit.numerical if independent else core.numerical_metrics)(truth,pred,cfg)
        else:
            result[key]=(audit.categorical if independent else core.categorical_metrics)(truth,pred,sorted(set(truth)))
    return result


def independent_rules(records):
    majority={};pairs={}
    for kind in KINDS:
        freq={};local={}
        for row in records:
            if row['type']!=kind:continue
            answer=row['answer'];freq[answer]=freq.get(answer,0)+1;key=','.join(map(str,row['channels']))
            local.setdefault(key,{});local[key][answer]=local[key].get(answer,0)+1
        majority[kind]=max(freq,key=freq.get);pairs[kind]={k:max(d,key=d.get) for k,d in local.items()}
    values=sorted(r['answer'] for r in records if r['type']=='statistics');n=len(values)
    return {'majority':majority,'pairs':pairs,'constant':(values[(n-1)//2]+values[n//2])/2}


def main(independent=False):
    OUT.mkdir(parents=True,exist_ok=True);cfg=load(CONFIG);hashes=check_frozen()
    if not independent:
        assert not (OUT/'models.pkl').exists(),'Refuse to overwrite frozen baseline fit'
        preserve(True)
        save('protocol.json',{'configuration':cfg,'configuration_sha256':digest(CONFIG),'qa_sha256':hashes,
            'qa_config_sha256':digest(ROOT/'configs/phase4_qa_v1.json'),'normalization_sha256':digest(DEFAULT_STATISTICS),
            'versions':{p:version(p) for p in ('numpy','scikit-learn','torch','timenet','pyarrow')},
            'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),*sorted((ROOT/'src/kernel_xray/qa_baselines').glob('*.py'))]}})
    else:
        protocol=load(OUT/'protocol.json');assert protocol['configuration_sha256']==digest(CONFIG)
        for p,h in protocol['source_sha256'].items():assert digest(ROOT/p)==h,p
        for p,h in load(OUT/'model_hashes.json').items():assert digest(OUT/p)==h
    train,train_features,_,_,train_ids=collect('train',independent)
    val,val_features,solver,altered,val_ids=collect('val',independent)
    assert not set(train_ids)&set(val_ids)
    rules=independent_rules(train) if independent else core.fit_rules(train)
    if independent:
        assert rules==load(OUT/'rules.json')
        with (OUT/'models.pkl').open('rb') as f:fits=pickle.load(f)
    else:
        save('rules.json',rules);fits={};repeat={}
        for kind in KINDS:
            selected=[r for r in train if r['type']==kind];answers=[r['answer'] for r in selected]
            for model in ('primary','aware'):
                m=matrix(selected,train_features,model=='aware')
                fit=core.fit_classifier(m,answers,cfg['logistic']);again=core.fit_classifier(m,answers,cfg['logistic'])
                assert np.array_equal(fit['model'].coef_,again['model'].coef_) and np.array_equal(fit['model'].intercept_,again['model'].intercept_)
                assert np.array_equal(fit['scaler'].mean_,again['scaler'].mean_) and np.array_equal(fit['scaler'].scale_,again['scaler'].scale_)
                fits[(kind,model)]=fit;repeat[f'{kind}/{model}']={'coefficient_repeat_bitwise':True,'iterations':fit['model'].n_iter_.tolist(),'training_n':len(answers),'features':m.shape[1]}
        with (OUT/'models.pkl').open('wb') as f:pickle.dump(fits,f)
        save('repeatability.json',repeat)
    for (kind,model),fit in fits.items():
        selected=[r for r in train if r['type']==kind];m=matrix(selected,train_features,model=='aware');n=fit['n_continuous']
        np.testing.assert_allclose(m[:,:n].mean(0),fit['scaler'].mean_,rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(m[:,:n].var(0),fit['scaler'].var_,rtol=1e-11,atol=1e-11)
        assert fit['scaler'].n_samples_seen_==len(selected)
        pred=core.predict_classifier(fit,m)
        assert np.array_equal(pred,audit.linear_replay(fit,m))
    predictions=make_predictions(val,val_features,solver,altered,rules,fits,independent)
    metrics=evaluate(predictions,independent)
    if independent:
        stored=load(OUT/'predictions.json');assert set(stored)==set(predictions)
        for key,items in predictions.items():
            assert len(items)==len(stored[key])
            for a,b in zip(items,stored[key],strict=True):audit.compare(a,b)
        audit.compare(evaluate(stored,True),load(OUT/'metrics.json'))
        audit.compare(metrics,load(OUT/'metrics.json'))
        save('metrics_independent.json',metrics)
        save('verification.json',{'verdict':'VERIFIED WITH QUALIFICATIONS','status':'passed',
             'validation_questions':len(val),'training_questions':len(train),'unique_validation_traces':len(val_ids),
             'all_model_replays_match':True,'independent_features':'Python fsum/std/explicit peak search',
             'independent_prediction':'manual StandardScaler and coefficient matrix replay; exact classes',
             'independent_metrics':'Python confusion counts, scalar precision/recall/F1, fsum numerical errors',
             'training_scaler_moments_recomputed':True,'training_rules_recomputed':True,
             'qa_sha256':check_frozen(),'preservation':preserve(),'no_test_arrays_decoded':True,
             'limitations':['shared workload families','single seed','primary features may indirectly reveal answers on special cases',
                            'aware features are answer-sufficient; direct solver is arithmetic, not learned reasoning',
                            'structured parameters supplied; no natural-language parsing evaluated']})
    else:
        save('predictions.json',predictions);save('metrics.json',metrics)
        save('model_hashes.json',{p:digest(OUT/p) for p in ('models.pkl','rules.json')})
        # Per-question feature caches contain no target arrays; identity is bookkeeping only.
        for split,records,feat in [('train',train,train_features),('val',val,val_features)]:
            for kind in KINDS:
                selected=[r for r in records if r['type']==kind]
                np.savez_compressed(OUT/f'{split}_{kind}_features.npz',ids=[r['id'] for r in selected],
                    primary=matrix(selected,feat,False),aware=matrix(selected,feat,True))
        # Independent feature replay also compares these engine caches during audit.
        uncertainty=audit.bootstrap(predictions,val_ids,cfg['bootstrap'],load(ROOT/'configs/phase4_qa_v1.json'))
        save('uncertainty.json',uncertainty)
        ablation={}
        for kind in KINDS:
            report={}
            for model in MODELS:
                original=predictions[f'{kind}/{model}/original'];rev=predictions[f'{kind}/{model}/reversed_original'];new=predictions[f'{kind}/{model}/reversed_altered']
                report[model]={'prediction_change_fraction':float(np.mean([a['prediction']!=b['prediction'] for a,b in zip(original,rev)])),
                    'original_accuracy':metrics[f'{kind}/{model}/original']['accuracy'],
                    'reversed_original_label_accuracy':metrics[f'{kind}/{model}/reversed_original']['accuracy'],
                    'reversed_altered_label_accuracy':metrics[f'{kind}/{model}/reversed_altered']['accuracy']}
            ablation[kind]={'correct_label_change_fraction':float(np.mean([a['reference']!=b['reference'] for a,b in zip(original,new)])),
                'all_altered_answers_well_defined':True,'models':report}
        save('ablation.json',ablation)
        shortcut={kind:{model:{'accuracy':metrics[f'{kind}/{model}/original']['accuracy'],
             'identities_only_accuracy':metrics[f'{kind}/{model}/identity_only']['accuracy'],
             'prediction_change_fraction':float(np.mean([a['prediction']!=b['prediction'] for a,b in zip(predictions[f'{kind}/{model}/original'],predictions[f'{kind}/{model}/identity_only'])]))}
             for model in ('primary','aware')} for kind in KINDS}
        quality=load(QA/'quality.json')['shortcuts']
        for kind in KINDS:
            actual=metrics[f'{kind}/lookup/original']['accuracy'];expected=quality[kind]['ordered_counter_pair_training_majority_accuracy']
            assert actual==expected,(kind,actual,expected)
            shortcut[kind]['lookup_reproduces_qa_v1']=True
        save('shortcuts.json',shortcut);save('preservation_after_fit.json',preserve())
    if independent:
        for split,records,feat in [('train',train,train_features),('val',val,val_features)]:
            for kind in KINDS:
                selected=[r for r in records if r['type']==kind]
                with np.load(OUT/f'{split}_{kind}_features.npz',allow_pickle=False) as saved:
                    assert saved['ids'].tolist()==[r['id'] for r in selected]
                    for model in ('primary','aware'):np.testing.assert_allclose(saved[model],matrix(selected,feat,model=='aware'),rtol=1e-10,atol=1e-10)
    print('Independent audit passed' if independent else 'Baseline fit/evaluation complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    try:
        with isolated_reads(),threadpool_limits(limits=1),warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);main(args.audit)
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);save('verification.json',{'status':'failed','verdict':'FAILED','error':str(error)})
        raise
