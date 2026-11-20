"""Generate or independently verify isolated KernelSight QA v1; never train/publish."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
from contextlib import ExitStack
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from kernel_xray.phase3.timef_dataset import TimeFTraceDataset,DEFAULT_STATISTICS,DEFAULT_VERSIONS
from kernel_xray.phase3.cnn_training import isolated_reads
from kernel_xray.qa.generation import generate_trace,model_input
from kernel_xray.qa.verification import verify_example
OUT=ROOT/'outputs/phase4/qa_v1'
CONFIG=ROOT/'configs/phase4_qa_v1.json'


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def line(row):return json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'


def authorized_dataset(split):
    if split not in ('train','val'):raise ValueError('Test access forbidden')
    return TimeFTraceDataset(split)


def preservation():
    inventory=json.loads((OUT/'preservation_before.json').read_text())
    # New Phase 4 implementation was captured while under development; protect all prior work.
    protected={p:h for p,h in inventory.items() if not p.startswith('src/kernel_xray/qa/') and p not in ('scripts/build_qa_v1.py','configs/phase4_qa_v1.json')}
    changed=[p for p,h in protected.items() if digest(ROOT/p)!=h]
    if changed:raise AssertionError(f'Protected files changed: {changed[:5]}')
    return {'status':'passed','files_unchanged':len(protected)}


def run(verify_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    save('verification.json',{'status':'running','mode':'verify' if verify_only else 'generate'})
    cfg=json.loads(CONFIG.read_text())
    if not (OUT/'preservation_before.json').exists():
        inventory={}
        for base in ['data','outputs/phase1','outputs/phase2','outputs/phase3','results','src','tests','scripts','configs','docs']:
            for p in (ROOT/base).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:inventory[str(p.relative_to(ROOT))]=digest(p)
        save('preservation_before.json',inventory)
    protocol={'config':cfg,'configuration_sha256':digest(CONFIG),'interval_sampling':'equal probability bands, uniform integer length in band, uniform valid start',
              'category_selection':'least represented available category per trace from fixed random pool; random tie choice',
              'comparison_ties':'oracle returns Tie for exact ties; generation rejects ties and numerical near-ties',
              'model_representation':'full original-order normalized float32 [24,512], no timestamps or identity metadata',
              'statistics':'arithmetic bin mean, not duration-weighted; float64 accumulation of float32 inputs',
              'normalization':{'path':str(DEFAULT_STATISTICS.relative_to(ROOT)),'sha256':digest(DEFAULT_STATISTICS)},
              'view_provenance':{s:digest(ROOT/f'outputs/phase3/task1/timef_views/{s}/view.json') for s in ('train','val')}}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol,'Protocol changed'
    else:save('protocol.json',protocol)
    stats={};all_ids=set();split_traces={};shortcut_training=defaultdict(Counter);heldout=[];illustrations=[];illustration_traces=set();verified=0
    for split in ('train','val'):
        print('Reading isolated',split,flush=True)
        counts=Counter();answers=defaultdict(Counter);lengths=[];frequencies=Counter();rejections=Counter();shortfalls={};numeric=[];per_trace={};records=[]
        regenerated=hashlib.sha256()
        path=OUT/f'{split}.jsonl'
        if verify_only:
            for text in path.read_text().splitlines():records.append(json.loads(text))
            by_trace=defaultdict(list)
            for row in records:by_trace[row['trace_id']].append(row)
        sink=None if verify_only else path.open('w')
        dataset=authorized_dataset(split);split_traces[split]=set()
        try:
            for start in range(0,len(dataset),16):
                for sample in dataset.__getitems__(list(range(start,min(start+16,len(dataset))))):
                    x=sample['X'].numpy();names=list(sample['metadata']['counter_names']);trace=sample['metadata']['trace_id']
                    split_traces[split].add(trace)
                    generated,rejected,short=generate_trace(x,names,split,trace,cfg,protocol['normalization'])
                    # Independent second generation in generate mode; on verify compare with persisted bytes.
                    again=by_trace.pop(trace,[]) if verify_only else generate_trace(x,names,split,trace,cfg,protocol['normalization'])[0]
                    assert list(map(line,generated))==list(map(line,again)),'Regeneration mismatch'
                    rejections.update(rejected)
                    if short:shortfalls[trace]=short
                    per_trace[trace]=len(generated)
                    seen=set()
                    for row in generated:
                        assert row['id'] not in all_ids;all_ids.add(row['id'])
                        assert row['question'] not in seen;seen.add(row['question'])
                        verify_example(row,x,names,split,trace,cfg,protocol['normalization']);verified+=1
                        model=model_input(row,x,names)
                        assert set(model)=={'question','time_series','counter_names'}
                        assert np.array_equal(np.asarray(model['time_series'],np.float32),x)
                        encoded=line(row);regenerated.update(encoded.encode())
                        if sink:sink.write(encoded)
                        kind=row['type'];counts[kind]+=1;lengths.append(row['end']-row['start']);frequencies.update(row['channels'])
                        if kind=='statistics':numeric.append(row['answer'])
                        else:
                            answers[kind][row['answer']]+=1
                            key=(kind,*row['channels'])
                            if split=='train':shortcut_training[key][row['answer']]+=1
                            else:heldout.append((key,row['answer']))
                        if split=='train' and trace not in illustration_traces and not any(k==kind for k,_,_ in illustrations):
                            illustrations.append((kind,row,x.copy()));illustration_traces.add(trace)
                print(split,min(start+16,len(dataset)),'/',len(dataset),flush=True) if start%256==0 else None
            if verify_only:assert not by_trace,'Unauthorized/extra trace questions'
        finally:
            dataset.close()
            if sink:sink.close()
        assert regenerated.hexdigest()==digest(path),'Persisted ordering/hash mismatch'
        stats[split]={'counts':dict(counts),'total':sum(counts.values()),'traces':len(per_trace),
            'questions_per_trace':dict(Counter(per_trace.values())), 'traces_meeting_all_targets':len(per_trace)-len(shortfalls),
            'target_completion_percent':100*(len(per_trace)-len(shortfalls))/len(per_trace),'shortfalls':shortfalls,
            'answer_categories':{k:dict(v) for k,v in answers.items()},'interval_lengths':dict(Counter(lengths)),
            'counter_frequencies':dict(frequencies),'candidate_rejections':dict(rejections),
            'numerical_answer_range':[min(numeric),max(numeric)] if numeric else None,'sha256':digest(path)}
    assert not split_traces['train']&split_traces['val']
    assert len(split_traces['train'])==1124 and len(split_traces['val'])==160
    shortcuts={}
    for kind in ('comparison','temporal'):
        subset=[(k,a) for k,a in heldout if k[0]==kind];train_counts=Counter()
        for k,counts in shortcut_training.items():
            if k[0]==kind:train_counts.update(counts)
        fallback=train_counts.most_common(1)[0][0]
        predictions=[shortcut_training[k].most_common(1)[0][0] if k in shortcut_training else fallback for k,a in subset]
        shortcuts[kind]={'validation_n':len(subset),'training_majority_answer':fallback,
            'template_only_majority_accuracy':sum(a==fallback for k,a in subset)/len(subset),
            'ordered_counter_pair_training_majority_accuracy':sum(p==a for p,(k,a) in zip(predictions,subset))/len(subset),
            'interpretation':'metadata-only shortcut diagnostic, not time-series model accuracy; train frequencies only, no fitting on validation'}
    save('quality.json',{'splits':stats,'shortcuts':shortcuts,'limitations':['traces are path-disjoint, not proven independent workload identities','comparison in channel-wise normalized units does not compare physical hardware units','unique peaks exclude flat/inactive or plateau counters; target shortfalls are retained','within-trace balancing is not natural prevalence; counter identity and same-bin synchrony may offer shortcuts']})
    save('illustrations.json',[{'type':k,'example':row,'X':x.tolist()} for k,row,x in illustrations])
    save('verification.json',{'status':'passed','questions_verified':verified,'agreement_percent':100,
         'independent_method':'math.fsum, Python max/sorted, explicit unique-index enumeration; does not call oracle',
         'deterministic_regeneration':True,'no_test_access':True,'strict_views_only':True,
         'label_based_generation':False,'model_serializer_allowlist_verified':True,'preservation':preservation(),
         'dataset_sha256':{s:stats[s]['sha256'] for s in stats}})
    print('Verified',verified,'questions; preservation passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    # Guard even accidental file/reader access; no label arrays are used by generation.
    try:
        with isolated_reads():run(args.verify)
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        save('verification.json',{'status':'failed','error':str(error),'type':type(error).__name__})
        raise
