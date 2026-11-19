"""Independent CNN audit: immutable artifacts, CPU replay, metrics, isolation and uncertainty."""
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from timenet.dataset import Record
from timenet.reader import TimeFReader, LoadPlan
from timenet.format.duckdb import connect_control
from timenet.storage.parquet import SignalRowGroupCache
from kernel_xray.phase3.independent_audit import (
    count_confusion, metrics_from_counts, independent_temporal_masks,
    functional_frozen_forward, bootstrap_whole_traces,
)

OUT=ROOT/'outputs/phase3/cnn_audit'
CNN=ROOT/'outputs/phase3/cnn'
VIEWS={s:ROOT/f'outputs/phase3/task1/timef_views/{s}/registry/williamhtan/kernelsight/1.0.0' for s in ['train','val']}


def read(path):return json.loads(Path(path).read_text())


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise AssertionError(message)


def array(payload):
    return np.array(payload['values'],dtype=np.dtype(payload['dtype'])).reshape(payload['shape'])


@contextmanager
def guarded_reads(official):
    allowed_views={p.resolve() for p in VIEWS.values()}
    allowed_raw={(ROOT/row['path']).resolve().parent.parent for records in official.values() for row in records}
    init,values,load=TimeFReader.__init__,SignalRowGroupCache.read_values,np.load
    def init_guard(reader,release,*args,**kwargs):
        require(Path(release.root).resolve() in allowed_views,'Audit attempted unapproved TimeF release')
        return init(reader,release,*args,**kwargs)
    def values_guard(cache,release,*args,**kwargs):
        require(Path(release.root).resolve() in allowed_views,'Audit attempted mixed/test counter row group')
        return values(cache,release,*args,**kwargs)
    def np_guard(file,*args,**kwargs):
        if isinstance(file,(Path,str)):
            p=Path(file).resolve()
            if p.is_relative_to((ROOT/'kernels').resolve()):require(p.parent.parent in allowed_raw,'Audit attempted test NPZ')
        return load(file,*args,**kwargs)
    with patch.object(TimeFReader,'__init__',init_guard),patch.object(SignalRowGroupCache,'read_values',values_guard),patch.object(np,'load',np_guard):yield


def integrity():
    hashes={};checks=[]
    def verify(path,expected,label):
        actual=digest(path);hashes[str(path.relative_to(ROOT))]=actual
        require(actual==expected,f'Integrity mismatch: {label}')
        checks.append(label)
    f=CNN/'frozen';manifest=read(f/'manifest.json');config=read(f/'config.json')
    verify(f/'baseline.pt',manifest['checkpoint_sha256'],'frozen checkpoint')
    verify(f/'config.json',manifest['configuration_sha256'],'frozen config')
    require(digest(f/'baseline.pt')==digest(CNN/'unweighted/best.pt'),'Frozen is not unweighted best')
    for name,expected in config['source_files'].items():
        verify(ROOT/name,expected,'live source '+name)
        verify(f/'source'/name,expected,'frozen source '+name)
    for name,expected in read(f/'bundle.json')['files'].items():verify(f/name,expected,'bundle '+name)
    reproduction=read(f/'reproduction_manifest.json')
    for key in ['unchanged_pipeline_source','post_training_scripts']:
        for name,expected in reproduction[key].items():verify(ROOT/name,expected,key+' '+name)
    verify(ROOT/'outputs/phase1/task7/train_statistics.json',manifest['normalization_sha256'],'training statistics')
    require(manifest['selected_experiment']=='unweighted' and manifest['best_epoch']==7,'Frozen selection differs')
    checkpoint=torch.load(f/'baseline.pt',map_location='cpu',weights_only=False)
    require(checkpoint['epoch']==7 and checkpoint['class_weights'] is None,'Frozen checkpoint identity')
    expected={ 'projection.weight':(64,24,1),'projection.bias':(64,),
               'head.weight':(12,64,1),'head.bias':(12,),
               'final_norm.norm.weight':(64,),'final_norm.norm.bias':(64,)}
    for block in range(3):
        for norm in [0,4]:
            for suffix in ['weight','bias']:expected[f'blocks.{block}.temporal.{norm}.norm.{suffix}']=(64,)
        for conv in [2,6]:
            expected[f'blocks.{block}.temporal.{conv}.weight']=(64,64,5)
            expected[f'blocks.{block}.temporal.{conv}.bias']=(64,)
    weights=checkpoint['model_state'];require(set(weights)==set(expected),'Unexpected model state keys')
    require(all(tuple(weights[k].shape)==v and weights[k].dtype==torch.float32 and torch.isfinite(weights[k]).all() for k,v in expected.items()),'Parameter shape/type/finiteness')
    require(sum(w.numel() for w in weights.values())==126540,'Parameter count')
    from kernel_xray.phase3.cnn import CNNConfig,L1TemporalCNN
    model=L1TemporalCNN(CNNConfig(**{**checkpoint['model_config'],'dilations':tuple(checkpoint['model_config']['dilations'])}))
    model.load_state_dict(weights);model.eval()
    convs=[m for m in model.modules() if isinstance(m,torch.nn.Conv1d)]
    require(all(m.stride==(1,) for m in convs),'Unexpected stride')
    rf=1+sum((m.kernel_size[0]-1)*m.dilation[0] for m in convs)
    require(rf==57 and [m.dilation[0] for m in convs if m.kernel_size[0]>1]==[1,1,2,2,4,4],'Actual receptive field')
    require(not list(model.buffers()),'Unexpected stateful normalization buffers')
    torch.manual_seed(19);probe=torch.randn(1,24,512,requires_grad=True)
    functional_frozen_forward(probe,weights)[0,0,256].backward()
    positions=torch.nonzero(probe.grad.abs().sum(1)[0]>1e-12).flatten()
    require((positions[0].item(),positions[-1].item())==(228,284),'Functional receptive field support')
    histories={};reported=read(CNN/'summary.json')
    for kind in ['unweighted','weighted']:
        history=read(CNN/kind/'history.json');best=max(history,key=lambda row:row['val_macro_f1']);bad=0;peak=-float('inf');best_epoch=None
        for row in history:
            if row['val_macro_f1']>peak:peak=row['val_macro_f1'];best_epoch=row['epoch'];bad=0
            else:bad+=1
            require(row['training_valid_bins']==233821,'Incomplete training supervision count')
            require(row['epoch']==len(history) or bad<5,'History continued past early stop')
        ckpt=torch.load(CNN/kind/'best.pt',map_location='cpu',weights_only=False)
        require(set(ckpt['model_state'])==set(expected) and all(tuple(ckpt['model_state'][k].shape)==shape for k,shape in expected.items()), 'Experiment parameter shapes')
        require(ckpt['training_config']==config['config'],'Experiment configuration differs')
        require(best['epoch']==best_epoch==ckpt['epoch']==reported['experiments'][kind]['best_epoch'],'History/checkpoint selection mismatch')
        require(peak==ckpt['selection_macro_f1']==reported['experiments'][kind]['validation']['macro_f1_supported'],'Selected score mismatch')
        require(bad==5 or len(history)==30,'Stopping rule mismatch')
        histories[kind]={'epochs':len(history),'best_epoch':best_epoch,'best_macro_f1':peak,'no_improvement_at_stop':bad}
    ah,bh=read(CNN/'unweighted/history.json'),read(CNN/'weighted/history.json')
    require(all(a['trace_order_sha256']==b['trace_order_sha256'] for a,b in zip(ah,bh)),'Controlled trace ordering differs')
    require(reported['experiments']['unweighted']['initial_parameters_sha256']==reported['experiments']['weighted']['initial_parameters_sha256'],'Different initial states')
    result={'status':'passed','verified_hashes':hashes,'checks':checks,'parameter_count':126540,'actual_receptive_field_bins':rf,
            'gradient_support_bins':[228,284],'tensor_shapes':{k:list(v) for k,v in expected.items()},'histories':histories,
            'history_limit':'Nonbest epoch states were not retained; their numeric scores are checked for internal/log consistency, not reexecuted.',
            'frozen_checkpoint':manifest,'source_to_frozen_exact':True}
    save('source_integrity.json',result)
    return weights,config,result


def audit_view(split,official):
    view=VIEWS[split];native={'train':'train','val':'validation'}[split]
    declared=read(view.parents[3]/'view.json')
    for relative,info in declared['compiled_files'].items():require(digest(view/relative)==info['sha256'],'View file checksum')
    expected_npz=set()
    for row in official:
        base=(ROOT/row['path']).resolve().parent.parent
        expected_npz.update(str(base/group/name) for group,name in [('input','tensor_input.npz'),('labels','labels.npz'),('fingerprint','fingerprint.npz')])
    require(set(declared['source_npz_hashes'])==expected_npz,'View declares sources outside authorized split')
    require(all(digest(Path(p))==h for p,h in declared['source_npz_hashes'].items()),'View source NPZ hash mismatch')
    manifest=read(view/'manifest.json');require(not manifest['payload_sources'],'View reuses external source payloads')
    with connect_control(view/'control.duckdb',read_only=True) as c:
        require({r[0] for r in c.execute('SELECT DISTINCT split FROM tasks').fetchall()}=={native},'Mixed native task partitions')
        ids=[r[0] for r in c.execute('SELECT record_id FROM records ORDER BY record_id').fetchall()]
        require(len(ids)==len(official),'Record count mismatch')
        sizes=c.execute('SELECT record_id,count(*) FROM sources JOIN signals USING(source_id) GROUP BY record_id').fetchall()
        require(len(sizes)==len(ids) and all(n==24 for _,n in sizes),'Not exactly 24 Signals per Record')
        require(c.execute('SELECT count(*) FROM signal_chunks WHERE source_key IS NOT NULL').fetchone()[0]==0,'External counter reuse')
        require(c.execute('SELECT count(*) FROM external_values WHERE source_key IS NOT NULL').fetchone()[0]==0,'External packed payload reuse')
        chunks=c.execute('SELECT value_path,chunk_major_index,list(chunk_minor_index) FROM signal_chunks GROUP BY 1,2').fetchall()
        by_file={}
        for path,group,rows in chunks:by_file.setdefault(path,{})[group]=set(rows)
        require(set(by_file)=={str(p.relative_to(view)) for p in (view/'time_series').glob('*.parquet')},'Extra/unreferenced counter files')
        for path,groups in by_file.items():
            require((view/path).resolve().is_relative_to(view.resolve()),'Counter path escapes isolated view')
            with pq.ParquetFile(view/path) as f:
                require(set(groups)==set(range(f.num_row_groups)),'Unreferenced physical counter groups')
                require(all(rows==set(range(f.metadata.row_group(g).num_rows)) for g,rows in groups.items()),'Unreferenced physical counter rows')
            original=ROOT/'outputs/phase2/task3/writer/unvalidated-registry/williamhtan/kernelsight/1.0.0'/path
            require(not (view/path).samefile(original),'Isolated view aliases original mixed counter file')
        packed=c.execute('SELECT value_path,list(DISTINCT row_index) FROM external_values GROUP BY 1').fetchall()
        for path,rows in packed:
            with pq.ParquetFile(view/path) as f:require(set(rows)==set(range(f.metadata.num_rows)),'Unreferenced packed payload rows')
    return ids,{'status':'passed','records':len(ids),'counter_files':len(by_file),'counter_row_groups':len(chunks),
                'packed_payload_files':len(packed),'authorized_sources':len(expected_npz),'external_payload_reuse':False,
                'all_physical_counter_and_packed_rows_referenced':True,'original_counter_file_alias':False}


def collect(split,official,stats):
    ids,layout=audit_view(split,official)
    n=len(ids);data={'raw':np.empty((n,24,512),np.float32),'X':np.empty((n,24,512),np.float32),
                    'y':np.empty((n,512),np.int64),'valid':np.empty((n,512),bool),'metadata':[None]*n,
                    'paths':[r['path'] for r in official]}
    seen=set();rounding=0;min_bin=None
    mean,std=np.asarray(stats['mean'])[:,None],np.asarray(stats['std'])[:,None]
    scale=np.where(std>1e-12,std,1)
    with TimeFReader.open(VIEWS[split]) as reader:
        reader.verify()
        for start in range(0,n,16):
            for record in reader.get_many(Record,ids[start:start+16],load=LoadPlan.ALL_EAGER):
                p=record.metadata['kernelsight_v4'];i=int(p['split_index'])
                require(0<=i<n and i not in seen and p['path']==official[i]['path'],'Duplicate or wrong official trace')
                require(p['split']=={'train':'train','val':'validation'}[split] and p['split_record']==official[i],'Partition metadata mismatch')
                seen.add(i);source=p['arrays'];names=array(source['input']['counter_names']).tolist()
                require(names==stats['channel_names'],'Channel ordering differs')
                signals=sorted(record.signals,key=lambda sig:sig.metadata['channel_index'])
                require([s.metadata['channel_index'] for s in signals]==list(range(24)) and [s.name for s in signals]==names,'Signal ordering')
                require(all(s.spec.spec_type=='kernelsight_counter' for s in signals),'Label-bearing Signal')
                raw=np.stack([s.to_numpy() for s in signals]);require(raw.shape==(24,512) and raw.dtype==np.float32,'Counter dtype/shape')
                edges=array(source['input']['time_edges_ns']);target_edges=array(source['labels']['time_edges_ns'])
                require(edges.dtype==np.int64 and edges.shape==(513,) and np.array_equal(edges,target_edges),'Exact ns grid')
                relative=[int(t)-int(edges[0]) for t in edges];q=np.array([t//1000 for t in relative],np.int64)
                require(all(np.array_equal(s.time_offsets_us(),q[:-1]) for s in signals),'Native offsets not aligned')
                require(record.time_span.start_us==0 and record.time_span.end_us==q[-1],'Final bin bound')
                rounding=max(rounding,max(t%1000 for t in relative));min_bin=min(min_bin or int(np.diff(q).min()),int(np.diff(q).min()))
                y=array(source['labels']['workload_l1']);valid=array(source['labels']['mask_labeled']).astype(bool)
                require(y.shape==(512,) and np.array_equal(valid,y>=0),'Primary label/mask mismatch')
                vocab=array(source['labels']['vocab_l1']).tolist();require(len(vocab)==12,'Vocabulary size')
                path=ROOT/p['path']
                require(digest(path)==p['source_sha256']['input'],'Input source identity hash')
                with np.load(path,allow_pickle=True) as src:
                    require(raw.tobytes()==src['data'].tobytes(),'TimeF/raw counter bits differ')
                    require(np.array_equal(edges,src['time_edges_ns']) and names==src['counter_names'].tolist(),'Source grid/channels')
                with np.load(path.parent.parent/'labels/labels.npz',allow_pickle=True) as src:
                    require(np.array_equal(y,src['workload_l1']) and np.array_equal(valid,src['mask_labeled'].astype(bool)),'Source labels/masks')
                    require(vocab==src['vocab_l1'].tolist() and np.array_equal(edges,src['time_edges_ns']),'Source label vocabulary/grid')
                x=((raw.astype(np.float64)-mean)/scale).astype(np.float32)
                motif=official[i].get('motif',Path(p['path']).parts[1])
                components=Path(p['path']).parts;config='/'.join(components[components.index('_out')+1:-2])
                symbol_names=sorted(set(array(source['input']['kernel_names']).reshape(-1).tolist()))
                symbol_key=hashlib.sha256(json.dumps(symbol_names,separators=(',',':')).encode()).hexdigest()
                problem=re.search(r'L\d+_P\d+',config)
                problem_key=motif+'/'+problem.group() if motif=='kernelbench' and problem else motif
                shape=re.match(r'(m\d+_n\d+_k\d+)',config)
                data['raw'][i]=raw;data['X'][i]=x;data['y'][i]=y;data['valid'][i]=valid
                data['metadata'][i]={'path':p['path'],'collector':motif,'configuration':motif+'/'+config,
                    'shape_configuration':motif+'/'+(shape.group() if shape else config),'problem_aware':problem_key,
                    'compiled_symbol_set':symbol_key,'unique_kernel_symbols':len(symbol_names),'kernel_symbols':symbol_names,
                    'input_sha256':p['source_sha256']['input'],
                    'counter_sha256':hashlib.sha256(raw.tobytes()).hexdigest()}
            if start%160==0:print('Independent input audit',split,start+len(ids[start:start+16]),'/',n,flush=True)
    require(len(seen)==n,'Missing view traces')
    data['names']=vocab
    h=hashlib.sha256()
    for key in ['X','y','valid']:h.update(data[key].tobytes())
    h.update(json.dumps(data['paths']).encode());data['cache_fingerprint']=h.hexdigest()
    layout.update(max_native_floor_error_ns=rounding,minimum_native_bin_width_us=min_bin,raw_source_equality='bitwise',channels=names)
    return data,layout


def prediction_audit(kind,validation,supported,names):
    with np.load(CNN/kind/'validation_predictions.npz',allow_pickle=False) as f:rows={k:f[k].copy() for k in f.files}
    ti,bi,truth,pred=[rows[k] for k in ['trace_index','bin_index','true_class','predicted_class']]
    require(rows['trace_ids'].tolist()==validation['paths'],'Prediction trace order')
    require(all(np.issubdtype(a.dtype,np.integer) for a in [ti,bi,truth,pred]),'Noninteger identity/class')
    require(np.all((ti>=0)&(ti<160)&(bi>=0)&(bi<512)),'Out-of-range prediction location')
    keys=ti.astype(np.int64)*512+bi
    expected=np.flatnonzero(validation['valid'].reshape(-1))
    require(len(keys)==len(np.unique(keys))==33416 and np.array_equal(np.sort(keys),expected),'Missing/duplicate/unknown predictions')
    require(np.array_equal(truth,validation['y'][ti,bi]),'Stored truth differs from source labels')
    prob=rows['probabilities'];require(prob.shape==(33416,12) and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all(),'Probability domain')
    np.testing.assert_allclose(prob.sum(1),1,atol=1e-6);require(np.array_equal(prob.argmax(1),pred),'Argmax mismatch')
    cm=count_confusion(truth,pred);metric=metrics_from_counts(cm,supported,names)
    original=read(CNN/kind/'results.json')['validation'];summary=read(CNN/'summary.json')['experiments'][kind]['validation']
    for reference in [original,summary,read(CNN/kind/'validation_results.json')]:
        require(np.array_equal(cm,reference['confusion_matrix']),'Integer confusion mismatch')
        for field in ['accuracy','micro_f1','macro_f1_supported','macro_f1_all12']:
            require(abs(metric[field]-reference[field])<1e-12,'Metric discrepancy '+field)
        for actual,saved in zip(metric['per_class'],reference['per_class']):
            for field in ['precision','recall','f1','support','predicted']:
                require(abs(actual[field]-saved[field])<1e-12,'Per-class discrepancy')
    per_trace=np.stack([count_confusion(truth[ti==i],pred[ti==i]) for i in range(160)])
    grid=np.full((160,512),-1,np.int64);grid[ti,bi]=pred
    metric.update(status='passed',one_to_one_trace_bin_coverage=True,unknown_evaluated=0,
                  prediction_trace_count=160,unsupported_other_learned=False)
    return rows,metric,per_trace,grid


def additional_control_checks(training, validation, frozen_config):
    """Reconstruct initial seed/order and observe eval state without an optimizer."""
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from kernel_xray.phase3.cnn import L1TemporalCNN, CNNConfig
    from kernel_xray.phase3.cnn_training import isolated_reads
    cfg=frozen_config['config']
    torch.manual_seed(cfg['seed'])
    model=L1TemporalCNN(CNNConfig(**{**cfg['model'],'dilations':tuple(cfg['model']['dilations'])}))
    initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    require(all(initial==read(CNN/kind/'results.json')['initial_parameters_sha256'] for kind in ['unweighted','weighted']),
            'Independent seeded initialization differs')
    loader=DataLoader(training['paths'],batch_size=cfg['batch_size'],shuffle=True,
                      generator=torch.Generator().manual_seed(cfg['seed']),num_workers=0)
    histories={kind:read(CNN/kind/'history.json') for kind in ['unweighted','weighted']}
    orders=[]
    for epoch in range(max(map(len,histories.values()))):
        h=hashlib.sha256()
        for batch in loader:h.update(json.dumps(batch).encode())
        orders.append(h.hexdigest())
        for kind,history in histories.items():
            if epoch<len(history):require(history[epoch]['trace_order_sha256']==orders[-1],'Independent trace order differs')
    ckpt=torch.load(CNN/'frozen/baseline.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model_state']);model.eval()
    before={name:value.detach().clone() for name,value in model.state_dict().items()}
    observed=[]
    def hook(module,args):
        require(len(args)==1 and isinstance(args[0],torch.Tensor) and args[0].shape==(2,24,512),'Unexpected forward modality')
        observed.append('single float32 counter tensor')
    handle=model.register_forward_pre_hook(hook)
    with torch.no_grad():actual=model(torch.from_numpy(validation['X'][:2]))
    handle.remove()
    independent=functional_frozen_forward(torch.from_numpy(validation['X'][:2]),ckpt['model_state'])
    torch.testing.assert_close(actual,independent,atol=1e-6,rtol=1e-6)
    require(all(torch.equal(value,model.state_dict()[name]) for name,value in before.items()),'Original eval mutates parameters/buffers')
    require(all(p.grad is None for p in model.parameters()),'Eval produced parameter gradients')
    with isolated_reads():
        try:
            TimeFReader(SimpleNamespace(root='/unapproved/test/release'))
        except AssertionError:
            blocked=True
        else:
            blocked=False
        require(blocked,'Actual development reader guard accepts an unapproved release')
    return {'status':'passed','seeded_initial_parameters_reconstructed':initial,
            'all_18_epoch_trace_orders_reconstructed':orders,'actual_module_eval_matches_functional_model':True,
            'actual_forward_hook':observed,'parameters_and_buffers_unchanged_after_eval':True,
            'no_validation_gradients':True,'actual_development_guard_rejects_unapproved_reader':True,
            'no_optimizer_applied_to_any_cnn':True}


def workloads(training,validation,predictions,supported,names,protocol):
    trainmeta,valmeta=training['metadata'],validation['metadata']
    exact=[];aliases=[]
    counter_map={};input_map={}
    for i,meta in enumerate(trainmeta):
        counter_map.setdefault(meta['counter_sha256'],[]).append(i)
        input_map.setdefault(meta['input_sha256'],[]).append(i)
    for j,meta in enumerate(valmeta):
        for i in counter_map.get(meta['counter_sha256'],[]):
            require(training['raw'][i].tobytes()==validation['raw'][j].tobytes(),'Counter hash collision')
            exact.append({'train':training['paths'][i],'val':validation['paths'][j],
                          'val_valid_bins':int(validation['valid'][j].sum()),
                          'all_zero':bool(not np.any(training['raw'][i])),
                          'same_input_container':trainmeta[i]['input_sha256']==meta['input_sha256']})
        for i in input_map.get(meta['input_sha256'],[]):aliases.append({'train':training['paths'][i],'val':validation['paths'][j]})
    grouped={}
    for key in ['collector','problem_aware','compiled_symbol_set','configuration','shape_configuration']:
        training_groups={m[key] for m in trainmeta}
        flags=np.array([m[key] in training_groups for m in valmeta])
        val_groups={m[key] for m in valmeta}
        groups={}
        for label,group_mask in [('seen',flags),('unseen',~flags)]:
            mask=validation['valid'] & group_mask[:,None]
            n_traces=int(group_mask.sum());n_bins=int(mask.sum())
            sufficient=n_traces>=protocol['families']['meaningful_group_support']['minimum_traces'] and n_bins>=protocol['families']['meaningful_group_support']['minimum_valid_bins']
            groups[label]={'positive_truth_class_ids':np.flatnonzero(np.bincount(validation['y'][mask],minlength=12)).tolist() if n_bins else [],'traces':n_traces,'valid_bins':n_bins,'meaningful_support':sufficient,
                           'metrics':metrics_from_counts(count_confusion(validation['y'][mask],predictions[mask]),supported,names) if n_bins else None}
        grouped[key]={'training_groups':len(training_groups),'validation_groups':len(val_groups),
                      'shared_groups':len(training_groups & val_groups),'unseen_validation_groups':sorted(val_groups-training_groups),
                      'seen_validation_fraction':float(flags.mean()),'groups':groups,
                      'comparison_supported':all(row['meaningful_support'] for row in groups.values())}
    symbol_union={symbol for meta in trainmeta for symbol in meta['kernel_symbols']}
    unseen_symbols=[]
    for meta in valmeta:
        novel=sorted(set(meta['kernel_symbols'])-symbol_union)
        unseen_symbols.append({'path':meta['path'],'symbol_count':len(meta['kernel_symbols']),
                               'new_symbol_count':len(novel),'new_symbol_sha256':[hashlib.sha256(x.encode()).hexdigest() for x in novel]})
    family_rows=[]
    for family in sorted({m['collector'] for m in trainmeta+valmeta}):
        tr=[i for i,m in enumerate(trainmeta) if m['collector']==family]
        va=[i for i,m in enumerate(valmeta) if m['collector']==family]
        family_rows.append({'family':family,'training_traces':len(tr),'validation_traces':len(va),
                            'validation_valid_bins':int(validation['valid'][va].sum()) if va else 0})
    a=validation['X'].reshape(len(valmeta),-1).astype(np.float64)
    b=training['X'].reshape(len(trainmeta),-1).astype(np.float64)
    an=np.linalg.norm(a,axis=1);bn=np.linalg.norm(b,axis=1);dot=a @ b.T
    denom=an[:,None]*bn[None,:]
    cosine=np.divide(dot,denom,out=np.zeros_like(dot),where=denom>0)
    cosine[(an[:,None]==0)&(bn[None,:]==0)]=1
    cosine=np.clip(cosine,-1,1)
    relative=np.sqrt(np.maximum(an[:,None]**2+bn[None,:]**2-2*dot,0))/np.maximum(np.maximum(an[:,None],bn[None,:]),1e-12)
    near=(cosine>=protocol['similarity']['near_cosine_minimum'])&(relative<=protocol['similarity']['near_relative_l2_maximum'])
    pairs=[]
    for j,i in zip(*np.nonzero(near)):
        vm,tm=valmeta[j],trainmeta[i]
        pairs.append({'val_trace':vm['path'],'train_trace':tm['path'],'cosine':float(cosine[j,i]),
                      'relative_l2':float(relative[j,i]),'exact_counter_bytes':vm['counter_sha256']==tm['counter_sha256'],
                      'same_collector':vm['collector']==tm['collector'],'same_configuration':vm['configuration']==tm['configuration'],
                      'same_shape_configuration':vm['shape_configuration']==tm['shape_configuration'],
                      'same_compiled_symbol_set':vm['compiled_symbol_set']==tm['compiled_symbol_set']})
    with (OUT/'near_duplicate_pairs.csv').open('w',newline='') as f:
        fields=list(pairs[0]) if pairs else ['val_trace','train_trace','cosine','relative_l2']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(pairs)
    nearest=[]
    for j in range(len(valmeta)):
        i=int(np.argmin(relative[j]));nearest.append({'val':valmeta[j]['path'],'train':trainmeta[i]['path'],
            'relative_l2':float(relative[j,i]),'cosine':float(cosine[j,i]),'near_pairs':int(near[j].sum())})
    duplicated_within={}
    for split,meta in [('train',trainmeta),('val',valmeta)]:
        hashes={}
        for m in meta:hashes.setdefault(m['counter_sha256'],[]).append(m['path'])
        duplicated_within[split]=[paths for paths in hashes.values() if len(paths)>1]
    return {'status':'diagnostic_complete','method_predeclared':protocol['similarity'],
            'trace_paths_disjoint':not bool(set(training['paths'])&set(validation['paths'])),
            'exact_cross_partition_counter_pairs':exact,'identical_input_container_aliases':aliases,
            'within_partition_exact_counter_groups':duplicated_within,
            'grouping_definitions':protocol['families'],'groupings':grouped,'collector_counts':family_rows,
            'near_duplicate_pair_count':len(pairs),'validation_traces_with_near_match':int(near.any(1).sum()),
            'validation_valid_bins_with_near_match':int(validation['valid'][near.any(1)].sum()),
            'all_near_pairs_csv':'near_duplicate_pairs.csv','nearest_by_validation_trace':nearest,
            'individual_compiled_symbol_diagnostic':{
                'validation_traces_with_any_new_symbol':sum(r['new_symbol_count']>0 for r in unseen_symbols),
                'details':unseen_symbols,'note':'A new complete symbol-set signature may be only a new combination of seen symbols; exact symbol equality is still not universal workload identity.'},
            'interpretation':'Counter similarity alone does not establish acquisition reuse or label leakage; collector overlap is not unseen-family generalization. Shape equality may retain different schedules/config tokens.',
            'checkpoint_or_splits_changed':False}


def seed_insensitive_configuration_report(official, rows, supported, names):
    """Compare KernelBench problem/dtype tags across input-seed variants only.

    Source data_info.md documents a selective dtype/seed sweep. The key is a
    metadata-resolution diagnostic, not proof of identical runtime input shapes.
    Non-KernelBench collector tags are retained verbatim.
    """
    def key(path):
        parts=Path(path).parts
        config=parts[1]+'/'+('/'.join(parts[parts.index('_out')+1:-2]))
        return re.sub(r'(dt-[^/]+)_s[0-9]+$',r'\1',config) if parts[1]=='kernelbench' else config
    train_keys={}
    for row in official['train']:train_keys.setdefault(key(row['path']),[]).append(row['path'])
    val_keys=[key(row['path']) for row in official['val']]
    seen=np.array([value in train_keys for value in val_keys]);summary={}
    for label,flag in [('seen',seen),('unseen',~seen)]:
        selected=flag[rows['trace_index']]
        metric=metrics_from_counts(count_confusion(rows['true_class'][selected],rows['predicted_class'][selected]),supported,names)
        summary[label]={'traces':int(flag.sum()),'valid_bins':int(selected.sum()),
                        'metrics':metric,'meaningful_support':bool(flag.sum()>=10 and selected.sum()>=500),
                        'positive_truth_class_ids':[r['class_id'] for r in metric['per_class'] if r['support']]}
    return {'definition':'KernelBench problem/dtype key drops only _s<number> suffix on dt-... tags; other collectors retain full config',
            'source_reference':'docs/dataset/data_info.md:29 selective dtype/seed sweep',
            'groups':summary,'shared_keys':[{'key':value,'training_paths':train_keys[value],
                 'validation_paths':[r['path'] for r in official['val'] if key(r['path'])==value]}
                 for value in sorted(set(val_keys)&set(train_keys))],
            'limitation':'Same problem/dtype metadata across seed tags is not an identical counter acquisition, and does not verify all hidden runtime parameters.'}


def temporal_audit(validation,grids,supported,names):
    masks={key:np.zeros_like(validation['valid']) for key in ['interior','boundary','multiclass_traces']}
    multi=[]
    for i,y in enumerate(validation['y']):
        masks['interior'][i],masks['boundary'][i]=independent_temporal_masks(y)
        if len(set(y[y>=0].tolist()))>=2:masks['multiclass_traces'][i]=y>=0;multi.append(validation['paths'][i])
    results={}
    for kind,prediction in grids.items():
        diagnostic={};reference=read(CNN/kind/'results.json')['validation']['diagnostics']
        for key,mask in masks.items():
            cm=count_confusion(validation['y'][mask],prediction[mask]);m=metrics_from_counts(cm,supported,names)
            require(np.array_equal(cm,reference[key]['confusion_matrix']),'Temporal subset confusion mismatch')
            require(abs(m['accuracy']-reference[key]['accuracy'])<1e-12,'Temporal accuracy mismatch')
            diagnostic[key]=m
        offsets=range(-4,5);common=validation['valid'].copy();common[:,:4]=False;common[:,-4:]=False
        for offset in offsets:
            shifted=np.zeros_like(common);start=max(0,-offset);stop=min(512,512-offset)
            shifted[:,start:stop]=validation['valid'][:,start+offset:stop+offset];common &= shifted
        shift_scores=[]
        for offset in offsets:
            shifted=np.roll(prediction,-offset,axis=1)
            mask=common
            boundary=common & masks['boundary']
            shift_scores.append({'prediction_offset':offset,'global_common_support_bins':int(mask.sum()),
                'global_common_accuracy':float((validation['y'][mask]==shifted[mask]).mean()),
                'boundary_common_support_bins':int(boundary.sum()),
                'boundary_common_accuracy':float((validation['y'][boundary]==shifted[boundary]).mean()) if boundary.any() else None})
        results[kind]={'subsets':diagnostic,'shift_diagnostic_common_support':shift_scores}
    return {'status':'passed','definitions':'unchanged from frozen protocol; independently reconstructed via equality windows and adjacent-label changes',
            'multiclass_traces':len(multi),'multiclass_trace_ids':multi,'experiments':results,
            'causation_limit':'Exact source/native indexing and functional CNN alignment can rule out implementation index shifts. Classification disagreement with envelopes is observable; true instruction boundaries/annotation ambiguity cannot be established from these labels/counters alone.',
            'checkpoint_or_predictions_realigned':False}


def synthetic_loss_checks():
    from kernel_xray.phase3.cnn import masked_cross_entropy
    rng=np.random.default_rng(23);values=rng.normal(size=(1,12,512)).astype(np.float32)
    labels=np.full((1,512),-1,np.int64);labels[0,[0,5,511]]=[0,2,1]
    batch={'targets':{'workload_l1':torch.from_numpy(labels)},'masks':{'valid_l1':torch.from_numpy(labels>=0)}}
    weights=np.array([.5,2,1.25]+[1]*8+[0],np.float32)
    points=np.array([0,5,511]);z=values[0,:,points].astype(np.float64)
    logp=z-z.max(1,keepdims=True);logp-=np.log(np.exp(logp).sum(1,keepdims=True));prob=np.exp(logp)
    expected=-np.mean(logp[np.arange(3),[0,2,1]]*weights[[0,2,1]])
    logits=torch.tensor(values,requires_grad=True)
    loss=masked_cross_entropy(logits,batch,torch.from_numpy(weights));require(abs(loss.item()-expected)<1e-6,'Independent weighted CE')
    loss.backward();grad=np.zeros_like(values)
    for row,(position,cid) in enumerate(zip(points,[0,2,1])):
        term=prob[row].copy();term[cid]-=1;grad[0,:,position]=term*weights[cid]/3
    np.testing.assert_allclose(logits.grad.numpy(),grad,atol=1e-7,rtol=1e-5)
    toy=torch.nn.Parameter(torch.tensor(values));optimizer=torch.optim.AdamW([toy],lr=.001,weight_decay=.0001)
    toy.grad=torch.tensor(grad);optimizer.step()
    expected_update=values*(1-.001*.0001)-.001*grad/(np.abs(grad)+1e-8)
    np.testing.assert_allclose(toy.detach().numpy(),expected_update,atol=2e-7,rtol=1e-6)
    empty={'targets':{'workload_l1':torch.full((1,512),-1,dtype=torch.long)},'masks':{'valid_l1':torch.zeros((1,512),dtype=torch.bool)}}
    unused=torch.nn.Parameter(torch.tensor(values));opt=torch.optim.AdamW([unused],lr=.001,weight_decay=.1)
    l=masked_cross_entropy(unused,empty)
    if l is not None:l.backward();opt.step()
    require(l is None and not opt.state and np.array_equal(unused.detach().numpy(),values),'Empty update/decay not skipped')
    return {'status':'passed','manual_cross_entropy':expected,'loss_gradient_max_abs_difference':float(np.max(np.abs(logits.grad.numpy()-grad))),
            'unknown_gradients_zero':bool(not np.any(logits.grad.numpy()[:,:,1:5])),
            'adamw_first_update_matches_manual':True,'empty_batch_skips_update_and_decay':True,
            'scope':'One synthetic logit tensor update only; no CNN/frozen/model retraining'}


def main():
    torch.set_num_threads(4)
    protocol=read(OUT/'protocol.json');weights,config,source=integrity()
    official={split:read(ROOT/f'splits/{split}.json')['traces'] for split in ['train','val']}
    require(not ({r['path'] for r in official['train']}&{r['path'] for r in official['val']}),'Official trace overlap')
    stats=read(CNN/'frozen/normalization.json')
    require(digest(ROOT/'splits/train.json')==stats['train_split_sha256'],'Normalization split hash')
    require(set(stats['train_inputs_sha256'])=={r['path'] for r in official['train']},'Statistics include nontraining inputs')
    data={};layouts={}
    with guarded_reads(official):
        for split in ['train','val']:
            data[split],layouts[split]=collect(split,official[split],stats)
            require(data[split]['cache_fingerprint']==config['cache_fingerprints'][split],'Independent normalized cache differs from training')
    require(data['train']['names']==data['val']['names']==config['names'],'Vocabulary mapping mismatch')
    require(all(stats['train_inputs_sha256'][m['path']]==m['input_sha256'] for m in data['train']['metadata']),
            'Normalization source input hashes differ')
    counts=np.bincount(data['train']['y'][data['train']['valid']],minlength=12);supported=np.flatnonzero(counts>0).tolist();names=config['names']
    require(supported==list(range(11))==config['supported_classes'] and counts.tolist()==config['training_counts'],'Supported class/frequency mismatch')
    values=np.moveaxis(data['train']['raw'],1,0).reshape(24,-1).astype(np.float64)
    np.testing.assert_allclose(values.mean(1),stats['mean'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(values.std(1),stats['std'],rtol=1e-12,atol=1e-12)
    del values
    class_weights=np.zeros(12);positive=counts>0;class_weights[positive]=np.minimum(4,np.sqrt(counts.max()/counts[positive]));class_weights*=counts.sum()/np.dot(class_weights,counts)
    saved_weights=read(CNN/'class_weights.json');np.testing.assert_allclose(class_weights,saved_weights['weights'],rtol=1e-6,atol=1e-7)
    require(saved_weights['training_counts']==counts.tolist() and saved_weights['weights'][11]==0,'Weight population/other')
    save('data_isolation.json',{'status':'passed','views':layouts,'no_test_arrays_decoded':True,
         'independent_inputs_match_training_cache':True,'training_counts':counts.tolist(),
         'normalization_population_training_only':True,'independent_weights':class_weights.tolist()})
    rows,metrics,trace_cms,grids={},{},{},{}
    for kind in ['unweighted','weighted']:
        rows[kind],metrics[kind],trace_cms[kind],grids[kind]=prediction_audit(kind,data['val'],supported,names)
    save('metrics_recalculated.json',{'status':'passed','implementation':'independent np.add.at counts + TP/FP/FN harmonic arithmetic; never calls cnn_metrics',
         'experiments':metrics,'fixed_supported_ids':supported,'unsupported_other':'no training/validation positive support; output retained; F1=0'})
    replay={}
    for kind in ['unweighted','weighted']:
        state=weights if kind=='unweighted' else torch.load(CNN/kind/'best.pt',map_location='cpu',weights_only=False)['model_state']
        before=hashlib.sha256(b''.join(t.numpy().tobytes() for t in state.values())).hexdigest()
        probabilities=[]
        with torch.no_grad():
            for start in range(0,160,16):
                logits=functional_frozen_forward(torch.from_numpy(data['val']['X'][start:start+16]),state)
                require(tuple(logits.shape)==(min(16,160-start),12,512),'Replay shape')
                probabilities.append(logits.softmax(1).permute(0,2,1).numpy())
        p=np.concatenate(probabilities);ti,bi=rows[kind]['trace_index'],rows[kind]['bin_index'];actual=p[ti,bi]
        tolerances=protocol['cpu_probability_tolerance']
        np.testing.assert_allclose(actual,rows[kind]['probabilities'],**tolerances)
        predicted=actual.argmax(1);different=predicted!=rows[kind]['predicted_class']
        cm=count_confusion(rows[kind]['true_class'],predicted)
        replay[kind]={'status':'passed','device':'cpu','functional_model_independent_of_original_forward':True,
                      'max_probability_abs_difference':float(np.abs(actual-rows[kind]['probabilities']).max()),
                      'class_disagreements':int(different.sum()),'integer_confusion_equal':bool(np.array_equal(cm,metrics[kind]['confusion_matrix'])),
                      'metrics':metrics_from_counts(cm,supported,names),'tolerances':tolerances,
                      'smallest_saved_top_two_margin_among_disagreements':float(np.diff(np.sort(rows[kind]['probabilities'][different],axis=1)[:,-2:],axis=1).min()) if different.any() else None}
        require(before==hashlib.sha256(b''.join(t.numpy().tobytes() for t in state.values())).hexdigest(),'Validation mutated state')
    save('cpu_replay.json',replay)
    save('training_synthetic_checks.json',synthetic_loss_checks())
    save('training_control_checks.json',additional_control_checks(data['train'],data['val'],config))
    print('Independent CPU replay and metric reconstruction complete',flush=True)
    overlap=workloads(data['train'],data['val'],grids['unweighted'],supported,names,protocol)
    overlap['seed_insensitive_configuration']=seed_insensitive_configuration_report(official,rows['unweighted'],supported,names)
    save('workload_overlap_analysis.json',overlap)
    save('temporal_diagnostics.json',temporal_audit(data['val'],grids,supported,names))
    bootstrap=bootstrap_whole_traces(trace_cms,supported,protocol['bootstrap']['replicates'],protocol['bootstrap']['seed'])
    save('uncertainty.json',{'status':'complete','protocol':protocol['bootstrap'],'experiments':bootstrap,
                           'point_estimates':{kind:{key:metrics[kind][key] for key in ['accuracy','macro_f1_supported']} for kind in metrics},
                           'interpretation':'Conditional validation-trace resampling, not independent bins; does not cover training randomness, selection optimism, workload/hardware extrapolation or clustered related-family dependence.'})
    save('verification.json',{'status':'core_checks_passed_pending_regression_and_interpretation',
         'integrity':'source_integrity.json','metrics':'metrics_recalculated.json','inference':'cpu_replay.json',
         'data_isolation':'data_isolation.json','loss_optimizer':'training_synthetic_checks.json',
         'workloads':'workload_overlap_analysis.json','temporal':'temporal_diagnostics.json','uncertainty':'uncertainty.json',
         'retrained':False,'test_evaluated':False,'frozen_changed':False})
    print('Independent audit core complete',flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    save('verification.json',{'status':'running','retrained':False,'test_evaluated':False})
    try:
        main()
    except Exception as error:
        save('verification.json',{'status':'failed','error_type':type(error).__name__,
                                 'error':str(error),'retrained':False,'test_evaluated':False})
        raise
