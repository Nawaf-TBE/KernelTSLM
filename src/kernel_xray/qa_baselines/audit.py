"""Independent Python feature arithmetic, linear replay, and count-based metrics."""
import math
import numpy as np


def independent_features(x,p,aware=False):
    rows=[[float(v) for v in x[c,p['start']:p['end']]] for c in p['channels']]
    basic=[];extras=[]
    for values in rows:
        mean=math.fsum(values)/len(values)
        std=math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))
        basic.append([mean,std,values[0],values[-1]])
        if aware:
            maximum=max(values);first=next(i for i,v in enumerate(values) if v==maximum)
            extras.append([min(values),maximum,first/max(1,len(values)-1)])
    output=basic[0]+basic[1]+[a-b for a,b in zip(*basic)]+[(p['end']-p['start'])/512]
    if aware:output+=extras[0]+extras[1]+[a-b for a,b in zip(*extras)]
    ids=[0.]*48;ids[p['channels'][0]]=1.;ids[24+p['channels'][1]]=1.
    return np.asarray(output+ids)


def independent_direct(x,p,cfg):
    rows=[[float(v) for v in x[c,p['start']:p['end']]] for c in p['channels']]
    if p['type']=='statistics':return math.fsum(rows[0])/len(rows[0])
    maxima=[max(v) for v in rows]
    if p['type']=='comparison':
        if maxima[0]==maxima[1]:return 'Tie'
        assert abs(maxima[0]-maxima[1])>cfg['comparison_atol']+cfg['comparison_rtol']*max(map(abs,maxima))
        return 'Counter A' if maxima[0]>maxima[1] else 'Counter B'
    peaks=[]
    for values,maximum in zip(rows,maxima):
        indices=[i for i,v in enumerate(values) if v==maximum]
        ordered=sorted(values,reverse=True)
        assert len(indices)==1 and ordered[0]-ordered[1]>cfg['peak_atol']+cfg['peak_rtol']*abs(maximum)
        peaks.append(indices[0])
    return 'Yes' if peaks[0]<peaks[1] else ('No' if peaks[0]>peaks[1] else 'Same bin')


def linear_replay(fit,matrix,identity_only=False):
    n=fit['n_continuous'];scaler=fit['scaler'];model=fit['model']
    normalized=np.zeros_like(matrix[:,:n]) if identity_only else (matrix[:,:n]-scaler.mean_)/scaler.scale_
    score=np.column_stack((normalized,matrix[:,n:]))@model.coef_.T+model.intercept_
    indices=(score[:,0]>0).astype(int) if len(model.classes_)==2 else score.argmax(1)
    return model.classes_[indices]


def categorical(truth,predicted,classes):
    counts=[[0 for c in classes] for c in classes];mapping={c:i for i,c in enumerate(classes)}
    for y,p in zip(truth,predicted,strict=True):counts[mapping[y]][mapping[p]]+=1
    rows={}
    for i,c in enumerate(classes):
        tp=counts[i][i];support=sum(counts[i]);positive=sum(row[i] for row in counts)
        precision=tp/positive if positive else 0.;recall=tp/support if support else 0.
        rows[c]={'precision':precision,'recall':recall,'f1':2*tp/(support+positive) if support+positive else 0.,'support':support}
    return {'n':len(truth),'classes':list(classes),'accuracy':sum(counts[i][i] for i in range(len(classes)))/len(truth),
        'balanced_accuracy':math.fsum(r['recall'] for r in rows.values())/len(classes),
        'macro_f1':math.fsum(r['f1'] for r in rows.values())/len(classes),
        'confusion_matrix':counts,'per_answer':rows}


def numerical(truth,predicted,cfg):
    errors=sorted(abs(float(y)-float(p)) for y,p in zip(truth,predicted,strict=True));n=len(errors)
    return {'n':n,'mae':math.fsum(errors)/n,'median_absolute_error':(errors[(n-1)//2]+errors[n//2])/2,
        'rmse':math.sqrt(math.fsum(v*v for v in errors)/n),
        'within_tolerance':sum(abs(y-p)<=cfg['answer_atol']+cfg['answer_rtol']*abs(y) for y,p in zip(truth,predicted))/n}


def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(int,float)):
        assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
    else:assert a==b,(a,b)


def bootstrap(predictions,trace_ids,cfg,qa_cfg):
    """Common whole-trace draws for every model/task/condition; pooled metrics."""
    index={t:i for i,t in enumerate(trace_ids)};n=len(index)
    weights=np.random.default_rng(cfg['seed']).multinomial(n,np.full(n,1/n),size=cfg['replicates'])
    results={};distributions={}
    for key,rows in predictions.items():
        kind=key.split('/')[0];ti=np.asarray([index[r['trace_id']] for r in rows]);truth=[r['reference'] for r in rows];pred=[r['prediction'] for r in rows]
        if kind!='statistics':
            classes=sorted(set(truth));mapping={c:i for i,c in enumerate(classes)};c=len(classes)
            trace_cm=np.zeros((n,c,c),dtype=np.int64)
            for t,y,p in zip(ti,truth,pred):trace_cm[t,mapping[y],mapping[p]]+=1
            cm=np.einsum('bt,tij->bij',weights,trace_cm);tp=np.diagonal(cm,axis1=1,axis2=2);support=cm.sum(2);positive=cm.sum(1)
            recall=np.divide(tp,support,out=np.zeros_like(tp,dtype=float),where=support>0)
            precision=np.divide(tp,positive,out=np.zeros_like(tp,dtype=float),where=positive>0)
            f1=np.divide(2*tp,support+positive,out=np.zeros_like(tp,dtype=float),where=support+positive>0)
            values={'accuracy':tp.sum(1)/cm.sum((1,2)),'balanced_accuracy':recall.mean(1),'macro_f1':f1.mean(1)}
            for i,c in enumerate(classes):
                for name,a in [('precision',precision),('recall',recall),('f1',f1)]:values[f'{c}/{name}']=a[:,i]
            missing={'any_supported_class_absent':int((support==0).any(1).sum()),'per_class':{c:int((support[:,i]==0).sum()) for i,c in enumerate(classes)}}
        else:
            truth=np.asarray(truth);error=np.abs(truth-np.asarray(pred));ok=error<=qa_cfg['answer_atol']+qa_cfg['answer_rtol']*abs(truth)
            totals=np.bincount(ti,minlength=n);denominator=weights@totals
            values={'mae':(weights@np.bincount(ti,weights=error,minlength=n))/denominator,
                'rmse':np.sqrt((weights@np.bincount(ti,weights=error**2,minlength=n))/denominator),
                'within_tolerance':(weights@np.bincount(ti,weights=ok.astype(int),minlength=n))/denominator}
            order=np.argsort(error);cumulative=np.cumsum(weights[:,ti[order]],axis=1)
            lo=(denominator-1)//2;hi=denominator//2
            left=(cumulative>lo[:,None]).argmax(1);right=(cumulative>hi[:,None]).argmax(1)
            values['median_absolute_error']=(error[order[left]]+error[order[right]])/2;missing={}
        distributions[key]=values
        results[key]={'percentile95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in values.items()},'missing_class_replicates':missing}
    pairs={}
    for kind in ('comparison','temporal'):
        for a,b in [('lookup','majority'),('primary','lookup'),('aware','lookup'),('aware','primary'),('direct','lookup')]:
            ka=f'{kind}/{a}/original';kb=f'{kind}/{b}/original'
            pairs[f'{ka} minus {kb}']={k:np.quantile(distributions[ka][k]-distributions[kb][k],[.025,.975]).tolist() for k in ('accuracy','balanced_accuracy','macro_f1')}
        for model in ('primary','aware','direct'):
            a=f'{kind}/{model}/reversed_original';b=f'{kind}/{model}/original'
            pairs[f'{a} minus {b}']={k:np.quantile(distributions[a][k]-distributions[b][k],[.025,.975]).tolist() for k in ('accuracy','balanced_accuracy','macro_f1')}
    return {'protocol':cfg,'units':n,'metrics':results,'paired_differences_percentile95':pairs,
        'limitations':'Conditional on frozen models/QA cohort; not training-seed, selection or unseen-workload uncertainty; shared workload families can violate trace independence.'}
