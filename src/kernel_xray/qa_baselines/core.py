"""Features and deterministic control accept only normalized arrays and safe parameters."""
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score,balanced_accuracy_score


def parameters(row):
    return {k:row[k] for k in ('type','channels','start','end')}


def validate(x,p):
    if set(p)!={'type','channels','start','end'}:raise ValueError('Unsafe parameter keys')
    if x.shape!=(24,512) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Input schema')
    if p['type'] not in ('statistics','comparison','temporal'):raise ValueError('Task')
    cs=p['channels'];s=p['start'];e=p['end']
    if len(cs)!=(1 if p['type']=='statistics' else 2) or any(type(c)!=int or not 0<=c<24 for c in cs) or len(set(cs))!=len(cs):raise ValueError('Channels')
    if type(s)!=int or type(e)!=int or not 0<=s<e<=512:raise ValueError('Interval')


def features(x,p,aware=False):
    validate(x,p)
    if p['type']=='statistics':raise ValueError('Classifier features are categorical-only')
    vals=x[p['channels'],p['start']:p['end']].astype(np.float64)
    basic=np.column_stack((vals.mean(1),vals.std(1),vals[:,0],vals[:,-1]))
    continuous=np.r_[basic.ravel(),basic[0]-basic[1],(p['end']-p['start'])/512]
    if aware:
        extra=np.column_stack((vals.min(1),vals.max(1),vals.argmax(1)/max(1,vals.shape[1]-1)))
        continuous=np.r_[continuous,extra.ravel(),extra[0]-extra[1]]
    ids=np.zeros(48);ids[p['channels'][0]]=1;ids[24+p['channels'][1]]=1
    return np.r_[continuous,ids]


def reverse_interval(x,p):
    validate(x,p);result=x.copy();s,e=p['start'],p['end']
    for c in p['channels']:result[c,s:e]=x[c,s:e][::-1]
    return result


def direct(x,p,cfg):
    validate(x,p);v=x[p['channels'],p['start']:p['end']].astype(np.float64)
    if p['type']=='statistics':return float(v[0].mean())
    mx=v.max(1)
    if p['type']=='comparison':
        gap=abs(mx[0]-mx[1]);threshold=cfg['comparison_atol']+cfg['comparison_rtol']*max(abs(mx))
        if gap==0:return 'Tie'
        if gap<=threshold:raise ValueError('Ambiguous comparison')
        return 'Counter A' if mx[0]>mx[1] else 'Counter B'
    if v.shape[1]<2:raise ValueError('Peak interval')
    ordered=np.sort(v,axis=1)
    if np.any(ordered[:,-1]-ordered[:,-2]<=cfg['peak_atol']+cfg['peak_rtol']*abs(mx)):raise ValueError('Ambiguous peak')
    peaks=v.argmax(1)
    return 'Yes' if peaks[0]<peaks[1] else ('No' if peaks[0]>peaks[1] else 'Same bin')


def fit_rules(rows):
    counts={};pairs={}
    for kind in ('comparison','temporal'):
        selected=[r for r in rows if r['type']==kind]
        counts[kind]=Counter(r['answer'] for r in selected).most_common(1)[0][0]
        local={}
        for row in selected:local.setdefault(tuple(row['channels']),Counter())[row['answer']]+=1
        pairs[kind]={','.join(map(str,k)):v.most_common(1)[0][0] for k,v in local.items()}
    return {'majority':counts,'pairs':pairs,'constant':float(np.median([r['answer'] for r in rows if r['type']=='statistics']))}


def rule_predict(rules,kind,channels,pair=False):
    default=rules['majority'][kind]
    return rules['pairs'][kind].get(','.join(map(str,channels)),default) if pair else default


def fit_classifier(matrix,answers,config):
    n=matrix.shape[1]-48;scaler=StandardScaler().fit(matrix[:,:n])
    design=np.column_stack((scaler.transform(matrix[:,:n]),matrix[:,n:]))
    model=LogisticRegression(**config).fit(design,answers)
    return {'scaler':scaler,'model':model,'n_continuous':n}


def predict_classifier(fit,matrix,identities_only=False):
    n=fit['n_continuous'];continuous=np.zeros_like(matrix[:,:n]) if identities_only else fit['scaler'].transform(matrix[:,:n])
    return fit['model'].predict(np.column_stack((continuous,matrix[:,n:])))


def categorical_metrics(truth,predicted,classes):
    precision,recall,f1,support=precision_recall_fscore_support(truth,predicted,labels=classes,zero_division=0)
    return {'n':len(truth),'classes':list(classes),'accuracy':float(accuracy_score(truth,predicted)),
            'balanced_accuracy':float(balanced_accuracy_score(truth,predicted)),
            'macro_f1':float(f1.mean()),'confusion_matrix':confusion_matrix(truth,predicted,labels=classes).tolist(),
            'per_answer':{c:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,c in enumerate(classes)}}


def numerical_metrics(truth,predicted,cfg):
    error=np.abs(np.asarray(truth)-predicted)
    return {'n':len(error),'mae':float(error.mean()),'median_absolute_error':float(np.median(error)),
            'rmse':float(np.sqrt(np.mean(error**2))),
            'within_tolerance':float(np.mean(error<=cfg['answer_atol']+cfg['answer_rtol']*np.abs(truth)))}
