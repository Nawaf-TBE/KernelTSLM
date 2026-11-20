"""Safe JSON numerical model interchange. No pickle, joblib or executable loading."""
import json
import math
from pathlib import Path


def feature_names(variant):
    basic=['mean','population_std','first','last'];extra=['min','max','first_peak_fraction']
    names=[f'{side}.{s}' for side in ('A','B') for s in basic]+[f'A_minus_B.{s}' for s in basic]+['length_over_512']
    if variant=='aware':names += [f'{side}.{s}' for side in ('A','B') for s in extra]+[f'A_minus_B.{s}' for s in extra]
    elif variant!='primary':raise ValueError('Variant')
    return names+[f'{side}.counter_{c}' for side in ('A','B') for c in range(24)]


class PortableModel:
    def __init__(self,metadata):
        self.metadata=metadata
        variant=metadata['variant'];n=61 if variant=='primary' else 70
        if metadata['feature_names']!=feature_names(variant):raise ValueError('Feature order')
        classes=metadata['classes'];rows=1 if len(classes)==2 else len(classes)
        if metadata['task'] not in ('comparison','temporal') or len(set(classes))!=len(classes):raise ValueError('Classes')
        if len(classes)!=(2 if metadata['task']=='comparison' else 3):raise ValueError('Class count')
        for key,length in [('mean',n-48),('scale',n-48),('variance',n-48)]:
            v=metadata['scaler'][key]
            if len(v)!=length or not all(type(x) in (int,float) and math.isfinite(x) for x in v):raise ValueError('Scaler')
        if any(v<=0 for v in metadata['scaler']['scale']):raise ValueError('Scale')
        if len(metadata['coefficients'])!=rows or any(len(row)!=n for row in metadata['coefficients']) or len(metadata['intercepts'])!=rows:raise ValueError('Coefficient shape')
        if not all(math.isfinite(x) for row in metadata['coefficients'] for x in row) or not all(math.isfinite(x) for x in metadata['intercepts']):raise ValueError('Nonfinite coefficient')

    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))

    def predict_proba(self,features):
        import numpy as np
        m=self.metadata;n=len(m['scaler']['mean']);x=np.asarray(features,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(m['feature_names']) or not np.isfinite(x).all():raise ValueError('Feature matrix')
        normalized=(x[:,:n]-m['scaler']['mean'])/m['scaler']['scale']
        score=np.column_stack((normalized,x[:,n:]))@np.asarray(m['coefficients']).T+m['intercepts']
        if len(m['classes'])==2:
            z=score[:,0];p=np.empty_like(z);positive=z>=0
            p[positive]=1/(1+np.exp(-z[positive]));e=np.exp(z[~positive]);p[~positive]=e/(1+e)
            return np.column_stack((1-p,p))
        exp=np.exp(score-score.max(1,keepdims=True));return exp/exp.sum(1,keepdims=True)

    def predict(self,features):
        import numpy as np
        # sklearn binary predict uses strict score>0, so exact ties select class 0.
        m=self.metadata;n=len(m['scaler']['mean']);x=np.asarray(features,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(m['feature_names']) or not np.isfinite(x).all():raise ValueError('Feature matrix')
        normalized=(x[:,:n]-m['scaler']['mean'])/m['scaler']['scale']
        scores=np.column_stack((normalized,x[:,n:]))@np.asarray(m['coefficients']).T+m['intercepts']
        indices=(scores[:,0]>0).astype(int) if len(m['classes'])==2 else scores.argmax(1)
        return np.asarray(m['classes'])[indices]
