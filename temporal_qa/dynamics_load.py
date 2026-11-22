import numpy as np, os
from .paths import LABEL_ROOT as ROOT, NPZ
D=np.load(NPZ)
X=D['x']; SPLIT=D['split']; Y=D['y']; CH=[str(c) for c in D['channels']]; LAB=[str(l) for l in D['labels']]; PATHS=D['paths']
CACHE=os.path.join(os.path.dirname(os.path.abspath(__file__)),'labels_cache.npz')
def _build():
    keys=['workload_l1','mask_any_kernel']
    out={k:[] for k in keys}
    for p in PATHS:
        L=np.load(ROOT+str(p).replace('input/tensor_input.npz','labels/labels.npz'))
        for k in keys: out[k].append(L[k])
    np.savez(CACHE,**{k:np.stack(v) for k,v in out.items()})
if not os.path.exists(CACHE): _build()
_C=np.load(CACHE); WL1=_C['workload_l1']; ACTIVE=_C['mask_any_kernel'].astype(bool)
