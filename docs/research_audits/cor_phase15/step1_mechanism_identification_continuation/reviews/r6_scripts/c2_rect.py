import sys, math, dataclasses, numpy as np
sys.path.insert(0, sys.argv[1])
from cprime import P, closed_form, certify
import cbmath
p=P()
try: closed_form(1.0,1.0,dataclasses.replace(p,lam=400.0)); print('lam=400 ok')
except OverflowError as e: print('OverflowError closed_form(1,1,lam=400):', e)
import itertools
for L in [100.0,200.0,300.0]:
    try: print("lam",L,closed_form(1.0,1.0,dataclasses.replace(p,lam=L))["m"])
    except OverflowError: print("lam",L,"OverflowError")
# dense sampling inside certified rectangles (float64 closed form)
rng=np.random.default_rng(99)
rects={'Q++':(0.09,3,0.09,3,1),'Q--':(-3,0,-3,0,-1),'Q+-':(0.02,3,-3,-0.02,-1),'Q-+':(-3,-0.02,0.02,3,-1)}
def cf_vec(a,b):
    return np.array([closed_form(x,y)['m'] for x,y in zip(a,b)])
for k,(a0,a1,b0,b1,want) in rects.items():
    a=np.concatenate([rng.uniform(a0,a1,40000), np.full(5000,a0), np.full(5000,a1), rng.uniform(a0,a1,5000)])
    b=np.concatenate([rng.uniform(b0,b1,40000), rng.uniform(b0,b1,5000), rng.uniform(b0,b1,5000), np.full(5000, b0 if want>0 or k=='Q--' else (b1 if k=='Q+-' else b0))])
    m=cf_vec(a,b); bad=(np.sign(m)!=want).sum()
    i=np.argmin(want*m)
    print(k,'samples',len(a),'wrong sign',bad,'min want*m',want*m[i],'at',a[i],b[i])
# float32 forward pass at the rectangle corners / near-boundary points
model,D,emb=cbmath.rereading()
def f32(a,b):
    import copy
    X=emb(a,b).astype(np.float32)
    # run forward in float32 by casting all params
    mm=copy.deepcopy(model)
    for L in mm.layers:
        for f in L.__dataclass_fields__:
            v=getattr(L,f)
            if isinstance(v,np.ndarray): setattr(L,f,v.astype(np.float32))
    mm.WU=mm.WU.astype(np.float32); mm.bU=mm.bU.astype(np.float32); mm.eps=np.float32(mm.eps)
    return cbmath.margin(cbmath.forward(mm,X,3))
worst={}
for k,(a0,a1,b0,b1,want) in rects.items():
    pts=[(a0 if want>0 else a1, b0 if want>0 else b1)]
    if k=='Q+-': pts=[(a1,b1),(a0,b1),(a1,-0.0201)]
    if k=='Q-+': pts=[(a0,b0),(a1,b0)]
    if k=='Q++': pts=[(0.09,0.09),(0.09,3.0),(3.0,0.09),(0.0901,0.2)]
    for a,b in pts:
        m64=closed_form(a,b)['m']; m32=f32(a,b)
        print(k,(a,b),'m64',m64,'m32',m32,'dtype',type(m32))
# conflict boundary b*(a) beyond 3 and the ratio region beyond 3.05
def bis(f,lo,hi):
    for _ in range(100):
        mid=(lo+hi)/2
        if (f(mid)>0)==(f(lo)>0): lo=mid
        else: hi=mid
    return (lo+hi)/2
for a in [3,5,10,20,50,100,1000]:
    f=lambda b: closed_form(a,b)['m']
    g=np.linspace(-a,-1e-9,4001); v=[f(x) for x in g]
    roots=[bis(f,g[i],g[i+1]) for i in range(len(g)-1) if (v[i]>0)!=(v[i+1]>0)]
    print('conflict a=',a,'roots b*',roots,'ratio',[-r/a for r in roots])
for b in [3,5,10,50]:
    f=lambda a: closed_form(a,b)['m']
    g=np.linspace(-b,-1e-9,4001); v=[f(x) for x in g]
    roots=[bis(f,g[i],g[i+1]) for i in range(len(g)-1) if (v[i]>0)!=(v[i+1]>0)]
    print('conflict b=',b,'roots a*',roots)
# agreement: large a, small b
for a in [3,10,50,1000]:
    f=lambda b: closed_form(a,b)['m']
    g=np.linspace(1e-9,1,2001); v=[f(x) for x in g]
    roots=[bis(f,g[i],g[i+1]) for i in range(len(g)-1) if (v[i]>0)!=(v[i+1]>0)]
    print('agree a=',a,'roots b*',roots, 'm(a,0.09)=',f(0.09))
