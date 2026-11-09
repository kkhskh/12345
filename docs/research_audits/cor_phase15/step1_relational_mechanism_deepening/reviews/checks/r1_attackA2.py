import sys, copy, numpy as np
sys.path.insert(0, "/tmp/pkgcopy_r1/src")
from cbmath import *
G=[v for v in np.linspace(-1.5,1.5,13) if abs(v)>1e-9]
m0,(e1,e2,e3,e4,e5),emb,coords=agreement_family(); d=6
for sd in (0.5,1.0,3.0):
    worst=0; wmin=1; rng_=[]; badbig=0
    for s in range(50):
        r=np.random.default_rng(s); mm=copy.deepcopy(m0); L=mm.layers[0]
        L.WQ=r.normal(0,sd,(1,d,d)); L.WK=r.normal(0,sd,(1,d,d))
        P=[];
        for a in G:
            for b in G:
                tr=forward(mm,emb(a,b),2); p=tr.pattern[0][0][2]; P.append(p); wmin=min(wmin,p[:2].min())
                mg=margin(tr)
                if np.sign(mg)!=np.sign(a*b) and abs(mg)>1e-9: badbig+=1
        rng_.append(np.ptp(np.array(P)[:,:2],axis=0).max())
    print(f"single head QK sd {sd}: sign errors with |m|>1e-9: {badbig} (of {50*144}); min source weight {wmin:.2e}; median range of source weights {np.median(rng_):.3f}")
