import sys, copy, numpy as np
sys.path.insert(0, "/tmp/pkgcopy_r1/src")
from cbmath import *
GR=np.linspace(-1.5,1.5,31); LV=[-1.0,-0.5,0.5,1.0]
mb,D,emb=qk_gate(final_ln=True,kappa_bias=1.0)
v=sum(np.sign(margin(forward(mb,*emb(a,b))))!=np.sign(a) for a in GR for b in GR if a*b>0)
print("B final LN+bias 1.0: unanimity violations",v,"of",sum(1 for a in GR for b in GR if a*b>0))
mo,_,_=qk_gate(qk_off=True)
T=np.array([[margin(forward(mo,*emb(a,b))) for b in LV] for a in LV]); print("PB6 qk_off margin range",T.min().round(3),T.max().round(3))
# B: B1 requires sign x^ = sign a; is it robust? layer-1 attention to A not exact; check x^ sign on grid
# C perturbation error locations
model,Dc,embc=reconciliation(); LVc=np.linspace(-1.5,1.5,13)
locs=[]
for s in range(20):
    r=np.random.default_rng(300+s); mp=copy.deepcopy(model)
    for Lp in mp.layers:
        for nm in ("WV","WO","Win","Wout"):
            A=getattr(Lp,nm); setattr(Lp,nm,A+0.03*r.normal(0,1,A.shape)*np.abs(A).max())
    mp.WU=mp.WU+0.03*r.normal(0,1,mp.WU.shape)*np.abs(mp.WU).max()
    locs+= [abs(a+b) for a in LVc for b in LVc if a+b!=0 and np.sign(margin(forward(mp,embc(a,b),2)))!=np.sign(a+b)]
print("C eta=0.03 errors: n",len(locs),"median |a+b|",np.median(locs),"max",max(locs))
# unperturbed C sign(m) vs sign(a+b)
print("C unperturbed errors vs sign(a+b):",sum(np.sign(margin(forward(model,embc(a,b),2)))!=np.sign(a+b) for a in LVc for b in LVc if abs(a+b)>1e-9))
# reachability of PA8 state x=3 for d=6 agreement_family
print("sup |x| reachable in agreement_family d=6:",np.sqrt(6))
m,(e1,e2,e3,e4,e5),em,coords=agreement_family()
print("coords at a=50:",coords(forward(m,em(50.0,0.0),2)))
