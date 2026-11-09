import sys, copy, numpy as np
sys.path.insert(0, "/tmp/pkgcopy_r1/src")
from cbmath import *
LV=[-1.0,-0.5,0.5,1.0]
GR=np.linspace(-1.5,1.5,31)
model,D,embed=qk_gate(); fU,fB,fQ=D["fU"],D["fB"],D["fQ"]; d=8
def m_of(mm,a,b): X,r=embed(a,b); return margin(forward(mm,X,r))
# 1. extra layer-2 head with W_Q=0 (uniform routing), value at B writes -kk*y on fU: within B1-B4 as literally stated
mm=copy.deepcopy(model); L=mm.layers[1]
dh=2; e0=np.array([1.,0.])
L.WQ=np.stack([L.WQ[0],np.zeros((d,dh))]); L.WK=np.stack([L.WK[0],np.zeros((d,dh))])
L.WV=np.stack([L.WV[0],np.outer(fB,e0)]); WO2=np.zeros((dh,d)); WO2[0]=-4.5*fU; L.WO=np.stack([L.WO[0],WO2])
viol=[(a,b,round(m_of(mm,a,b),3)) for a in LV for b in LV if a*b>0 and np.sign(m_of(mm,a,b))!=np.sign(a)]
print("1. extra uniform head: agreeing cells with sign(m)!=shared sign:",viol)
# 2. perturbations of all weight matrices
for eta in (0.03,0.1,0.3):
    un=[];rv=[]
    for s in range(20):
        r=np.random.default_rng(200+s); mp=copy.deepcopy(model)
        for Lp in mp.layers:
            for nm in ("WQ","WK","WV","WO"):
                A=getattr(Lp,nm); setattr(Lp,nm,A+eta*r.normal(0,1,A.shape)*np.abs(A).max())
        mp.WU=mp.WU+eta*r.normal(0,1,mp.WU.shape)*np.abs(mp.WU).max()
        un.append(np.mean([np.sign(m_of(mp,a,b))==np.sign(a) for a in GR for b in GR if a*b>0]))
        # gate effect: wB increasing in ab?
        gaps=[]
        for a in (0.5,1.0):
            for b in (0.5,1.0):
                gaps.append(abs(m_of(mp,a,b)-m_of(mp,a,-b)) > abs(m_of(mp,-a,b)-m_of(mp,-a,-b)) )
        rv.append(np.mean(gaps))
    print(f"2. B perturbation eta={eta}: unanimity accuracy mean {np.mean(un):.3f} min {np.min(un):.3f}; 'B counts more when agreeing with A>0' frac {np.mean(rv):.2f}")
# 3. parameter sweep: does B's structure (PB2, PB5, PB8-type) hold over parameter ranges
cnt=0;ok2=0;ok5=0
for lam in (1,2,4,8):
  for wU in (0.2,0.6,1.5):
    for kB in (0.5,1.5,3):
        mm,_,_=qk_gate(lam=lam,wU=wU,kB=kB); mo,_,_=qk_gate(lam=lam,wU=wU,kB=kB,qk_off=True)
        cnt+=1
        ok2+=all(np.sign(m_of(mm,a,b))==np.sign(a) for a in LV for b in LV if a*b>0)
        ok5+=any(np.sign(m_of(mo,a,b))==np.sign(b) and np.sign(m_of(mm,a,b))==np.sign(a) for a in GR for b in GR if a*b<0)
print(f"3. sweep lam,wU,kB ({cnt} configs): PB2 holds {ok2}, PB5 reversal band nonempty {ok5}")
# 4. B with final LN no bias, B with layer-1 MLP absent -> check exact formula residual when final LN
mf,_,_=qk_gate(final_ln=True)
print("4. final LN (no bias): unanimity", all(np.sign(m_of(mf,a,b))==np.sign(a) for a in GR for b in GR if a*b>0))
