import sys, copy, numpy as np
sys.path.insert(0, "/tmp/pkgcopy_r1/src")
from cbmath import *
LV=[-1.0,-0.5,0.5,1.0]; GR=np.linspace(-1.5,1.5,31)
model,D,embed=qk_gate(); fU,fB=D["fU"],D["fB"]; d=8
def m_of(mm,a,b): X,r=embed(a,b); return margin(forward(mm,X,r))
mm=copy.deepcopy(model); L=mm.layers[1]; dh=2; e0=np.array([1.,0.])
L.WQ=np.stack([L.WQ[0],np.zeros((d,dh))]); L.WK=np.stack([L.WK[0],np.zeros((d,dh))])
L.WV=np.stack([L.WV[0],np.outer(fB,e0)]); WO2=np.zeros((dh,d)); WO2[0]=-6*fU; L.WO=np.stack([L.WO[0],WO2])
agree=[(a,b) for a in GR for b in GR if a*b>0]
v=[(round(a,2),round(b,2),round(m_of(mm,a,b),3)) for a,b in agree if np.sign(m_of(mm,a,b))!=np.sign(a)]
print(f"1. extra uniform head (-6 fU): agreeing cells violating unanimity {len(v)}/{len(agree)}; e.g. {v[:3]}; binary (1,1): {m_of(mm,1,1):.3f}")
for eta in (0.03,0.1,0.3):
    fr=[]
    for s in range(20):
        r=np.random.default_rng(200+s); mp=copy.deepcopy(model)
        for Lp in mp.layers:
            for nm in ("WQ","WK","WV","WO"):
                A=getattr(Lp,nm); setattr(Lp,nm,A+eta*r.normal(0,1,A.shape)*np.abs(A).max())
        mp.WU=mp.WU+eta*r.normal(0,1,mp.WU.shape)*np.abs(mp.WU).max()
        T=lambda a,b: m_of(mp,a,b)-m_of(mp,a,0.0)
        c=[abs(T(a,b))>abs(T(a,-b)) for a in GR[::3] for b in GR[::3] if a*b>0]
        fr.append(np.mean(c))
    print(f"2. eta={eta}: frac of agreeing cells where B's contribution |T(a,b)| > |T(a,-b)| : mean {np.mean(fr):.3f} min {np.min(fr):.3f}")
