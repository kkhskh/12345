import sys, copy, numpy as np
sys.path.insert(0, "/tmp/pkgcopy_r1/src")
from cbmath import *
G=[v for v in np.linspace(-1.5,1.5,13) if abs(v)>1e-9]
def err(mm, emb, ref=lambda a,b: np.sign(a*b)):
    return np.mean([np.sign(margin(forward(mm, emb(a,b),2)))!=ref(a,b) for a in G for b in G])
# (a) moving attention, single head, random nonzero Q/K
m0,(e1,e2,e3,e4,e5),emb,coords=agreement_family()
d=6; worst=0; pats=[]
for s in range(50):
    r=np.random.default_rng(s); mm=copy.deepcopy(m0); L=mm.layers[0]
    L.WQ=r.normal(0,3,(1,d,d)); L.WK=r.normal(0,3,(1,d,d))
    e=err(mm,emb); worst=max(worst,e)
    A=[forward(mm,emb(a,b),2).pattern[0][0][2] for a in G for b in G]
    pats.append(np.ptp(np.array(A),axis=0).max())
print("(a) single head random QK sd3: worst grid err",worst," median pattern range over grid",np.median(pats), "min attn wt", )
# (b) two heads: head0 uniform +3 a e1 + 3 b e2 ; head1 attends to A iff b<0 and writes -9 a e1
mm=copy.deepcopy(m0); L=mm.layers[0]
WQ=np.zeros((2,d,d)); WK=np.zeros((2,d,d)); WV=np.zeros((2,d,d)); WO=np.zeros((2,d,d))
WV[0]=np.eye(d); WO[0]=3*(np.outer(e1,e1)+np.outer(e2,e2))
# head1: query from read anchor e5, key: A position has kappa e4 & a e1; B has b e2.  key reads -e2 (so B scores high when b<0?) -> we want attend A when b<0: score(B) = +20*b, score(A)=0
WQ[1]=np.outer(e5,e1)*1.0; WK[1]=np.outer(e2,e1)*40.0
WV[1]=np.eye(d); WO[1]=-9*np.outer(e1,e1)
L.WQ,L.WK,L.WV,L.WO=WQ,WK,WV,WO
rows=[]
for a,b in [(1,1),(1,-1),(-1,1),(-1,-1),(0.5,-0.5),(-0.5,-0.5)]:
    tr=forward(mm,emb(a,b),2); x,y=coords(tr)
    rows.append((a,b,round(margin(tr),3),round(x,3),round(y,3),np.round(tr.pattern[0][1][2],3).tolist()))
for r_ in rows: print("(b)",r_)
print("(b) grid err vs sign(ab):",err(mm,emb)," vs sign(a)*1:",err(mm,emb,ref=lambda a,b: np.sign(a)))
# (c) 3 candidates: unembed u1=e3,u2=-e3,u3 = 0.5 e3 + bias
m3,_,emb3,_=agreement_family()
U=np.stack([e3,-e3,0.5*e3],1); m3.WU=U; m3.bU=np.array([0,0,0.0])
dec={}
for a,b in [(1,1),(-1,-1),(1,-1),(-1,1)]:
    lg=forward(m3,emb3(a,b),2).logits; dec[(a,b)]=(int(np.argmax(lg)),np.round(lg,3).tolist())
print("(c) 3 candidates, no bias:",dec)
m3.bU=np.array([0,0,0.5])
dec={}
for a,b in [(1,1),(-1,-1),(0.3,0.3),(1,-1)]:
    lg=forward(m3,emb3(a,b),2).logits; dec[(a,b)]=int(np.argmax(lg))
print("(c) 3 candidates, bias 0.5 on c3:",dec)
# (d) PA11 null: errors uniformly spread
absv=sorted(set(round(abs(v),3) for v in G))
cells=[(a,b) for a in G for b in G]
mins=np.array([min(abs(a),abs(b)) for a,b in cells]); print("(d) PA11 null: median min(|a|,|b|) over all cells =",np.median(mins), "; random 5% subsets: max median", max(np.median(np.random.default_rng(i).choice(mins,7)) for i in range(2000)))
# (e) final LN with non-uniform gain
mm,_,emb,_=agreement_family(final_ln=True)
r=np.random.default_rng(0)
for sd in (0.1,0.3):
    es=[]
    for s in range(20):
        mx=copy.deepcopy(mm); mx.gf=1+np.random.default_rng(s).normal(0,sd,d); es.append(err(mx,emb))
    print(f"(e) final LN gain 1+N(0,{sd}): mean grid err {np.mean(es):.3f}, max {np.max(es):.3f}")
# (f) zero-bias perturbation only (all else exact)
for sd in (0.01,0.05,0.1):
    es=[]
    for s in range(20):
        mx=copy.deepcopy(m0); mx.layers[0].bin=np.random.default_rng(s).normal(0,sd,4); es.append(err(mx,emb:=agreement_family()[2]))
    print(f"(f) MLP bias N(0,{sd}) only: mean grid err {np.mean(es):.3f}")
