import sys, copy, numpy as np
sys.path.insert(0, "/tmp/pkgcopy_r1/src")
from cbmath import *
from scipy.optimize import brentq
model,D,embed=reconciliation(); e1,e2,e3,fU,anR=D["e1"],D["e2"],D["e3"],D["fU"],D["anR"]
def run(mm,x,y):
    tr0=forward(mm,embed(1.,1.),2); mid=tr0.mid[0].copy(); mid[2]=x*e1+y*e2+3.0*anR
    t=forward(mm,embed(1.,1.),2,replace_site(0,"mid",mid))
    z=layernorm(t.resid_in[1][2],mm.layers[1].g2,mm.layers[1].b2,mm.eps)
    return margin(t), float(z@(e1+e2)), float(z@e3)
# 1. the package's PC2 pairs are not matched at the reader
_,Sa,_=run(model,0.6,0.6); _,Sd,_=run(model,1.5,-0.3)
print(f"1. package pair (0.6,0.6) vs (1.5,-0.3) at block-2 reader: S^ {Sa:.4f} vs {Sd:.4f}")
# matched post-LN
for nm,mm in (("designed",model),("block1 MLP ablated",reconciliation(block1_mlp=False)[0])):
    ma,Sa,qa=run(mm,0.6,0.6)
    t=brentq(lambda t: run(mm,1.5*t,-0.3*t)[1]-Sa,0.2,5); md,Sd,qd=run(mm,1.5*t,-0.3*t)
    print(f"   {nm}: matched S^={Sa:.6f}/{Sd:.6f}: m_agree {ma:.4f}, m_dis {md:.4f}, gap {ma-md:.4f}")
# 2. downstream re-reader block 3 (ReLU unit reading fU above threshold, subtracting)
def add_block3(mm,gain=6.0,thr=0.15):
    m3=copy.deepcopy(mm); d=mm.d
    l3=Layer(WQ=np.zeros((1,d,1)),WK=np.zeros((1,d,1)),WV=np.zeros((1,d,1)),WO=np.zeros((1,1,d)),bO=np.zeros(d),
             Win=np.outer(fU,[1.0]),bin=np.array([-thr]),Wout=np.array([-gain*fU]),bout=np.zeros(d),act="relu",attn_on=False)
    m3.layers.append(l3); return m3
m3=add_block3(model)
LV=np.linspace(-1.5,1.5,13); bad=0;n=0; ex=None
for a in LV:
    for b in LV:
        tr=forward(m3,embed(a,b),2); s=np.sign(a+b)
        if s!=0:
            n+=1
            if np.sign(margin(tr))!=s: bad+=1; ex=ex or (a,b,round(margin(tr),3))
print(f"2. +block-3 re-reader: sign(m)!=sign(S) in {bad}/{n} cells, e.g. {ex}")
ma=margin(forward(m3,embed(*[0.3*x for x in (1,1)]),2))
# does agreement now REDUCE preference at fixed S?
ma,Sa,_=run(m3,0.6,0.6); t=brentq(lambda t: run(m3,1.5*t,-0.3*t)[1]-Sa,0.2,5); md,_,_=run(m3,1.5*t,-0.3*t)
print(f"   matched-S^ pair with re-reader: m_agree {ma:.4f} m_dis {md:.4f}  (C1.2 amplification reversed: {abs(ma)<abs(md)})")
# 3. final LN with bias
mf=copy.deepcopy(model); mf.final_ln=True; mf.bf=0.5*fU*2  # margin offset +? 
bad=sum(np.sign(margin(forward(mf,embed(a,b),2)))!=np.sign(a+b) for a in LV for b in LV if a+b!=0)
print(f"3. final LN + bias (Δu·β={float(fU@mf.bf):.2f}): sign(m)!=sign(S) in {bad} cells")
# 4. perturbations
for eta in (0.03,0.1,0.3):
    e_s=[];amp=[]
    for s in range(20):
        r=np.random.default_rng(300+s); mp=copy.deepcopy(model)
        for Lp in mp.layers:
            for nm in ("WV","WO","Win","Wout"):
                A=getattr(Lp,nm); setattr(Lp,nm,A+eta*r.normal(0,1,A.shape)*np.abs(A).max())
        mp.WU=mp.WU+eta*r.normal(0,1,mp.WU.shape)*np.abs(mp.WU).max()
        e_s.append(np.mean([np.sign(margin(forward(mp,embed(a,b),2)))!=np.sign(a+b) for a in LV for b in LV if a+b!=0]))
        try:
            ma,Sa,_=run(mp,0.6,0.6); t=brentq(lambda t: run(mp,1.5*t,-0.3*t)[1]-Sa,0.05,20); md,_,_=run(mp,1.5*t,-0.3*t)
            amp.append(abs(ma)>abs(md))
        except ValueError: amp.append(np.nan)
    print(f"4. C perturbation eta={eta}: sign(m)!=sign(S) frac mean {np.mean(e_s):.3f}; matched-S^ amplification holds {np.nansum(amp)}/{len(amp)}")
# 5. parameter sweep of block 2
ok=0;tot=0
for beta in (0.5,1,2):
  for gamma in (0.5,1.5,3):
    for theta in (0.2,0.6,1.2):
        mm,_,_=reconciliation(beta=beta,gamma=gamma,theta=theta); tot+=1
        ma,Sa,_=run(mm,0.6,0.6); t=brentq(lambda t: run(mm,1.5*t,-0.3*t)[1]-Sa,0.2,5); md,_,_=run(mm,1.5*t,-0.3*t)
        ok+= (ma-md)>0.1*abs(ma)
print(f"5. block-2 sweep: matched-S^ amplification >10% in {ok}/{tot} configs")
# 6. sum-only block 1 (writes h(S) to e3 instead of q): relation-free control in the C architecture
ms=copy.deepcopy(model); L=ms.layers[0]; L.Win=1.5*np.column_stack([e1+e2,-(e1+e2)]); L.bin=np.zeros(2); L.Wout=np.stack([e3,e3])*1.0
ma,Sa,_=run(ms,0.6,0.6); t=brentq(lambda t: run(ms,1.5*t,-0.3*t)[1]-Sa,0.2,5); md,_,_=run(ms,1.5*t,-0.3*t)
print(f"6. sum-only block 1, matched S^: gap {ma-md:.2e};  binary-ish pair at raw S: ", end="")
ma2,_,_=run(ms,0.6,0.6); md2,_,_=run(ms,1.5,-0.3); print(f"raw-matched gap {ma2-md2:.4f}")
