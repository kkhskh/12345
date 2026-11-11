import json, os, sys
PK="/home/claude/12345/docs/research_audits/cor_phase15/step1_perfect_agreement_boundary"
sys.path.insert(0, PK+"/src")
import numpy as np
from scipy.optimize import minimize
import boundary as B
R, E, G = B.R, B.E, B.G
L=PK+"/logs/conf"; g=np.load(L+"/generator.npz", allow_pickle=True)
NF=2500; X,q=g["X"][:NF],g["q"][:NF]; idx=np.arange(NF)
fams=["base","F45","F34","F23"]; Xs={"base":X}|{f:g[f"X_{f}"][:NF] for f in fams[1:]}
J={}
for f in fams:
    Jf=np.zeros((NF,6,3))
    for qq in range(6):
        s=q==qq; Jf[s]=R.J_at(Xs[f][s],qq)
    J[f]=Jf
V={f:E.loo(Xs[f])[idx,q] for f in fams}
Jc=np.concatenate([J[f] for f in fams]); Vc=np.concatenate([V[f] for f in fams])
Bex=R.mix_exact(Jc,Vc)
# perfect-agreement discount feature: for each group with n=5 that votes c at q (both sources agree), -1 on c
def Dfeat(f):
    D=np.zeros((NF,3))
    for gg in range(3):
        v=B.group_vote(Xs[f],q,gg) if False else None
        a,b=G.GROUPS[gg]; xa=Xs[f][idx,q,a]; xb=Xs[f][idx,q,b]
        m=(V[f][:,gg]==5)&(xa==xb)
        D[m, xa[m]] -= 1
    return D-D.mean(1,keepdims=True)
Dc=np.concatenate([Dfeat(f) for f in fams])
meta={f:dict(o=g[f"orient_{f}"][:NF],v=np.maximum(g[f"vote_{f}"][:NF],0),r=g[f"regime_{f}"][:NF]) for f in fams[1:]}
def obs(pred):
    blk={f:pred[i*NF:(i+1)*NF] for i,f in enumerate(fams)}; bb={f:Bex[i*NF:(i+1)*NF] for i,f in enumerate(fams)}; out={}
    for f in ("F45","F34"):
        o,v,r=meta[f]["o"],meta[f]["v"],meta[f]["r"]
        e=(B.oriented_margin(blk[f],v)-B.oriented_margin(blk["base"],v))-(B.oriented_margin(bb[f],v)-B.oriented_margin(bb["base"],v))
        for rg in ("R1","R2","R3"):
            s=o&(np.char.startswith(r.astype(str),rg))
            if s.sum(): out[f"{f}{rg}"]=round(float(e[s].mean()),3)
    return out
th_ex=np.r_[np.arange(1,6)*np.log(E.RHO),np.full(5,-3.0),0.0]
res={}
for arm,seeds in (("T",range(70,78)),("P",range(80,88))):
    for sd in seeds:
        m=f"x{arm}{sd}_4000"; z=np.load(f"{L}/model_{m}.npz")
        tgt=np.concatenate([z["c_base"][:NF]]+[z[f"c_{f}"][:NF] for f in fams[1:]])
        var=((tgt-tgt.mean(0))**2).mean()*3
        # gain + discount: closed form LS on two regressors
        A=np.stack([Bex.ravel(),Dc.ravel()],1); coef,*_=np.linalg.lstsq(A,tgt.ravel(),rcond=None)
        pg=A@coef; r2gd=1-((pg-tgt.ravel())**2).mean()*3/var
        c1,*_=np.linalg.lstsq(A[:,:1],tgt.ravel(),rcond=None); r2g=1-((A[:,:1]@c1-tgt.ravel())**2).mean()*3/var
        # mix + discount
        f=lambda th: R.mix_logits(th[:11],Jc,Vc)+th[11]*Dc
        lo=lambda th: ((f(th)-tgt)**2).mean()
        rr=minimize(lo,np.r_[th_ex,0.0],method="L-BFGS-B",options=dict(maxiter=2000))
        r2md=1-rr.fun*3/var
        res[m]=dict(obs=obs(tgt),gain_r2=round(r2g,3),gd_coef=coef.round(3).tolist(),gd_r2=round(r2gd,3),gd_pred=obs(pg.reshape(tgt.shape)),
                    md_r2=round(r2md,3),md_delta=round(float(rr.x[11]),3),md_s=round(float(rr.x[4]-rr.x[3]),3),md_pred=obs(f(rr.x)))
        print(m,json.dumps(res[m]),flush=True)
json.dump(res,open("/tmp/r5/pad.json","w"),indent=1)
