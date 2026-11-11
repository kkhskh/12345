import json,sys,numpy as np
exec(open("pad.py").read().split("th_ex=")[0])
from scipy.optimize import minimize
th_ex=np.r_[np.arange(1,6)*np.log(E.RHO),np.full(5,-3.0),0.0]
for sd in range(80,88):
    m=f"xP{sd}_4000"; z=np.load(f"{L}/model_{m}.npz")
    tgt=np.concatenate([z["c_base"][:NF]]+[z[f"c_{f}"][:NF] for f in fams[1:]])
    dT=np.concatenate([tgt[NF*i:NF*(i+1)]-tgt[:NF] for i in (1,2,3)])
    def lo(th):
        p=R.mix_logits(th,Jc,Vc); d=np.concatenate([p[NF*i:NF*(i+1)]-p[:NF] for i in (1,2,3)])
        return ((d-dT)**2).mean()+0.2*((p-tgt)**2).mean()
    best=None
    for s0 in (-2.0,0.0,1.36):
        t0=th_ex.copy(); t0[4]=t0[3]+s0
        rr=minimize(lo,t0,method="L-BFGS-B",options=dict(maxiter=1500))
        if best is None or rr.fun<best.fun: best=rr
    p=R.mix_logits(best.x,Jc,Vc); var=((tgt-tgt.mean(0))**2).mean()*3
    print(m,"obs",obs(tgt)["F45R1"],"pred",obs(p),"s=%.2f r2_surface=%.3f"%(best.x[4]-best.x[3],1-((p-tgt)**2).mean()*3/var),flush=True)
