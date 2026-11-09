import sys, dataclasses, numpy as np, math
sys.path.insert(0,'/home/claude/12345/docs/research_audits/cor_phase15/step1_mechanism_identification_continuation/src')
from cprime import *
sys.path.insert(0,'/home/claude/12345/docs/research_audits/cor_phase15/step1_mechanism_identification_continuation/src')
from cprime_regions import suff, target
rng=np.random.default_rng(7); worst=0; viol=0; n=0
for _ in range(60):
    p=dataclasses.replace(P(), lam=float(rng.uniform(0.1,50)), mu=float(rng.uniform(0.1,8)), kappa=float(rng.uniform(0,4)),
        rho=float(rng.uniform(0.1,6)), lam1=float(rng.uniform(0.1,6)), c1=float(rng.uniform(0.1,6)), wv=float(rng.uniform(0.05,4)), prior_v=float(rng.uniform(0.05,5)))
    for a,b in rng.uniform(-50,50,size=(30,2)).tolist()+rng.uniform(-0.05,0.05,size=(30,2)).tolist():
        m=closed_form(a,b,p)['m']; f=forward_margin(a,b,p)
        worst=max(worst,abs(m-f)/(1+abs(f)))
        s=suff(a,b,p); n+=1
        if s!=0 and s!=(1 if m>0 else -1): viol+=1; print("viol",p,a,b,m)
print("closed form vs forward, random params incl |a|<=50: max rel diff",worst, "suff violations",viol,"/",n)
