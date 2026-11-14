import sys, os
PK="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation"; sys.path.insert(0,PK+"/src"); os.chdir(PK)
import numpy as np, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
env=GL.Env(3,1); rng=np.random.default_rng(74_100_000); N_=2000
d=NV.Bm.sample_fast(NV.ENV,3*N_,rng); X=d["X"]; q=rng.integers(0,6,len(X)); ii=np.arange(len(X))
rel,col,noi=GL._group_lik(env,X); relm=rel.mean(-1); lr=np.log(relm/noi); W=np.exp(lr.sum(1)-lr[ii,q]); Lg=rel[ii,q]/noi
a_=X.reshape(len(X),6,3,2)[ii,q]; e=((GL.counts(X,3)[ii,q]==5)&(a_[...,0]==a_[...,1])).astype(float); S=e.sum(1,keepdims=True)
rB=W*(S-e)*Lg.sum(-1); rB/=rB.sum(1,keepdims=True)
pf=(GL.counts(X,3)[ii,q]==5).astype(float); nrm=lambda w:w/np.maximum(w.sum(1,keepdims=True),1e-300)
rA1=nrm(W*np.maximum(pf.sum(1,keepdims=True)-pf,0)*Lg.sum(-1)+1e-300)
base=np.arange(N_); donor=np.array([rng.choice(np.where(q==q[b])[0]) for b in base])
m=(pf[donor]!=e[donor]).any(1); dI=donor[m]
for p,heads in (("uP161",[2,3]),("uP165",[0,1,3]),("uP164",[0])):
    M=F.load_model(p,"full_c1",seed=20,suffix="_eval_conf13"); r={}; F.forward(M,jnp.asarray(X[dI]),rec=r); A=np.asarray(r["pattern2"]); n=len(dI); k=np.arange(n)
    for h in heads:
        row=A[k,h,36+q[dI]]; pg=np.stack([row[k,q[dI]*6+2*g]+row[k,q[dI]*6+2*g+1] for g in range(3)],1); pg/=pg.sum(1,keepdims=True)
        # mass on sole eligible group
        sole=e[dI].argmax(1)
        print(p,"h",h,"E!=A1 donors n=%d corr(mass,rB)=%.3f corr(mass,rA1)=%.3f  mean mass on sole-eligible group=%.3f (E:0, A1:%.3f)"%(n,np.corrcoef(pg.ravel(),rB[dI].ravel())[0,1],np.corrcoef(pg.ravel(),rA1[dI].ravel())[0,1],pg[k,sole].mean(),rA1[dI][k,sole].mean()))
