import sys; PK="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation"; sys.path.insert(0,PK+"/src")
import numpy as np, native as NV, genlib as GL
np.seterr(all="ignore")
st=int(sys.argv[1]); env=GL.Env(3,1); rng=np.random.default_rng(st); N_=3000
d=NV.Bm.sample_fast(NV.ENV,N_,rng); X=d["X"]; q=rng.integers(0,6,N_); ii=np.arange(N_)
rel,col,noi=GL._group_lik(env,X); relm=rel.mean(-1); lr=np.log(relm/noi); W=np.exp(lr.sum(1)-lr[ii,q]); Lg=rel[ii,q]/noi
a_=X.reshape(N_,6,3,2)[ii,q]; e=((GL.counts(X,3)[ii,q]==5)&(a_[...,0]==a_[...,1])).astype(int); S=e.sum(1)
lw=np.log(W); lsl=np.log(Lg.sum(-1)); m=S[:,None]-e
for nm,v in (("logW",lw),("logSumL",lsl),("log m",np.log(np.maximum(m,1e-300))),("log r (full)",lw+lsl+np.log(np.maximum(m,1e-300)))):
    el=e==1; d2=((v*el).sum(1)/2-(v*~el).sum(1))[S==2]; print(nm,"S=2 contrast mean %.3f"%np.mean(d2[np.isfinite(d2)]), "n",(S==2).sum())
