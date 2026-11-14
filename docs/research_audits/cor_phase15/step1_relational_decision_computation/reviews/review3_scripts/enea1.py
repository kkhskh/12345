import json, os, sys
PK="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation"
sys.path.insert(0, PK+"/src")
import numpy as np
import native as NV, genlib as GL
np.seterr(all="ignore")
env = GL.Env(3, 1); rng = np.random.default_rng(74_100_000); N_ = 2000
d = NV.Bm.sample_fast(NV.ENV, 3 * N_, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); ii = np.arange(len(X))
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(len(X), 6, 3, 2)[ii, q]; e = ((GL.counts(X, 3)[ii, q] == 5) & (a_[..., 0] == a_[..., 1])).astype(float); S = e.sum(1, keepdims=True)
rB = W * (S - e) * Lg.sum(-1); rB = rB / rB.sum(1, keepdims=True); Ln = Lg / Lg.sum(-1, keepdims=True)
nrm = lambda w: w / np.maximum(w.sum(1, keepdims=True), 1e-300)
pf = (GL.counts(X, 3)[ii, q] == 5).astype(float); rA1 = nrm(W * np.maximum(pf.sum(1, keepdims=True) - pf, 0) * Lg.sum(-1) + 1e-300)
rA3 = nrm(W * np.maximum(1 - e, 0) * Lg.sum(-1) + 1e-300)
PB = GL.posterior_closed(env, X, q)
base = np.arange(N_); donor = np.array([rng.choice(np.where(q == q[b])[0]) for b in base])
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
def pred_from(r):
    Pp = (r[..., None] * Ln[base]).sum(1); Pp /= Pp.sum(-1, keepdims=True); return cen(Pp) - cen(PB[base])
def fid(o,p):
    o,p=o.ravel(),p.ravel(); b=(o*p).sum()/max((p*p).sum(),1e-12); return b, 1-((o-b*p)**2).sum()/((o-o.mean())**2).sum(), np.corrcoef(o,p)[0,1]
D=json.load(open(PK+"/logs/l2_mix_causal_conf.json"))
pB=pred_from(rB[donor]); pA1=pred_from(rA1[donor]); pA3=pred_from(rA3[donor])
m=(pf[donor]!=e[donor]).any(1)
print("n EneA1",m.sum(), "S of donor in subset", np.bincount(e[donor][m].sum(1).astype(int)), "S_pf", np.bincount(pf[donor][m].sum(1).astype(int)))
print("donor S overall", np.bincount(e[donor].sum(1).astype(int)))
# also base==donor same episode?
print("donor==base", (donor==base).sum())
rngb=np.random.default_rng(0)
for u,v in D.items():
    if "per_episode_obs_donor" not in v: continue
    obs=np.array(v["per_episode_obs_donor"]); sp=np.array(v["per_episode_pred_donor"])
    print(u[:5],"repro max|pred diff|",float(np.abs(sp-pB).max()))
    o=obs[m]; 
    fb=fid(o,pB[m]); fa=fid(o,pA1[m])
    # per-episode scale-free: correlation per episode then mean ; and R2 when each episode's prediction is optimally scaled (per-episode slope)
    def perep(o,p):
        num=(o*p).sum(1); den=np.maximum((p*p).sum(1),1e-12); b=num/den; res=((o-b[:,None]*p)**2).sum(); tot=((o-o.mean())**2).sum(); return 1-res/tot
    # bootstrap of R2 difference over episodes
    idx=np.where(m)[0]; diffs=[]
    for _ in range(2000):
        s=rngb.choice(idx,len(idx)); diffs.append(fid(obs[s],pB[s])[1]-fid(obs[s],pA1[s])[1])
    # magnitude of predictions
    print("   EneA1 Bayes slope/R2/corr %.2f %.3f %.3f | A1 %.2f %.3f %.3f | diff CI [%.2f, %.2f]"%(*fb,*fa,*np.percentile(diffs,[2.5,97.5])))
    print("   per-episode-rescaled R2: Bayes %.3f A1 %.3f ; mean|pred| Bayes %.2f A1 %.2f obs %.2f"%(perep(o,pB[m]),perep(o,pA1[m]),np.abs(pB[m]).mean(),np.abs(pA1[m]).mean(),np.abs(o).mean()))
    # S==1 stratum vs others over all donors
    Sd=e[donor].sum(1)
    for s in (0,1,2,3):
        mm=Sd==s
        if mm.sum()>20: print("   donor S=%d n=%d  R2 B %.3f A3 %.3f A1 %.3f"%(s,mm.sum(),fid(obs[mm],pB[mm])[1],fid(obs[mm],pA3[mm])[1],fid(obs[mm],pA1[mm])[1]))
    # excluding E!=A1: Bayes vs A1
    print("   E==A1 subset R2 B %.3f A1 %.3f"%(fid(obs[~m],pB[~m])[1],fid(obs[~m],pA1[~m])[1]))
# correlation of rB and rA1 with each other in subset
print("corr rB vs rA1 in subset", np.corrcoef(rB[donor][m].ravel(), rA1[donor][m].ravel())[0,1])
