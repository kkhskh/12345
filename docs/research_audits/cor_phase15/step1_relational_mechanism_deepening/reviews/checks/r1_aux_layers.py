import sys, numpy as np, json
R="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening"; sys.path.insert(0,R+"/src")
import jax, jax.numpy as jnp, tasks3 as T3, fwd3 as F3, ckpt as C
from sklearn.linear_model import LogisticRegression
d=T3.sample_trans_pairs(6000,np.random.default_rng(9_950_000),T3.ALL_PAIRS); X=d["X"]; tr=np.arange(6000)<4000
cm=T3.composite_index(d["sigma"],d["tau"])
for ck in ("trans_aux_s2","trans_full_s0"):
    p=C.load(R+f"/checkpoints/{ck}.npz"); p.pop("WX",None); p=jax.tree_util.tree_map(jnp.asarray,p)
    @jax.jit
    def run(Xb):
        rec={}
        def hk(n,a):
            if n.startswith("resid_post"): rec[n]=jnp.concatenate([a[:,36:42].mean(1),a[:,42]],-1)
            return a
        F3.forward(p,Xb,hk); return rec
    o=[run(jnp.asarray(X[i:i+2000])) for i in range(0,6000,2000)]
    res={}
    for l in range(3):
        Z=np.concatenate([np.asarray(t[f"resid_post{l}"]) for t in o]); D=Z.shape[1]//2
        for zn,ZZ in (("readout",Z[:,:D]),("ep",Z[:,D:])):
            mu,sd=ZZ[tr].mean(0),ZZ[tr].std(0)+1e-6; Zs=(ZZ-mu)/sd
            res[f"L{l}_{zn}"]={tn:round(float(LogisticRegression(max_iter=2000).fit(Zs[tr],t[tr]).score(Zs[~tr],t[~tr])),3) for tn,t in (("sigma",d["sigma"]),("tau",d["tau"]))}
    print(ck,json.dumps(res),flush=True)
