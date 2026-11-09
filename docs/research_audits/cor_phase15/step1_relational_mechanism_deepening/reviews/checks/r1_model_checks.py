import sys, os, numpy as np, json
R="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening"; sys.path.insert(0,R+"/src")
import jax, jax.numpy as jnp
import tasks3 as T3, tasks_ext as TX, fwd3 as F3, ckpt as C
from sklearn.linear_model import LogisticRegression
K,Q=3,6; MASK=TX.MASK; AP=T3.ALL_PAIRS; TRP=T3.TRAIN_PAIRS; HOP=T3.HOLDOUT_PAIRS
def mk(ck):
    p=C.load(R+"/"+ck); WX=p.pop("WX",None); p=jax.tree_util.tree_map(jnp.asarray,p)
    @jax.jit
    def run(X):
        rec={}
        def hk(n,a):
            if n=="final": rec["f"]=a
            return a
        o=F3.forward(p,X,hk); f=rec["f"]; return jax.nn.softmax(o["answer"],-1), f[:,36:42].mean(1), f[:,42]
    def apply(X):
        o=[run(jnp.asarray(X[i:i+2000])) for i in range(0,len(X),2000)]
        return [np.concatenate([np.asarray(t[k]) for t in o]) for k in range(3)]
    return apply, WX
def rbayes(X,pairs):
    idx=[AP.index(pq) for pq in pairs]; J=TX.trans_joint(X)[:,:,idx]; lM=np.log(J.sum(-1))
    loo=np.stack([lM[:,[k for k in range(Q) if k!=q]].sum(1) for q in range(Q)],1)
    w=np.exp(loo-loo.max(-1,keepdims=True)); w/=w.sum(-1,keepdims=True); py=(w[...,None]*J).sum(2); return py/py.sum(-1,keepdims=True)
def regen_factory(d,X):
    rng=np.random.default_rng(9_950_500); si,ti=d["sigma"].copy(),d["tau"].copy(); Y,typ=d["Y"],d["typ"]
    def regen(Xb,which):
        Xn=Xb.copy(); n=len(Xn); new=(si+rng.integers(1,6,n))%6 if which=="sigma" else (ti+rng.integers(1,6,n))%6
        Sm=np.stack([T3.PERMS[i] for i in (new if which=="sigma" else si)]); Tm=np.stack([T3.PERMS[i] for i in (ti if which=="sigma" else new)])
        if which=="sigma":
            vB=np.take_along_axis(Sm,Y,1); ab=typ==0
            for s in (2,3):
                kn=Xb[:,:,s]!=np.take_along_axis(np.stack([T3.PERMS[i] for i in si]),Y,1)
                Xn[:,:,s]=np.where(ab,np.where(kn,(vB+1+rng.integers(0,K-1,vB.shape))%K,vB),Xn[:,:,s])
        else:
            vB=np.take_along_axis(np.stack([T3.PERMS[i] for i in si]),Y,1); vC=np.take_along_axis(Tm,vB,1); bc=typ==1
            vCo=np.take_along_axis(np.stack([T3.PERMS[i] for i in ti]),vB,1)
            for s in (4,5):
                kn=Xb[:,:,s]!=vCo
                Xn[:,:,s]=np.where(bc,np.where(kn,(vC+1+rng.integers(0,K-1,vC.shape))%K,vC),Xn[:,:,s])
        return Xn,new
    return regen
def uniq(Pm): s=np.sort(Pm,-1); return s[...,-1]-s[...,-2]>1e-6
out={}
# ---------- aux run: where is the map information?
apply,WX=mk("checkpoints/trans_aux_s2.npz")
d=T3.sample_trans_pairs(6000,np.random.default_rng(9_950_000),AP); X=d["X"]; cm=T3.composite_index(d["sigma"],d["tau"]); tr=np.arange(6000)<4000
P,Zr,Ze=apply(X)
def probe(Z,t):
    mu,sd=Z[tr].mean(0),Z[tr].std(0)+1e-6; Zs=(Z-mu)/sd; return float(LogisticRegression(max_iter=3000).fit(Zs[tr],t[tr]).score(Zs[~tr],t[~tr]))
a={}
for zn,Z in (("readout_mean",Zr),("episode_tok",Ze)):
    a[zn]={tn:probe(Z,t) for tn,t in (("sigma",d["sigma"]),("tau",d["tau"]),("composite",cm))}
# per-readout-position (C-only rows) not pooled: probe readout tokens of single positions
h=Ze@WX; sh,th=h[:,:6].argmax(-1),h[:,6:].argmax(-1)
a["aux_head_acc"]=dict(sigma=float((sh==d["sigma"]).mean()),tau=float((th==d["tau"]).mean()),
    composite_by_composing_head_argmaxes=float((T3.composite_index(sh,th)==cm).mean()))
out["aux"]=a; print(json.dumps(a),flush=True)
# ---------- holdout model
apply,_=mk("checkpoints/trans_holdout_s1.npz"); hres={}
for nm,PS in (("train_eps",TRP),("holdout_eps",HOP)):
    d=T3.sample_trans_pairs(6000,np.random.default_rng(9_950_000),PS); X,Y,typ=d["X"],d["Y"],d["typ"]; c=typ==2
    P=apply(X)[0]; pf=TX.bayes_trans(X)["post_y"]; pr=rbayes(X,TRP); m=P.argmax(-1)
    r=dict(agree_restricted=float((m==pr.argmax(-1))[c].mean()),agree_full=float((m==pf.argmax(-1))[c].mean()),
           agree_restricted_where_they_differ=float((m==pr.argmax(-1))[c&(pr.argmax(-1)!=pf.argmax(-1))].mean()),
           agree_full_where_they_differ=float((m==pf.argmax(-1))[c&(pr.argmax(-1)!=pf.argmax(-1))].mean()),
           n_differ=int((c&(pr.argmax(-1)!=pf.argmax(-1))).sum()),
           KL_model_vs_restricted=float((pr*np.log((pr+1e-12)/(P+1e-12))).sum(-1)[c].mean()),
           KL_model_vs_full=float((pf*np.log((pf+1e-12)/(P+1e-12))).sum(-1)[c].mean()))
    regen=regen_factory(d,X)
    for which in ("sigma","tau"):
        Xn,new=regen(X,which); Pn=apply(Xn)[0]; pbn=TX.bayes_trans(Xn)["post_y"]; prn=rbayes(Xn,TRP); pb=pf
        sel=c&(pb.argmax(-1)!=pbn.argmax(-1))&uniq(pb)&uniq(pbn)
        newpair=[(int(a_),int(b_)) for a_,b_ in (zip(new,d["tau"]) if which=="sigma" else zip(d["sigma"],new))]
        lands_train=np.array([p_ in TRP for p_ in newpair])[:,None]&np.ones((1,Q),bool)
        fol=(Pn.argmax(-1)==pbn.argmax(-1)); rfol=(prn.argmax(-1)==pbn.argmax(-1))
        # validity: reconstructed labels for non-intervened questions under new world
        S=np.stack([T3.PERMS[i] for i in d["sigma"]]); Sn=np.stack([T3.PERMS[i] for i in new]) if which=="sigma" else S
        if which=="sigma":
            inv=np.argsort(Sn,1); Yn=np.where(typ==0,Y,np.take_along_axis(inv,np.take_along_axis(S,Y,1),1))
        else: Yn=Y
        ab=typ==0; vB=np.take_along_axis(Sn,Y,1)
        err=float(((Xn[:,:,2]!=vB)[ab]).mean()) if which=="sigma" else None
        r[which]=dict(n=int(sel.sum()),frac_new_pair_in_train=float(lands_train[sel].mean()),
            model_follows=float(fol[sel].mean()),model_follows_newpair_train=float(fol[sel&lands_train].mean()),
            model_follows_newpair_heldout=float(fol[sel&~lands_train].mean()) if (sel&~lands_train).any() else None,
            n_heldout_landing=int((sel&~lands_train).sum()),
            restrictedBayes_follows=float(rfol[sel].mean()),restrictedBayes_follows_train=float(rfol[sel&lands_train].mean()),
            restrictedBayes_follows_heldout=float(rfol[sel&~lands_train].mean()) if (sel&~lands_train).any() else None,
            newBayes_acc_Conly_on_reconstructed_labels=float((pbn.argmax(-1)==Yn)[c].mean()) if which=="sigma" else None,
            regenerated_B_error_rate_AB=err)
    hres[nm]=r; print(nm,json.dumps(r),flush=True)
out["holdout"]=hres
json.dump(out,open("/tmp/r1/model_checks.json","w"),indent=1)
