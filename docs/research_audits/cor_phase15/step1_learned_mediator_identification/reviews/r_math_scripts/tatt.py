import sys, numpy as np
R="/home/claude/12345/docs/research_audits/cor_phase15/step1_learned_mediator_identification"
sys.path.insert(0,R+"/src")
import ckpt, model as M, jax.numpy as jnp
for name in ["T4_TF1_s2","T1_TF1_s4","T3_TF1_s2"]:
    P=ckpt.load(f"{R}/checkpoints/historical/{name}.npz"); task=name.split("_")[0]
    g=np.linspace(-3,3,41); A,B=np.meshgrid(g,g,indexing="ij")
    for pv in ([1.0,-1.0] if task=="T3" else [0.0]):
        inp=dict(a=jnp.asarray(A.ravel(),jnp.float32),b=jnp.asarray(B.ravel(),jnp.float32),p=jnp.full(A.size,pv,jnp.float32))
        st={}
        def hk(n,a):
            if n=="resid_mid0": st["h"]=np.asarray(a)[:,-1]
            return a
        M.forward(P,task,inp,hk)
        X=np.asarray(M.embed(P,task,inp)).astype(np.float64); L=P["layers"][0]
        def ln(x): mu=x.mean(-1,keepdims=True); v=((x-mu)**2).mean(-1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*L["g1"]+L["b1"]
        Z=ln(X); q=np.einsum("nd,hde->nhe",Z[:,-1],L["WQ"]); k=np.einsum("ntd,hde->nhte",Z,L["WK"]); v=np.einsum("ntd,hde->nhte",Z,L["WV"])
        om=np.exp(np.einsum("nhe,nhte->nht",q,k)/np.sqrt(8))
        num=np.einsum("nht,nhte,hed->nhd",om,v,L["WO"]); den=om.sum(-1)
        h=X[:,-1]+(num/den[...,None]).sum(1)+L["bO"]
        hnob=h-L["bO"]
        print(name,pv,"formula err",np.abs(h-st["h"]).max(),"|bO|",np.abs(L["bO"]).max(),"err w/o bO",np.abs(hnob-st["h"]).max())
        # freeze normalizer only
        den0=den[A.size//2+20]  # some ref
        hf=X[:,-1]+(num/den0[None,:,None]).sum(1)+L["bO"]
        Hf=hf.reshape(41,41,-1); Hs=Hf.mean(1,keepdims=True)+Hf.mean(0,keepdims=True)-Hf.mean((0,1),keepdims=True)
        H=h.reshape(41,41,-1); Hs2=H.mean(1,keepdims=True)+H.mean(0,keepdims=True)-H.mean((0,1),keepdims=True)
        print("  interaction share full",((H-Hs2)**2).sum()/((H-H.mean((0,1)))**2).sum()," frozen-normalizer",((Hf-Hs)**2).sum()/((Hf-Hf.mean((0,1)))**2).sum())
