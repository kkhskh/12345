import os, sys
R="/home/claude/12345/docs/research_audits/cor_phase15"; P=R+"/step1_relational_operator_reuse"
sys.path.insert(0,P+"/src"); sys.path.insert(0,R+"/step1_cross_family_relational_foundations/src")
import numpy as np, jax.numpy as jnp, gatelib as GL, xviislot as XS, nets as N, xvitask as T
p = GL.load(f"{P}/confirm_outputs/A2_s705/step1400.npz")
X = T.sample(300, np.random.default_rng(5), 2, .9, .9, .05)["X"]
rng=np.random.default_rng(1); mx=0
for _ in range(5):
    perm=rng.permutation(6); mx=max(mx,XS.theorem_P_check(p,X,perm))
# full-window relabel == codes swapped == content swap
a=np.asarray(XS.forward(p,jnp.asarray(X),(4,0,0,3)))
p2=dict(p); E=np.asarray(p["Eq"]).copy(); E[[4,0]]=E[[0,4]]; p2["Eq"]=jnp.asarray(E)
b=np.asarray(N.forward(p2,jnp.asarray(X))["answer"])
Xs=X.copy(); Xs[:,[4,0]]=X[:,[0,4]]; c=np.array(N.forward(p,jnp.asarray(Xs))["answer"]); c[:,[4,0]]=c[:,[0,4]]
# token order shuffle (no PE): permute token positions arbitrarily
x=N.embed(p,jnp.asarray(X)); print("thP",mx,"relabel_vs_codeswap",np.abs(a-b).max(),"codeswap_vs_contentswap",np.abs(b-c).max())
