import sys, os, numpy as np
B="/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_computation_atlas"; LR=B+"/../step1_latent_reliability"
sys.path.insert(0,B+"/src"); sys.path.insert(0,LR+"/src")
import jax, jax.numpy as jnp
import gen_family as G, nets as N, ckpt as C
f4,f1=G.Fam(pc=1.0,rho=1.0),G.Fam(pc=0.0); NB=150; K=3; Q=6
X0=G.sample(f4,NB,np.random.default_rng(6_100_000))["X"]
Xe=[];bi=[];qq=[];gg=[]
for q in range(Q):
  for k in range(Q):
    if k==q: continue
    for g,(a,b) in enumerate(G.GROUPS):
      X=X0.copy(); ag=X[:,k,a]==X[:,k,b]; X[:,k,b]=np.where(ag,(X[:,k,a]+1)%K,X[:,k,a]); Xe.append(X); bi+= list(range(NB)); qq+=[q]*NB; gg+=[g]*NB
Xe=np.concatenate(Xe); bi=np.array(bi); qq=np.array(qq); gg=np.array(gg); n=len(Xe); ar=np.arange(n)
A0=np.stack([(X0[:,:,a]==X0[:,:,b]) for a,b in G.GROUPS],-1).astype(int); Ae=np.stack([(Xe[:,:,a]==Xe[:,:,b]) for a,b in G.GROUPS],-1).astype(int)
cb=A0[bi].sum(1)[ar,gg]-A0[bi,qq,gg]; ca=Ae.sum(1)[ar,gg]-Ae[ar,qq,gg]; bd=((cb==4)&(ca==5))|((cb==5)&(ca==4))
with np.errstate(all="ignore"):
  l40=np.log(G.posterior(f4,X0)["post_y"]); l4e=np.log(G.posterior(f4,Xe)["post_y"]); l10=np.log(G.posterior(f1,X0)["post_y"]); l1e=np.log(G.posterior(f1,Xe)["post_y"])
o=np.argsort(-np.nan_to_num(l40,nan=-1e9),-1); c1=o[bi,qq,0]; c2=o[bi,qq,1]
mg=lambda lp,r: lp[r,qq,c1]-lp[r,qq,c2]
with np.errstate(all="ignore"):
  dm4=mg(l4e,ar)-mg(l40,bi); dm1=mg(l1e,ar)-mg(l10,bi)
df=np.isfinite(dm4)
models={"tf_s12":("tf",LR+"/checkpoints/replication/T4_M_K3_s12_fromT1.npz"),"tf_s0":("tf",LR+"/checkpoints/posthoc_curriculum/T4_M_K3_s0_fromT1.npz"),"ds_s0":("ds",LR+"/checkpoints/main/T4_M_K3_s0_deepsets.npz")}
for nm,(kind,path) in models.items():
  if kind=="tf": p=jax.tree_util.tree_map(jnp.asarray,C.load(path)); fw=lambda Xb: N.forward(p,Xb)
  else: p=jax.tree_util.tree_map(jnp.asarray,C.load_base(path)); fb=N.make_forward("deepsets",3); fw=lambda Xb: fb(p,Xb)
  jf=jax.jit(lambda Xb: jax.nn.log_softmax(fw(Xb)["answer"],-1))
  L=lambda X: np.concatenate([np.asarray(jf(jnp.asarray(X[i:i+3000]))) for i in range(0,len(X),3000)])
  m0,me=L(X0),L(Xe); dmm=mg(me,ar)-mg(m0,bi)
  for th in [0.0,0.1,0.5]:
    s=df&(np.abs(dm4)>th); sb=s&bd
    print(nm,f"|dm|>{th}: n {s.sum()} sign agree {(np.sign(dmm[s])==np.sign(dm4[s])).mean():.3f} | boundary n {sb.sum()} agree {(np.sign(dmm[sb])==np.sign(dm4[sb])).mean():.3f} T1plug@bd {(np.sign(dm1[sb])==np.sign(dm4[sb])).mean():.3f}")
  # per-base-episode clustering: fraction of base episodes with any boundary sign error
  sb=df&(np.abs(dm4)>0.5)&bd; err=(np.sign(dmm)!=np.sign(dm4))&sb
  print(nm,"boundary edits come from",len(np.unique(bi[sb])),"base episodes; episodes with >=1 error",len(np.unique(bi[err])))
# composition of boundary edits
sb=df&(np.abs(dm4)>0.5)&bd
print("boundary scored: 4->5",((cb==4)&(ca==5)&sb).sum()," 5->4",((cb==5)&(ca==4)&sb).sum(), "| all defined boundary", (df&bd).sum(), "off-support share of all edits", (~df).mean().round(3))
