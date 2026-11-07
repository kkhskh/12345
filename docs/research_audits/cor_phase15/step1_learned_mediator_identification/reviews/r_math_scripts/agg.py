import sys, itertools, numpy as np, jax.numpy as jnp
R="/home/claude/12345/docs/research_audits/cor_phase15/step1_learned_mediator_identification"
sys.path.insert(0,R+"/src")
import ckpt, model as M
g=np.linspace(-3,3,31); A,B=np.meshgrid(g,g,indexing="ij")
for arch in ["TF1","TF2","TF2AO"]:
  for s in range(5):
    P=ckpt.load(f"{R}/checkpoints/new/T6_{arch}_s{s}.npz")
    m=np.asarray(M.margin(M.forward(P,"T6",dict(a=jnp.asarray(A.ravel(),jnp.float32),b=jnp.asarray(B.ravel(),jnp.float32),p=jnp.zeros(A.size,jnp.float32))))).reshape(31,31)
    out=[]
    for tol in (0, 0.05*m.std()):
        n=0
        for i,j in itertools.combinations(range(31),2):
            d=m[i]-m[j]; n+=(d>tol).any() and (d<-tol).any()
        out.append(n)
    print(arch,s,out)
