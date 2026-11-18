import os, sys, json
R="/home/claude/12345/docs/research_audits/cor_phase15"; P=R+"/step1_relational_operator_reuse"
sys.path.insert(0,P+"/src"); sys.path.insert(0,R+"/step1_cross_family_relational_foundations/src")
import numpy as np, jax, jax.numpy as jnp, gatelib as GL, xviislot as XS, nets as N
run, cond, step, pairs = sys.argv[1], sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4])
p = GL.load(f"{P}/confirm_outputs/{run}/step{step}.npz"); S=6; Q=6
def fwd(p, X, at=None, delta=None):
    x = N.embed(p, X); L=len(p["layers"]); res=[]
    for l, lay in enumerate(p["layers"]):
        if at==l: x = x + delta
        res.append(x)
        z = N.ln(x, lay["g1"], lay["b1"])
        q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        a = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", q, k)/jnp.sqrt(q.shape[-1]), -1)
        x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
        x = x + jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True) @ lay["W2"] + lay["c2"]
    zf = N.ln(x, p["gf"], p["bf"]); return zf[:, S*Q:S*Q+Q] @ p["WA"], res
D = XS.SlotDiag(cond, n=600)
# per-token-role slot means of the residual entering layer 2 (clean), and of layer-1 input
_, res = fwd(p, jnp.asarray(D.X)); 
def Fg(delta_fn, at):
    out=[]
    X=[D.X]+[D.Xf[q] for q in range(Q)]
    P0=np.asarray(jax.nn.softmax(fwd(p,jnp.asarray(D.X),at,delta_fn)[0],-1))
    for q in range(Q):
        Pf=np.asarray(jax.nn.softmax(fwd(p,jnp.asarray(D.Xf[q]),at,delta_fn)[0],-1))[:,q]; a0,af=P0[:,q].argmax(-1),Pf.argmax(-1)
        out.append(float(((af!=a0)&(af==D.newG[q]))[D.dec[q]].mean()))
    return out
T=res[0].shape[1]; d=res[0].shape[2]
out={}
for at in (1,2):
    M = np.asarray(res[at]).mean(0)  # T,d mean residual per token position (slot,role)
    for (i,j) in pairs:
        ti,tj = XS.tok_index(i), XS.tok_index(j)
        raw = np.zeros((T,d)); dq = np.asarray(p["Eq"][j]-p["Eq"][i]); raw[ti]+=dq; raw[tj]-=dq
        mean = np.zeros((T,d)); mean[ti]+= M[tj]-M[ti]; mean[tj]+= M[ti]-M[tj]
        r1=Fg(jnp.asarray(raw),at); r2=Fg(jnp.asarray(mean),at)
        out[f"L{at} {i}->{j}"]=dict(raw_donor=r1[i],raw_recip=r1[j],mean_donor=r2[i],mean_recip=r2[j])
        print(at,i,j,{k:round(v,2) for k,v in out[f"L{at} {i}->{j}"].items()},flush=True)
# slot decodability from layer-2 residual with raw Eq projected out
Eq=np.asarray(p["Eq"]); Qb,_=np.linalg.qr(Eq.T); 
R2=np.asarray(res[2])[:, :S*Q].reshape(-1,Q,S,d)  # source tokens
Xr = R2 - (R2@Qb)@Qb.T
lab=np.broadcast_to(np.arange(Q)[None,:,None],R2.shape[:3]).ravel(); F=Xr.reshape(-1,d)
n=len(F)//2; idx=np.random.default_rng(0).permutation(len(F)); tr,te=idx[:n],idx[n:]
from numpy.linalg import lstsq
Y=np.eye(Q)[lab]; W=lstsq(np.c_[F[tr],np.ones(n)],Y[tr],rcond=None)[0]; acc=(np.c_[F[te],np.ones(len(te))]@W).argmax(1)==lab[te]
out["slot_decode_acc_layer2_input_Eq_projected_out"]=float(acc.mean()); print(out["slot_decode_acc_layer2_input_Eq_projected_out"])
json.dump(out,open(f"/tmp/revA17/late_{run}_{sys.argv[5] if len(sys.argv)>5 else 0}.json","w"),indent=1)
