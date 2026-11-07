"""R5 baseline reviewer checks. Uses package src/ (learned.py) unchanged; same data/optimizer/steps."""
import sys, json
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, optax
SRC = Path(__file__).resolve().parents[2] / "src"; sys.path.insert(0, str(SRC))
import learned as Lm

def mlp_p(k, nin, K, w=32):
    return dict(W1=0.3*jax.random.normal(next(k),(nin,w)), c1=jnp.zeros(w),
                W2=0.3*jax.random.normal(next(k),(w,w)), c2=jnp.zeros(w),
                W3=0.3*jax.random.normal(next(k),(w,K)), c3=jnp.zeros(K))
def mlp(p, x):
    h = jax.nn.gelu(x@p["W1"]+p["c1"], approximate=True); h = jax.nn.gelu(h@p["W2"]+p["c2"], approximate=True)
    return h@p["W3"]+p["c3"]

def run(task, kind, seed, steps=Lm.STEPS, lr=Lm.LR):
    K = Lm.n_classes(task); T5 = task.startswith("T5")
    k = iter(jax.random.split(jax.random.PRNGKey(seed+999), 32))
    def src(inp):
        if T5:
            P = jax.nn.one_hot(inp["p"],3) if task=="T5b" else jnp.zeros((inp["zA"].shape[0],3))
            return inp["zA"], inp["zB"], P
        return inp["a"][:,None], inp["b"][:,None], inp["p"][:,None]
    vin = 3 if T5 else 1; pin = 3 if T5 else 1
    if kind == "B-add-strict":      # f(a)+g(b)+h(p): p NOT given to f,g
        P = dict(f=mlp_p(k,vin,K), g=mlp_p(k,vin,K), h=dict(W=0.1*jax.random.normal(next(k),(pin,K))))
        f = lambda P,inp: (lambda A,B,Pp: mlp(P["f"],A)+mlp(P["g"],B)+Pp@P["h"]["W"])(*src(inp))
    elif kind == "B-add":           # package version: f(a,p)+g(b,p)
        P = dict(f=mlp_p(k,vin+pin,K), g=mlp_p(k,vin+pin,K))
        f = lambda P,inp: (lambda A,B,Pp: mlp(P["f"],jnp.concatenate([A,Pp],1))+mlp(P["g"],jnp.concatenate([B,Pp],1)))(*src(inp))
    elif kind == "B-add+bil":       # B-add plus a bilinear zA^T M_k zB term (relation available)
        P = dict(f=mlp_p(k,vin+pin,K), g=mlp_p(k,vin+pin,K), M=0.1*jax.random.normal(next(k),(K,vin,vin)))
        def f(P,inp):
            A,B,Pp = src(inp)
            return mlp(P["f"],jnp.concatenate([A,Pp],1))+mlp(P["g"],jnp.concatenate([B,Pp],1))+jnp.einsum("ni,kij,nj->nk",A,P["M"],B)
    elif kind == "B-bil-pgated":    # bilinear term whose matrix depends on p (p x zA x zB), plus linear
        P = dict(M=0.1*jax.random.normal(next(k),(K,pin+1,vin,vin)), W=0.1*jax.random.normal(next(k),(2*vin+pin,K)), c=jnp.zeros(K))
        def f(P,inp):
            A,B,Pp = src(inp); P1 = jnp.concatenate([Pp, jnp.ones((A.shape[0],1))],1)
            return jnp.einsum("nq,ni,kqij,nj->nk",P1,A,P["M"],B)+jnp.concatenate([A,B,Pp],1)@P["W"]+P["c"]
    elif kind == "B-lin-sign":      # linear on per-source hard support features (sign / one-hot argmax)
        P = dict(W=0.1*jax.random.normal(next(k),(2*vin+pin,K)), c=jnp.zeros(K))
        def f(P,inp):
            A,B,Pp = src(inp)
            if T5: A,B = jax.nn.one_hot(A.argmax(1),3), jax.nn.one_hot(B.argmax(1),3)
            else: A,B = jnp.sign(A), jnp.sign(B)
            return jnp.concatenate([A,B,Pp],1)@P["W"]+P["c"]
    elif kind == "B-lin-oracle-agree":   # linear on raw zA,zB,P plus ORACLE agreement indicator and its products with zA,zB,P
        P = dict(W=0.1*jax.random.normal(next(k),(2*(2*vin+pin)+1,K)), c=jnp.zeros(K))
        def f(P,inp):
            A,B,Pp = src(inp)
            ag = ((A.argmax(1)==B.argmax(1)) if T5 else (A[:,0]*B[:,0]>0)).astype(jnp.float32)[:,None]
            X = jnp.concatenate([A,B,Pp],1)
            return jnp.concatenate([X, ag*X, ag],1)@P["W"]+P["c"]
    elif kind == "B-MLP-wide":      # B-MLP with ~2.4k params (same budget as B-add)
        P = dict(m=mlp_p(k,2*vin+pin,K,w=44))
        f = lambda P,inp: mlp(P["m"], jnp.concatenate(src(inp),1))
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps)); st = opt.init(P)
    @jax.jit
    def step(P, st, inp, tgt):
        l, g = jax.value_and_grad(lambda P: -(tgt*jax.nn.log_softmax(f(P,inp))).sum(-1).mean())(P)
        up, st = opt.update(g, st, P); return optax.apply_updates(P, up), st, l
    rng = np.random.default_rng(40_000*(seed+1)+Lm.TASKS.index(task))
    for _ in range(steps):
        bt = Lm.sample(task, Lm.BATCH, rng); P, st, l = step(P, st, Lm.to_inp(task, bt), Lm.soft_target(task, bt))
    val = Lm.sample(task, 20000, np.random.default_rng(1000)); lg = np.asarray(f(P, Lm.to_inp(task, val)))
    acc = float((lg.argmax(1)==val.y).mean()) if T5 else float((np.where(lg[:,0]>lg[:,1],1.,-1.)==val.y).mean())
    npar = sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(P))
    return acc, npar

if __name__ == "__main__":
    jobs = [(t,k) for t,k in (a.split(":") for a in sys.argv[2:])]
    steps = int(sys.argv[1])
    out = {}
    for t,kd in jobs:
        accs = []; 
        for s in (0,1,2):
            a, n = run(t, kd, s, steps=steps); accs.append(a)
        out[f"{t}/{kd}/steps{steps}"] = dict(accs=accs, mean=float(np.mean(accs)), params=n)
        print(t, kd, steps, [round(x,4) for x in accs], "mean", round(float(np.mean(accs)),4), "params", n, flush=True)
    print(json.dumps(out))
