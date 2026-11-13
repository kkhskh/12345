"""Investigation IX native-state utilities (JAX).
Parents: the P-count / T1 Transformers (3 layers, d=64, 43 tokens: 36 source tokens (index q*6+s), 6 readout tokens (36+q), 1 episode token).
'Native state' = residual stream of the frozen pretrained layers (resid_pre2 = after layers 0-1, the input the original T4 downstream block
sees) -- NO engineered counts, reports, eligibility or raw-input side channels enter native readers.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); ROOT = os.path.join(PKG, "..")
V7 = os.path.join(ROOT, "step1_r1_causal_origin_and_selective_rescue", "src"); V8 = os.path.join(ROOT, "step1_learned_relational_computation")
sys.path.insert(0, V7)
import numpy as np, jax, jax.numpy as jnp, optax
import nnlib as NL, evalrun as ER, r1lib as L
N = NL.N; Bm = NL.Bm; ENV = NL.ENV
EVAL_DEV = os.path.join(V7, "..", "logs", "eval_dev.npz")


def parent_path(name):
    """xP80/xP81/xT70 (Inv VI dev), zP100..zP107 / zT1110.. (Inv VIII Stage-0; dev for IX), yP130.. (Inv IX Stage-0 confirmation)."""
    if name.startswith("x"): return NL.dev_ckpt(name, 0)
    if name.startswith("z"): return os.path.join(V8, "checkpoints", name, "pre.npz")
    if name.startswith("y"): return os.path.join(PKG, "checkpoints", "z" + name[1:], "pre.npz")
    raise ValueError(name)


def load_parent(name): return NL.load(parent_path(name))


def states(pf, X, upto=2):
    """Residual stream before layer `upto` (upto=1: after layer 0; upto=2: after layers 0-1 = resid_pre2)."""
    x = N.embed(pf, X)
    for l in range(upto): x = NL.block(pf["layers"][l], x, l=l)
    return x


_states_j = {}
def states_np(pf, X, upto=2, bs=1000, dtype=np.float32):
    key = (id(pf), upto)
    if key not in _states_j: _states_j[key] = jax.jit(lambda X_: states(pf, X_, upto))
    f = _states_j[key]; return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + bs]))).astype(dtype) for i in range(0, len(X), bs)])


def decision_tokens(Z, q):
    """Z (n,43,d), q (n,) -> source tokens at question q (n,6,d), readout token q (n,d), episode token (n,d)."""
    idx = np.arange(len(Z)); src = Z[idx[:, None], q[:, None] * 6 + np.arange(6)[None]]; return src, Z[idx, 36 + q], Z[:, 42]


# ------------------------------------------------------------------ native-state readers (per decision: q-question tokens only, or all tokens)
def dense(k, a, b): return dict(W=jax.random.normal(k, (a, b)) * a ** -0.5, b=jnp.zeros(b))
def mlp_init(k, dims): ks = jax.random.split(k, len(dims)); return [dense(kk, a, b) for kk, a, b in zip(ks, dims[:-1], dims[1:])]
def mlp(ps, x):
    for i, l in enumerate(ps):
        x = x @ l["W"] + l["b"]
        if i < len(ps) - 1: x = jax.nn.gelu(x)
    return x
def ln(x): mu = x.mean(-1, keepdims=True); return (x - mu) / jnp.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5)

KINDS = ("lin_ro", "mlp_ro", "mlp_q", "ds_src", "pair_src", "attn_q", "bilin_src", "ds_all", "attn_all")


def reader_init(kind, seed, d=64, H=128):
    k = jax.random.split(jax.random.PRNGKey(seed), 8)
    if kind == "lin_ro": return dict(l=dense(k[0], d, 3))
    if kind == "mlp_ro": return dict(m=mlp_init(k[0], [d, H, H, 3]))
    if kind == "mlp_q": return dict(m=mlp_init(k[0], [7 * d, H, H, 3]))                       # readout + 6 source tokens, concatenated
    if kind == "ds_src": return dict(phi=mlp_init(k[0], [2 * d, H, H]), rho=mlp_init(k[1], [H + d, H, 3]))   # deep sets over sources
    if kind == "bilin_src":                                                                   # Theorem-1-shaped: rho bilinear in two pooled sums
        return dict(phi=mlp_init(k[0], [2 * d, H, 2 * 16]), out=dense(k[1], 16 * 16 + 2 * 16 + d, 3))
    if kind == "pair_src": return dict(phi=mlp_init(k[0], [3 * d, H, H]), psi=mlp_init(k[1], [2 * d, H, H]), rho=mlp_init(k[2], [2 * H + d, H, 3]))
    if kind == "ds_all": return dict(phi=mlp_init(k[0], [2 * d, H, H]), rho=mlp_init(k[1], [H + d, H, 3]))  # deep sets over all 36 source tokens
    if kind in ("attn_q", "attn_all"):                                                        # two attention blocks over the q-tokens / all tokens
        return dict(emb=dense(k[0], d, 64), blocks=[dict(Wq=dense(k[1], 64, 64)["W"], Wk=dense(k[2], 64, 64)["W"], Wv=dense(k[3], 64, 64)["W"],
                    Wo=dense(k[4], 64, 64)["W"], ff=mlp_init(k[5], [64, H, 64])) for _ in range(2)], out=dense(k[6], 64, 3))
    raise ValueError(kind)


def reader_apply(kind, p, src, ro, allsrc=None):
    """src (n,6,d) source tokens at q ; ro (n,d) readout token q ; allsrc (n,36,d) all source tokens -> logits (n,3)."""
    if kind == "ds_all":
        rob = jnp.broadcast_to(ro[:, None], allsrc.shape); h = mlp(p["phi"], jnp.concatenate([allsrc, rob], -1)).sum(1)
        return mlp(p["rho"], jnp.concatenate([h, ro], -1))
    if kind == "attn_all": src = allsrc
    if kind == "lin_ro": return ro @ p["l"]["W"] + p["l"]["b"]
    if kind == "mlp_ro": return mlp(p["m"], ro)
    if kind == "mlp_q": return mlp(p["m"], jnp.concatenate([ro, src.reshape(len(src), -1)], -1))
    rob = jnp.broadcast_to(ro[:, None], src.shape)
    if kind == "ds_src":
        h = mlp(p["phi"], jnp.concatenate([src, rob], -1)).sum(1); return mlp(p["rho"], jnp.concatenate([h, ro], -1))
    if kind == "bilin_src":
        h = mlp(p["phi"], jnp.concatenate([src, rob], -1)).sum(1); a, b = h[:, :16], h[:, 16:]
        f = jnp.concatenate([(a[:, :, None] * b[:, None, :]).reshape(len(h), -1), a, b, ro], -1); return f @ p["out"]["W"] + p["out"]["b"]
    if kind == "pair_src":
        i, j = np.triu_indices(6, 1)
        P = mlp(p["phi"], jnp.concatenate([src[:, i], src[:, j], rob[:, i]], -1)).sum(1)
        S = mlp(p["psi"], jnp.concatenate([src, rob], -1)).sum(1); return mlp(p["rho"], jnp.concatenate([P, S, ro], -1))
    if kind in ("attn_q", "attn_all"):
        x = jnp.concatenate([ro[:, None], src], 1) @ p["emb"]["W"] + p["emb"]["b"]
        for b in p["blocks"]:
            z = ln(x); a = jax.nn.softmax(jnp.einsum("ntd,nsd->nts", z @ b["Wq"], z @ b["Wk"]) / 8.0, -1)
            x = x + jnp.einsum("nts,nsd->ntd", a, z @ b["Wv"]) @ b["Wo"]; x = x + mlp(b["ff"], ln(x))
        return x[:, 0] @ p["out"]["W"] + p["out"]["b"]
    raise ValueError(kind)


def count_params(p): return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


# ------------------------------------------------------------------ evaluation helpers
def load_eval(path=EVAL_DEV): return dict(np.load(path, allow_pickle=True))


def summarize(r):
    """Primary metrics (mandate 11.2): per-episode absolute error (absE) is primary; signed mean E separately."""
    F = r["families"]; g = lambda f, rg, k: F[f].get(rg, {}).get(k, float("nan"))
    def pooled(f, regs, k="absE"):
        tot = sum(F[f][x]["n"] for x in regs if x in F[f]); return sum(F[f][x][k] * F[f][x]["n"] for x in regs if x in F[f]) / max(tot, 1)
    return dict(KL=r["quality"]["KL"], acc=r["quality"]["acc"],
                absE_R1=g("F45", "R1", "absE"), E_R1=g("F45", "R1", "E"), dm_R1=g("F45", "R1", "dmM"),
                absE_R2=g("F45", "R2", "absE"), E_R2=g("F45", "R2", "E"), absE_R3=g("F45", "R3", "absE"), E_R3=g("F45", "R3", "E"),
                absE_R23=pooled("F45", ("R2", "R3")), absE_F34=g("F34", "R1", "absE"), absE_F54o=g("F54o", "all", "absE"),
                absE_F0=g("F0", "all", "absE"), absE_FLOC=g("FLOC", "all", "absE"), absE_FID=g("FID", "all", "absE"))
