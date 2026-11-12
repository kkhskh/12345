"""Readers on per-decision relational inputs (Conditions O and R) for the learned-composition experiment.
Per decision (episode, question q) and group g the Condition-R input is x_g = [one-hot n_g (6), one-hot of g's two q-reports (2 x 3),
one-hot group identity (3)] (15 dims).  Condition O adds one-hot m_g (Investigation VII eligibility, 3).  Nothing else.
Readers output 3 candidate logits per decision:
  linear  : logits = W concat(x_1..x_3) + b
  mlp     : concat -> H -> H -> 3 (GELU)
  attn    : group tokens -> d -> K self-attention+MLP blocks over the 3 group tokens -> mean pool -> 3  (pairwise interaction by attention)
  relnet  : permutation-invariant relation network: rho( sum_{g != h} phi([x_g, x_h]), sum_g psi(x_g) ) -> 3 (group identity dropped)
Parameter counts are reported by count_params."""
import numpy as np, jax, jax.numpy as jnp
import v8lib as V
L, E = V.L, V.E


def features(X, oracle=False, drop_id=False):
    n, R = V.ingredients(X)                                         # N,6,3 ; N,6,3,2
    f = [np.eye(6)[n], np.eye(3)[R].reshape(*R.shape[:3], 6)]
    if not drop_id: f.append(np.broadcast_to(np.eye(3)[None, None], n.shape + (3,)))
    if oracle: f.append(np.eye(3)[np.minimum(V.eligibility_inv7(X), 2)])
    return np.concatenate(f, -1).astype(np.float32)                 # N,6,3,F


def dense(key, a, b): return dict(W=jax.random.normal(key, (a, b)) * a ** -0.5, b=jnp.zeros(b))
def mlp_init(key, dims): ks = jax.random.split(key, len(dims)); return [dense(k, a, b) for k, a, b in zip(ks, dims[:-1], dims[1:])]
def mlp_apply(ps, x):
    for i, l in enumerate(ps):
        x = x @ l["W"] + l["b"]
        if i < len(ps) - 1: x = jax.nn.gelu(x, approximate=True)
    return x


def init(kind, F, seed, H=128, d=32, depth=1):
    k = jax.random.PRNGKey(seed); ks = jax.random.split(k, 8)
    if kind == "linear": return dict(l=dense(ks[0], 3 * F, 3))
    if kind == "mlp": return dict(m=mlp_init(ks[0], [3 * F, H, H, 3]))
    if kind == "attn":
        blocks = []
        for i in range(depth):
            kk = jax.random.split(ks[1 + i], 5)
            blocks.append(dict(Wq=jax.random.normal(kk[0], (d, d)) * d ** -0.5, Wk=jax.random.normal(kk[1], (d, d)) * d ** -0.5,
                               Wv=jax.random.normal(kk[2], (d, d)) * d ** -0.5, Wo=jax.random.normal(kk[3], (d, d)) * d ** -0.5, ff=mlp_init(kk[4], [d, 4 * d, d])))
        return dict(emb=dense(ks[0], F, d), blocks=blocks, out=dense(ks[7], d, 3))
    if kind == "relnet": return dict(phi=mlp_init(ks[0], [2 * F, H, H]), psi=mlp_init(ks[1], [F, H, H]), rho=mlp_init(ks[2], [2 * H, H, 3]))
    raise ValueError(kind)


def lnorm(x): mu = x.mean(-1, keepdims=True); return (x - mu) / jnp.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5)


def apply(kind, p, f):
    """f: (..., 3, F) -> (..., 3) logits"""
    if kind == "linear": return f.reshape(*f.shape[:-2], -1) @ p["l"]["W"] + p["l"]["b"]
    if kind == "mlp": return mlp_apply(p["m"], f.reshape(*f.shape[:-2], -1))
    if kind == "attn":
        x = f @ p["emb"]["W"] + p["emb"]["b"]
        for b in p["blocks"]:
            z = lnorm(x); q, k, v = z @ b["Wq"], z @ b["Wk"], z @ b["Wv"]
            a = jax.nn.softmax(jnp.einsum("...gd,...hd->...gh", q, k) / jnp.sqrt(q.shape[-1]), -1)
            x = x + jnp.einsum("...gh,...hd->...gd", a, v) @ b["Wo"]; x = x + mlp_apply(b["ff"], lnorm(x))
        return x.mean(-2) @ p["out"]["W"] + p["out"]["b"]
    if kind == "relnet":
        pairs = [(g, h) for g in range(3) for h in range(3) if g != h]
        P = sum(mlp_apply(p["phi"], jnp.concatenate([f[..., g, :], f[..., h, :]], -1)) for g, h in pairs)
        S = sum(mlp_apply(p["psi"], f[..., g, :]) for g in range(3))
        return mlp_apply(p["rho"], jnp.concatenate([P, S], -1))
    raise ValueError(kind)


def count_params(p): return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))
