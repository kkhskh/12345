"""XIV GATE library: model loading (continuation/deepening checkpoints and XIV snapshots), forward with hooks, exact rules, edits.
Task: tasks_ext.sample_gate (K=3, Q=6, beta=0.9).  Token layout: question q sources 6q..6q+5 (G = 6q, 6q+1; E = 6q+2..6q+5),
readout 36+q, episode token 42."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_foundational_atlas_continuation/src", "step1_relational_mechanism_deepening/src", "step1_latent_reliability/src", "step1_relational_computation_atlas/src"):
    sys.path.insert(0, os.path.join(ROOT, p_))
import numpy as np, jax, jax.numpy as jnp
import train_tasks as TT, tasks_ext as TX, fwd3 as F3, ckpt as C
K, Q, BETA = 3, 6, 0.9; E_ = (1 - BETA) / (K - 1)
DEV_FINALS = {"s0": os.path.join(ROOT, "step1_foundational_atlas_continuation", "checkpoints", "gate_s0.npz"),
              "s1": os.path.join(ROOT, "step1_relational_mechanism_deepening", "checkpoints", "gate_s1.npz"),
              "s2": os.path.join(ROOT, "step1_relational_mechanism_deepening", "checkpoints", "gate_s2.npz")}
_TREE = None
def load(path):
    global _TREE
    if path.endswith(".npz") and os.path.basename(path).startswith("gate_"):
        p = C.load(path); p.pop("WX", None); return jax.tree_util.tree_map(jnp.asarray, p)
    if _TREE is None: _TREE = jax.tree_util.tree_structure(TT.init("gate", 0))
    z = np.load(path); return jax.tree_util.tree_unflatten(_TREE, [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))])
def snap_steps(d): return sorted(int(f[4:-4]) for f in os.listdir(d) if f.startswith("step") and f.endswith(".npz"))
_FJ = jax.jit(lambda p, X_: jax.nn.softmax(F3.forward(p, X_)["answer"], -1))      # compiled once (parameters are an argument)
def probs(p, X, hook=None, bs=2000):
    if hook is None: return np.concatenate([np.asarray(_FJ(p, jnp.asarray(X[i:i + bs]))) for i in range(0, len(X), bs)])
    f = lambda X_: jax.nn.softmax(F3.forward(p, X_, hook)["answer"], -1)
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + bs]))) for i in range(0, len(X), bs)])
def lik(X, S):
    ys = np.arange(K); out = np.ones(X.shape[:2] + (K,))
    for s in S: out = out * np.where(X[:, :, s, None] == ys, BETA, E_)
    return out
def gate(X): return (X[:, :, 0] + X[:, :, 1]) % K == 0
def rule(name, X):
    if name == "bayes": return TX.bayes_gate(X)
    if name == "gatefree": return TX.bayes_gate_no_relation(X)
    if name == "sourcewise":
        m = 0.5 * BETA + 0.5 / K; mo = (1 - m) / (K - 1); ys = np.arange(K); L = np.ones(X.shape[:2] + (K,))
        for s in (2, 3, 4, 5): L = L * np.where(X[:, :, s, None] == ys, m, mo)
        return L / L.sum(-1, keepdims=True)
    raise ValueError(name)
def kl(P, R): return float((P * (np.log(P + 1e-12) - np.log(R + 1e-12))).sum(-1).mean())
def gate_flip(X, rng):
    """redraw x1 so that the gate state switches (on-support); evidence kept."""
    Xf = X.copy(); g = gate(X); x0 = X[:, :, 0]
    closed = (-x0 + rng.integers(1, K, x0.shape)) % K
    Xf[:, :, 1] = np.where(g, closed, (-x0) % K); return Xf
def gate_keep_shift(X, rng):
    """relation-preserving joint shift (x0, x1) -> (x0 + k, x1 - k), k in {1, 2}: g unchanged, both gate tokens changed."""
    Xs = X.copy(); k = rng.integers(1, K, X.shape[:2]); Xs[:, :, 0] = (X[:, :, 0] + k) % K; Xs[:, :, 1] = (X[:, :, 1] - k) % K; return Xs
def swap35(X):
    Xs = X.copy(); Xs[:, :, 3], Xs[:, :, 5] = X[:, :, 5], X[:, :, 3]; return Xs
def unique(P, tol=1e-6): s = np.sort(P, -1); return s[..., -1] - s[..., -2] > tol
def following(Pm0, Pm1, R0, R1):
    """fraction of decisive edits (rule argmax changes, unique before and after) on which the model's argmax moves to the rule's new argmax."""
    dec = unique(R0) & unique(R1) & (R0.argmax(-1) != R1.argmax(-1))
    if dec.sum() == 0: return float("nan"), 0
    mv = (Pm1.argmax(-1) != Pm0.argmax(-1)) & (Pm1.argmax(-1) == R1.argmax(-1)); return float(mv[dec].mean()), int(dec.sum())
