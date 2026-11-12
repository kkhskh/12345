"""Investigation VIII library: re-exports Investigation VII's r1lib/nnlib/evalrun (read-only) and adds relational-feature builders."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.join(HERE, "..", "..", "step1_r1_causal_origin_and_selective_rescue", "src")
sys.path.insert(0, V7)
import numpy as np
import r1lib as L
E, B = L.E, L.B


def ingredients(X):
    """Condition R raw relational ingredients, per question q and group g (N,6,3,*):
    n  : leave-q agreement count of g (int 0..5)
    r  : g's two reports at q (int 0..2 each) -> the pair (r_a, r_b)
    Nothing combined across groups."""
    n = E.loo(X)                                     # N,6,3
    R = np.stack([X[:, :, [a, b]] for a, b in L.GROUPS], 2)   # N,6,3,2
    return n, R


def eligibility_inv7(X):
    """Investigation VII's m_g = #{h != g : n_h = 5 and h's q-pair agrees}  (the factor in the exact closed form)."""
    n = E.loo(X); V = np.stack([L.votes(X, qq) for qq in range(6)], 1); perf = (n == 5) & (V >= 0)
    return perf.sum(-1, keepdims=True) - perf


def eligibility_mandate(X):
    """The mandate's written formula: m'_g = sum_{h != g} 1[n_h = 5] 1[r_h(q) = r_g(q)]  (vote-matching with g; see MATHEMATICAL_RELATIONAL_OPERATION.md)."""
    n = E.loo(X); V = np.stack([L.votes(X, qq) for qq in range(6)], 1); out = np.zeros(n.shape, int)
    for g in range(3):
        for h in range(3):
            if h != g: out[..., g] += ((n[..., h] == 5) & (V[..., h] >= 0) & (V[..., h] == V[..., g]) & (V[..., g] >= 0)).astype(int)
    return out


# ------------------------------------------------------------------ checkpoint I/O for parameter trees with lists
def save_tree(path, tree):
    """Pickle-free: every leaf stored under a path key ('a/0/W'); lists marked by integer components."""
    flat = {}
    def rec(x, pre):
        if isinstance(x, dict):
            for k, v in x.items(): rec(v, pre + [str(k)])
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x): rec(v, pre + [f"#{i}"])
        else: flat["/".join(pre)] = np.asarray(x)
    rec(tree, []); np.savez(path, **flat)


def load_tree(path):
    """Loads save_tree files; also loads this package's earlier ckpt.save files of list-containing trees (object arrays written by
    this package's own scripts; loaded with allow_pickle for that reason only)."""
    z = np.load(path, allow_pickle=True)
    if any("/" in k for k in z.files):
        root = {}
        for k in z.files:
            parts = k.split("/"); cur = root
            for a, b in zip(parts[:-1], parts[1:]):
                cur = cur.setdefault(a, {})
            cur[parts[-1]] = z[k]
        def fix(x):
            if isinstance(x, dict):
                if x and all(k.startswith("#") for k in x): return [fix(x[f"#{i}"]) for i in range(len(x))]
                return {k: fix(v) for k, v in x.items()}
            return x
        return fix(root)
    out = {}
    for k in z.files:
        v = z[k]; v = v.item() if v.dtype == object and v.shape == () else (list(v) if v.dtype == object else v)
        parts = k.split("."); cur = out
        for a in parts[:-1]: cur = cur.setdefault(a, {})
        cur[parts[-1]] = v
    return out
