"""C/D: mediator localisation and causal tests for learned models (see PROTOCOL.md for the frozen version).

Terminology
  site      : (hook name, positions). Positions 'R' = last token; 'ALL' = every position flattened.
  state     : site activation, shape (N, D) after flattening the chosen positions.
  candidate : an orthonormal basis U (D x k) of a subspace of the site state.

Candidate mediators
  S_rel  (P1, output-free)   top-k right singular vectors of {h(agree_i) - h(conflict_i)} over matched pairs with
                             IDENTICAL aggregate (z, p). Uses generator labels only; the aggregate cancels exactly
                             in the difference. k = smallest with >= 90% singular energy (cap 8).
  S_das  (P2, target-using)  distributed alignment search: U trained so that interchanging U-components from the
                             source member of a matched pair makes the model output the TASK TARGET of the source.
                             Uses task-target (high-level causal model) labels, never the evaluated model margin.
  S_grad (output-derived)    top-k eigenvectors of E[g g^T], g = d margin / d state  (direct downstream sensitivity).
  S_pca  (generic)           top-k principal components of the site state.
  S_rand (random)            20 random k-dim subspaces.
  S_alt  (equally decodable) k directions in the orthogonal complement of S_rel chosen greedily to decode s
                             (logistic regression), i.e. relation-decodable but not the identified subspace.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.linear_model import LogisticRegression

import model as M

CODE = {"AGG-LIN": "T2", "AGG-NL": "T6", "REL-XNOR": "T1", "REL-GATE": "T3", "REL-MIS": "T4",
        "K3-BAYES": "T5a", "K3-GATE": "T5b", "REL-XOR3": "T7"}


def to_inp(task, d):
    if task.startswith("K3"):
        return dict(zA=jnp.asarray(d["zA"], jnp.float32), zB=jnp.asarray(d["zB"], jnp.float32), p=jnp.asarray(d["p"]))
    return dict(a=jnp.asarray(d["a"], jnp.float32), b=jnp.asarray(d["b"], jnp.float32), p=jnp.asarray(d["p"], jnp.float32))


def positions(P, task, spec):
    T = M.n_tokens(CODE[task])
    return [T - 1] if spec == "R" else list(range(T))


class Site:
    def __init__(self, P, task, hook, pos="R"):
        self.P, self.task, self.code, self.hook, self.pos = P, task, CODE[task], hook, positions(P, task, pos)

    def get(self, d):
        store = {}
        def hk(name, arr):
            if name == self.hook:
                store["x"] = np.asarray(arr)
            return arr
        lg = np.asarray(M.forward(self.P, self.code, to_inp(self.task, d), hk))
        x = store["x"][:, self.pos, :]
        return lg, x.reshape(x.shape[0], -1)

    def run_with(self, d, new_state, also=None):
        """Recompute the model with the site state replaced by new_state (N, D). `also` = extra hooks (name->fn)."""
        pos, hook = self.pos, self.hook
        def hk(name, arr):
            if name == hook:
                a = np.asarray(arr).copy()
                a[:, pos, :] = np.asarray(new_state).reshape(a.shape[0], len(pos), -1)
                arr = jnp.asarray(a)
            if also and name in also:
                arr = also[name](arr)
            return arr
        return np.asarray(M.forward(self.P, self.code, to_inp(self.task, d), hk))


def decision(task, lg):
    if task.startswith("K3"):
        return lg.argmax(1)
    return np.where(lg[:, 0] > lg[:, 1], 1.0, -1.0)


def margin(lg):
    return lg[:, 0] - lg[:, 1]


def orth(U):
    q, _ = np.linalg.qr(U)
    return q


# ----------------------------------------------------------------------------- candidates


def s_rel(site, A, C, energy=0.9, kmax=8):
    _, hA = site.get(A); _, hC = site.get(C)
    D = hA - hC
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    e = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    k = int(min(kmax, np.searchsorted(e, energy) + 1))
    return Vt[:k].T, k, (sv ** 2 / np.sum(sv ** 2))[:10]


def s_pca(site, bank, k):
    _, h = site.get(bank)
    h = h - h.mean(0)
    _, _, Vt = np.linalg.svd(h, full_matrices=False)
    return Vt[:k].T


def s_rand(D, k, seed):
    return orth(np.random.default_rng(seed).normal(size=(D, k)))


def s_alt(site, bank, U, k):
    _, h = site.get(bank)
    s = bank["s"]
    Pc = np.eye(h.shape[1]) - U @ U.T
    dirs = []
    for _ in range(k):
        x = h @ Pc
        w = LogisticRegression(C=1.0, max_iter=3000).fit(x, s).coef_[0] @ Pc
        if np.linalg.norm(w) < 1e-9:
            break
        w = w / np.linalg.norm(w); dirs.append(w)
        Pc = Pc - np.outer(w, w)
    return orth(np.array(dirs).T) if dirs else None


def s_grad(site, bank, k):
    """Directions of largest margin sensitivity at the site (output-derived)."""
    P, code, hook, pos = site.P, site.code, site.hook, site.pos
    _, h0 = site.get(bank)
    inp = to_inp(site.task, bank)
    def f(state):
        def hk(name, arr):
            if name == hook:
                arr = arr.at[:, pos, :].set(state.reshape(arr.shape[0], len(pos), -1))
            return arr
        lg = M.forward(P, code, inp, hk)
        return (lg[:, 0] - lg[:, 1]).sum() if not site.task.startswith("K3") else \
            (lg.max(1) - jnp.sort(lg, 1)[:, -2]).sum()
    g = np.asarray(jax.grad(f)(jnp.asarray(h0)))
    _, _, Vt = np.linalg.svd(g, full_matrices=False)
    return Vt[:k].T


def s_das(site, A, C, k, steps=600, seed=0, lr=0.03):
    """Train U so that interchange (both directions) yields the source member's task target."""
    P, code, hook, pos, task = site.P, site.code, site.hook, site.pos, site.task
    _, hA = site.get(A); _, hC = site.get(C)
    inA, inC = to_inp(task, A), to_inp(task, C)
    K = M.n_classes(code)
    def lab(d):
        if task.startswith("K3"):
            return jax.nn.one_hot(jnp.asarray(d["y"]), K)
        y = jnp.asarray(d["y"]); return jnp.stack([(y > 0).astype(jnp.float32), (y < 0).astype(jnp.float32)], 1)
    yA, yC = lab(A), lab(C)
    W0 = jnp.asarray(np.random.default_rng(seed).normal(size=(hA.shape[1], k)), jnp.float32)
    def patched_logits(W, inp, hb, hs):
        U, _ = jnp.linalg.qr(W)
        new = hb + (hs - hb) @ U @ U.T
        def hk(name, arr):
            if name == hook:
                arr = arr.at[:, pos, :].set(new.reshape(arr.shape[0], len(pos), -1))
            return arr
        return M.forward(P, code, inp, hk)
    def loss(W):
        l1 = -(yC * jax.nn.log_softmax(patched_logits(W, inA, jnp.asarray(hA), jnp.asarray(hC)))).sum(-1).mean()
        l2 = -(yA * jax.nn.log_softmax(patched_logits(W, inC, jnp.asarray(hC), jnp.asarray(hA)))).sum(-1).mean()
        return l1 + l2
    opt = optax.adam(lr); st = opt.init(W0)
    g = jax.jit(jax.value_and_grad(loss))
    W = W0
    for _ in range(steps):
        l, gr = g(W)
        up, st = opt.update(gr, st); W = optax.apply_updates(W, up)
    return orth(np.asarray(W))


# ----------------------------------------------------------------------------- interventions


def interchange(site, base, src, U):
    """Patch the U-component of the source state into the base run (and the full state if U is None)."""
    _, hb = site.get(base); _, hs = site.get(src)
    new = hs if U is None else hb + (hs - hb) @ U @ U.T
    return site.run_with(base, new)


def model_dec(site, d):
    return decision(site.task, site.get(d)[0])


def iia(site, A, C, U):
    """Interchange-intervention accuracy, MODEL-relative: over matched pairs on which the model's own decisions
    differ, the fraction where patching the U-component from the source makes the base run output the model's
    decision on the source. Both directions."""
    out = []
    for base, src in ((A, C), (C, A)):
        lg = interchange(site, base, src, U)
        yb, ys = model_dec(site, base), model_dec(site, src)
        diff = yb != ys
        out.append((decision(site.task, lg)[diff] == ys[diff]))
    x = np.concatenate(out)
    return float(x.mean()) if len(x) else float("nan"), int(len(x))


def complement_interchange(site, A, C, U):
    """Sufficiency: replace everything EXCEPT U with the matched partner's state. Fraction keeping own target."""
    out = []
    for base, src in ((A, C), (C, A)):
        _, hb = site.get(base); _, hs = site.get(src)
        new = hs + (hb - hs) @ U @ U.T          # partner state with base's U-component restored
        lg = site.run_with(base, new)
        yb, ys = model_dec(site, base), model_dec(site, src)
        diff = yb != ys
        out.append(decision(site.task, lg)[diff] == yb[diff])
    x = np.concatenate(out)
    return float(x.mean()) if len(x) else float("nan")


def pair_mean_ablation(site, A, C, U):
    """Necessity: set the U-component of BOTH pair members to the pair mean (same z, both relations).
    Returns (decision-difference rate between members, accuracy vs target) before and after."""
    lgA0, hA = site.get(A); lgC0, hC = site.get(C)
    m = (hA + hC) / 2
    newA = hA + (m - hA) @ U @ U.T; newC = hC + (m - hC) @ U @ U.T
    lgA, lgC = site.run_with(A, newA), site.run_with(C, newC)
    dA0, dC0, dA, dC = (decision(site.task, x) for x in (lgA0, lgC0, lgA, lgC))
    diff = dA0 != dC0
    return dict(flip_before=float((dA0 != dC0)[diff].mean()), flip_after=float((dA != dC)[diff].mean()),
                acc_before=float(np.concatenate([dA0 == A["y"], dC0 == C["y"]]).mean()),
                acc_after=float(np.concatenate([dA == A["y"], dC == C["y"]]).mean()))


def path_split(site, A, C, U, layer):
    """For a resid_mid site: interchange through the MLP input only vs through the residual (direct) path only."""
    res = {}
    for name, mode in (("mlp_path_only", "mlp"), ("direct_path_only", "direct")):
        out = []
        for base, src in ((A, C), (C, A)):
            _, hb = site.get(base); _, hs = site.get(src)
            new = hb + (hs - hb) @ U @ U.T
            pos = site.pos
            if mode == "mlp":
                def f(arr, new=new):
                    a = np.asarray(arr).copy(); a[:, pos, :] = new.reshape(a.shape[0], len(pos), -1); return jnp.asarray(a)
                lg = np.asarray(M.forward(site.P, site.code, to_inp(site.task, base),
                                          lambda n, arr: f(arr) if n == f"mlp_in{layer}" else arr))
            else:
                def g(arr, hb=hb):
                    a = np.asarray(arr).copy(); a[:, pos, :] = hb.reshape(a.shape[0], len(pos), -1); return jnp.asarray(a)
                lg = site.run_with(base, new, also={f"mlp_in{layer}": g})
            yb, ys = model_dec(site, base), model_dec(site, src)
            diff = yb != ys
            out.append(decision(site.task, lg)[diff] == ys[diff])
        res[name] = float(np.concatenate(out).mean())
    return res


def manifold_distance(site, bank, A, C, U):
    """Patched-state distance to the nearest natural state (bank), relative to natural nearest-neighbour distances."""
    _, hbank = site.get(bank)
    _, hA = site.get(A); _, hC = site.get(C)
    new = hA + (hC - hA) @ U @ U.T
    def nn(x, exclude_self=False):
        d = ((x[:, None, :] - hbank[None, :, :]) ** 2).sum(-1)
        if exclude_self:
            d[d < 1e-12] = np.inf
        return np.sqrt(d.min(1))
    nat = nn(hA[:200]); pat = nn(new[:200])
    return dict(natural_nn_median=float(np.median(nat)), patched_nn_median=float(np.median(pat)),
                ratio=float(np.median(pat) / max(np.median(nat), 1e-12)))


# ----------------------------------------------------------------------------- content surrogates (binary tasks)


class GridCoords:
    """S-coordinates c(a, b, p) = U^T state on a product grid, for surrogate fits."""

    def __init__(self, site, U, n=61, lim=3.0):
        self.g = np.linspace(-lim, lim, n)
        Aa, Bb = np.meshgrid(self.g, self.g, indexing="ij")
        self.pvals = [-1.0, 1.0] if site.task in ("REL-GATE", "REL-XOR3") else [0.0]
        self.C = {}
        for p in self.pvals:
            d = dict(a=Aa.ravel(), b=Bb.ravel(), p=np.full(Aa.size, p))
            _, h = site.get(d)
            self.C[p] = (h @ U).reshape(n, n, -1)

    def _interp2(self, Z, a, b):
        g = self.g
        ia = np.clip(np.searchsorted(g, a) - 1, 0, len(g) - 2); ib = np.clip(np.searchsorted(g, b) - 1, 0, len(g) - 2)
        ta = ((a - g[ia]) / (g[ia + 1] - g[ia]))[:, None]; tb = ((b - g[ib]) / (g[ib + 1] - g[ib]))[:, None]
        return ((1 - ta) * (1 - tb) * Z[ia, ib] + ta * (1 - tb) * Z[ia + 1, ib] + (1 - ta) * tb * Z[ia, ib + 1] + ta * tb * Z[ia + 1, ib + 1])

    def surrogate(self, kind, d):
        out = np.zeros((len(d["a"]), next(iter(self.C.values())).shape[-1]))
        for p in self.pvals:
            m = d["p"] == p if len(self.pvals) > 1 else np.ones(len(d["a"]), bool)
            Z = self.C[p]
            if kind == "additive":       # best f(a) + g(b): two-way ANOVA main effects on the grid
                Zs = Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean((0, 1), keepdims=True)
            elif kind == "linear":       # best affine function of (a, b)
                Aa, Bb = np.meshgrid(self.g, self.g, indexing="ij")
                X = np.column_stack([np.ones(Aa.size), Aa.ravel(), Bb.ravel()])
                beta, *_ = np.linalg.lstsq(X, Z.reshape(-1, Z.shape[-1]), rcond=None)
                Zs = (X @ beta).reshape(Z.shape)
            elif kind == "aggregate":    # best function of z = a + b
                Aa, Bb = np.meshgrid(self.g, self.g, indexing="ij")
                zz = np.round((Aa + Bb) / (self.g[1] - self.g[0])).astype(int)
                Zs = np.zeros_like(Z)
                for v in np.unique(zz):
                    msk = zz == v; Zs[msk] = Z[msk].mean(0)
            elif kind == "exact":
                Zs = Z
            else:
                raise ValueError(kind)
            out[m] = self._interp2(Zs, d["a"][m], d["b"][m])
        return out

    def additive_r2(self):
        r = []
        for Z in self.C.values():
            Zs = Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean((0, 1), keepdims=True)
            r.append(1 - ((Z - Zs) ** 2).sum() / ((Z - Z.mean((0, 1))) ** 2).sum())
        return float(np.mean(r))


def surrogate_test(site, U, gc, kind, ev, A, C):
    """Replace the U-component of the state by the surrogate's prediction; keep the complement.
    Returns accuracy on ev (vs target) and matched-pair decision-difference rate on (A, C)."""
    def run(d):
        _, h = site.get(d)
        c = gc.surrogate(kind, d)
        new = h - (h @ U) @ U.T + c @ U.T
        return decision(site.task, site.run_with(d, new))
    acc = float((run(ev) == ev["y"]).mean())
    dA, dC = run(A), run(C)
    diff = model_dec(site, A) != model_dec(site, C)
    return dict(acc=acc, pair_flip=float((dA != dC)[diff].mean()))


# ----------------------------------------------------------------------------- K = 3 content surrogates


def _kfeats(d, interaction):
    zA, zB, p = d["zA"], d["zB"], d["p"]
    def poly2(Z):
        return np.column_stack([Z] + [Z[:, i:i + 1] * Z[:, j:j + 1] for i in range(3) for j in range(i, 3)])
    oA, oB, oP = np.eye(3)[zA.argmax(1)], np.eye(3)[zB.argmax(1)], np.eye(3)[p]
    F = [np.ones((len(p), 1)), poly2(zA), poly2(zB), oA, oB, oP, oA * oP.sum(1, keepdims=True)]
    F += [np.einsum("ni,nj->nij", oA, oP).reshape(len(p), -1), np.einsum("ni,nj->nij", oB, oP).reshape(len(p), -1)]
    if interaction:
        F += [np.einsum("ni,nj->nij", oA, oB).reshape(len(p), -1), np.einsum("ni,nj->nij", zA, zB).reshape(len(p), -1),
              np.einsum("ni,nj,nk->nijk", oA, oB, oP).reshape(len(p), -1)]
    return np.column_stack(F)


class KCoords:
    def __init__(self, site, U, bank):
        _, h = site.get(bank)
        self.c = h @ U
        self.Xa, self.Xi = _kfeats(bank, False), _kfeats(bank, True)
        self.Ba = np.linalg.lstsq(self.Xa, self.c, rcond=None)[0]
        self.Bi = np.linalg.lstsq(self.Xi, self.c, rcond=None)[0]

    def additive_r2(self):
        res = self.c - self.Xa @ self.Ba
        return float(1 - res.var(0).sum() / self.c.var(0).sum())

    def surrogate(self, kind, d):
        if kind == "additive":
            return _kfeats(d, False) @ self.Ba
        if kind == "interaction":
            return _kfeats(d, True) @ self.Bi
        raise ValueError(kind)
