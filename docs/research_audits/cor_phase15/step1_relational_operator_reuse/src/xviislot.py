"""XVII per-slot diagnostics and slot-identity interventions for the GATE Transformer (XIV/XV/XVI architecture; nets.forward semantics).

Architecture facts used (step1_latent_reliability/src/nets.py): bidirectional attention, no positional encoding, no mask.  Question identity
enters ONLY through E_q[q], added to the 6 source tokens and the readout token of question q.  Hence (Theorem P, see
COMPETING_NEURAL_MECHANISMS.md) the network is exactly equivariant under any permutation of question slots applied jointly to the input
contents and to the rows of E_q.

Per-slot measures (one slot at a time; other slots untouched):
  F_G[q]  gate-flip following at slot q: only slot q's last gate source is redrawn so that its g switches; over flips that change the
          Bayes argmax at q (unique before and after), the fraction on which the model's argmax at q moves to the new Bayes argmax.
  KL[q], acc[q], conf[q] at slot q;  A_conf[q]: XV's swap(4,5) agreement index at slot q, normalised by sharpness (as src/xvidiag.py).
Interventions (all keep every slot's own input content; nothing is copied between slots):
  relabel(i, j, lo, hi): inside layers lo..hi-1 the slot codes of slots i and j are exchanged: before layer lo, x_tok(i) += E_q[j] - E_q[i]
          and x_tok(j) += E_q[i] - E_q[j] on the 7 tokens of each slot; after layer hi-1 the exchange is undone.  lo = 0 means from the
          embedding; hi = L means through the final layer (then the readout is read with the exchanged codes).  Full window (0, L) is
          exactly a joint permutation of E_q rows (Theorem P).
  eq_edit(j, v): E_q[j] replaced by v (a parameter edit; used for interpolation and controls)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "step1_latent_reliability", "src")); sys.path.insert(0, os.path.join(ROOT, "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax, jax.numpy as jnp
import nets as N, xvitask as T, xvidiag as XD
K, Q, S = 3, 6, 6


def tok_index(q): return np.array([q * S + s for s in range(S)] + [S * Q + q])


def forward(p, X, relabel=None, pat=None):
    """nets.forward with an optional identity-relabel window relabel = (i, j, lo, hi) and an optional attention-routing transplant
    pat = (donor, recipient, layers, heads, rows): in the listed layers and heads, the attention rows of the recipient slot's tokens (rows:
    'all' 7 tokens, 'evid' evidence tokens, 'gate' gate tokens, 'read' readout) are replaced by the donor slot's corresponding rows with the
    columns of the two slots exchanged, so the recipient attends to ITS OWN tokens with the donor's weights.  Values are the recipient's
    own; no activation or evidence is moved."""
    x = N.embed(p, X); L = len(p["layers"])
    def swap(x, sign):
        i, j = relabel[0], relabel[1]; d = (p["Eq"][j] - p["Eq"][i]) * sign
        x = x.at[:, tok_index(i)].add(d); return x.at[:, tok_index(j)].add(-d)
    for l, lay in enumerate(p["layers"]):
        if relabel is not None and l == relabel[2]: x = swap(x, 1.0)
        z = N.ln(x, lay["g1"], lay["b1"])
        q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        a = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1]), -1)
        if pat is not None and l in pat[2]: a = _route(a, pat)
        x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
        x = x + jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True) @ lay["W2"] + lay["c2"]
        if relabel is not None and l == relabel[3] - 1 and relabel[3] < L: x = swap(x, -1.0)
    zf = N.ln(x, p["gf"], p["bf"]); return zf[:, S * Q:S * Q + Q] @ p["WA"]
def _route(a, pat):
    d, r, _, heads, rows = pat; ti, tj = tok_index(d), tok_index(r)
    sel = {"all": list(range(7)), "evid": [3, 4, 5], "gate": [0, 1, 2], "read": [6]}[rows]
    perm = np.arange(a.shape[-1]); perm[ti] = tj; perm[tj] = ti            # column exchange donor <-> recipient tokens
    src = a[:, :, ti[sel]][..., perm]                                       # donor rows, columns re-indexed onto the recipient's tokens
    hs = np.array(heads)
    return a.at[:, hs[:, None], tj[sel][None, :]].set(src[:, hs])
_fj = jax.jit(forward, static_argnums=(2, 3))
def probs(p, X, relabel=None, bs=1500, pat=None):
    return np.concatenate([np.asarray(jax.nn.softmax(_fj(p, jnp.asarray(X[i:i + bs]), relabel, pat), -1)) for i in range(0, len(X), bs)])


def cen(P): L_ = np.log(P + 1e-12); return L_ - L_.mean(-1, keepdims=True)
def unique(P, tol=1e-6): s = np.sort(P, -1); return s[..., -1] - s[..., -2] > tol


class SlotDiag:
    def __init__(self, cond, stream=99_800_000, n=1500, cache=None):
        self.cond = cond; self.a, self.b1, self.b0, self.nu = XD.parse(cond); rng = np.random.default_rng(stream)
        d = T.sample(n, rng, self.a, self.b1, self.b0, self.nu); self.X, self.Y = d["X"], d["Y"]
        self.R = XD.refs(cond, cache or os.path.join(HERE, "..", "logs")); L = lambda nm, X: XD.lookup(self.R, nm, X)
        self.PB = L("bayes", self.X); self.Xf, self.dec, self.newG = [], [], []
        for q in range(Q):     # flip only slot q's gate
            Xf = self.X.copy(); Xf[:, q] = T.gate_flip(self.X[:, q][:, None], self.a, rng)[:, 0]; PBf = L("bayes", Xf)[:, q]
            self.Xf.append(Xf); self.dec.append(unique(self.PB[:, q]) & unique(PBf) & (self.PB[:, q].argmax(-1) != PBf.argmax(-1))); self.newG.append(PBf.argmax(-1))
        self.Xw = T.swap_pair_single(self.X); self.dG = cen(L("gatefree", self.Xw)) - cen(L("gatefree", self.X))
        self.dBw = cen(L("bayes", self.Xw)) - cen(self.PB); self.dO = cen(L("order1", self.Xw)) - cen(L("order1", self.X))
        self.balance = dict(answer_freq=[np.bincount(self.Y[:, q], minlength=K).tolist() for q in range(Q)], gate_rate=T.gate(self.X, self.a).mean(0).tolist(),
                            decisive_flips=[int(m.sum()) for m in self.dec])

    def score(self, p, relabel=None, pat=None):
        P0 = probs(p, self.X, relabel, pat=pat); Pw = probs(p, self.Xw, relabel, pat=pat); out = {k: [] for k in ("F_G", "KL", "acc", "conf", "A_p")}
        for q in range(Q):
            Pf = probs(p, self.Xf[q], relabel, pat=pat)[:, q]; a0, af = P0[:, q].argmax(-1), Pf.argmax(-1)
            out["F_G"].append(float(((af != a0) & (af == self.newG[q]))[self.dec[q]].mean()))
            pb = self.PB[:, q]; out["KL"].append(float((pb * (np.log(pb + 1e-12) - np.log(P0[:, q] + 1e-12))).sum(-1).mean()))
            out["acc"].append(float((a0 == self.Y[:, q]).mean())); out["conf"].append(float(P0[:, q].max(-1).mean()))
            m = np.abs(self.dG[:, q]).max(-1) > 0.05; dmw = cen(Pw[:, q]) - cen(P0[:, q])
            Z = np.stack([self.dG[m, q].ravel(), self.dBw[m, q].ravel(), self.dO[m, q].ravel()], 1); out["A_p"].append(float(np.linalg.lstsq(Z, dmw[m].ravel(), rcond=None)[0][0]))
        return out


def eq_edit(p, j, v): p2 = dict(p); p2["Eq"] = p["Eq"].at[j].set(v); return p2


def theorem_P_check(p, X, perm):
    """max |f_{theta with E_q permuted}(X permuted) - f_theta(X) permuted| over answer logits."""
    p2 = dict(p); p2["Eq"] = p["Eq"][np.array(perm)]
    Xp = X[:, np.array(perm)]; a = np.asarray(forward(p, jnp.asarray(X)))[:, np.array(perm)]; b = np.asarray(forward(p2, jnp.asarray(Xp)))
    return float(np.abs(a - b).max())
