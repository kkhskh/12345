"""Mechanism of successful relational readers (Condition R, 'attn'/'attn2' group-attention readers).  Usage: mech_reader.py TAG [...]
(TAG = checkpoints/compose/<tag>.npz).  On development F45 pairs (R1/R2/R3), F34-R1, F54o:
  base      : responses of the trained reader
  selfonly  : attention replaced by the identity (each group token attends only to itself) in every block -> the reader becomes additive
              over groups up to the shared output map (FORMAL_RESULTS Prop. C predicts regime-independent responses)
  noperf    : attention from every group to OTHER groups that are perfect-and-agree-at-q is removed (logit -inf; renormalised)
  noother   : attention to all other groups removed except perfect-and-agreeing ones (keeps only the eligibility-relevant partners)
  probes    : held-out linear decodability of m_g (Investigation VII eligibility) and of [n_g = 5] from each group token after the last block
  interchange: in R1 pairs, replace the after-episode group-token states of the TWO perfect groups' *partners*... (see doc); here: replace all
              group-token states after the attention sublayer of block 1 by their before-episode values except for the component on the
              m-decoder subspace (rank 2 per token) -> share of the response carried by the m-subspace; random rank-2 subspace control."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
import v8lib as V, readers as RD
sys.path.insert(0, V.V7); import ckpt as C
L, E, B = V.L, V.E, V.B
CONF = os.environ.get("PHASE", "dev") == "conf"   # confirmation: eval_conf8 (stream 18_900_000), checkpoints/compose_conf
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_conf8.npz") if CONF else os.path.join(V.V7, "..", "logs", "eval_dev.npz"), allow_pickle=True)); out = {}
def run(p, f, mode="base", elig=None, patch=None):
    """f (n,3,F).  mode: base|selfonly|noperf|noother.  elig (n,3) bool eligibility (perfect & agree at q).  patch: (states_after, proj) to
    substitute after the first block's attention: x_patched = x_b + P(x_a - x_b)."""
    x = f @ p["emb"]["W"] + p["emb"]["b"]; states = []
    for bi, b in enumerate(p["blocks"]):
        z = RD.lnorm(x); q, k, v = z @ b["Wq"], z @ b["Wk"], z @ b["Wv"]
        s = jnp.einsum("ngd,nhd->ngh", q, k) / jnp.sqrt(q.shape[-1])
        eye = jnp.eye(3, dtype=bool)[None]
        if mode == "selfonly": s = jnp.where(eye, s, -1e9)
        if mode == "noperf": s = jnp.where(eye | ~elig[:, None, :], s, -1e9)
        if mode == "noother": s = jnp.where(eye | elig[:, None, :], s, -1e9)
        a = jax.nn.softmax(s, -1); x = x + jnp.einsum("ngh,nhd->ngd", a, v) @ b["Wo"]
        if patch is not None and bi == 0:
            xa, Pm = patch; x = x + jnp.einsum("de,nge->ngd", Pm, xa - x)
        x = x + RD.mlp_apply(b["ff"], RD.lnorm(x)); states.append(x)
    return np.asarray(x.mean(-2) @ p["out"]["W"] + p["out"]["b"]), [np.asarray(s_) for s_ in states]
def first_block_state(p, f):
    x = f @ p["emb"]["W"] + p["emb"]["b"]; b = p["blocks"][0]; z = RD.lnorm(x); q, k, v = z @ b["Wq"], z @ b["Wk"], z @ b["Wv"]
    a = jax.nn.softmax(jnp.einsum("ngd,nhd->ngh", q, k) / jnp.sqrt(q.shape[-1]), -1); return np.asarray(x + jnp.einsum("ngh,nhd->ngd", a, v) @ b["Wo"])
def elig_of(X, q):
    idx = np.arange(len(X)); n = E.loo(X)[idx, q]; return jnp.asarray((n == 5) & (L.votes(X, q) >= 0))
def feats_q(X, q): return jnp.asarray(RD.features(X)[np.arange(len(X)), q])
for tag in sys.argv[1:]:
    p = jax.tree_util.tree_map(jnp.asarray, V.load_tree(os.path.join(HERE, "..", "checkpoints", "compose_conf" if CONF else "compose", tag + ".npz"))); res = {}
    sets = {}
    for f_ in ("F45", "F34"):
        for rg in (("R1", "R2", "R3") if f_ == "F45" else ("R1",)):
            s = ev[f"orient_{f_}"] & (ev[f"reg_{f_}"] == rg); sets[f"{f_}_{rg}"] = (ev["X"][s], ev[f"X_{f_}"][s], ev["q"][s], ev[f"vote_{f_}"][s], ev[f"cB_{f_}"][s], ev["cB_base"][s])
    sets["F54o"] = (ev["X_O"], ev["X_F54o"], ev["q_O"], ev["vote_F54o"], ev["cB_F54o"], ev["cB_O"])
    for mode in ("base", "selfonly", "noperf", "noother"):
        r = {}
        for nm, (Xb, Xa, q, v, cba, cbb) in sets.items():
            lb = run(p, feats_q(Xb, q), mode, elig_of(Xb, q))[0]; la = run(p, feats_q(Xa, q), mode, elig_of(Xa, q))[0]
            cl = lambda c: c - c.mean(-1, keepdims=True); dm = B.oriented_margin(cl(la), v) - B.oriented_margin(cl(lb), v); db = B.oriented_margin(cba, v) - B.oriented_margin(cbb, v)
            r[nm] = dict(n=len(q), dmM=float(dm.mean()), E=float((dm - db).mean()), absE=float(np.abs(dm - db).mean()))
        res[mode] = r
    # probes on natural decisions (fit/test split of the development natural set)
    X, q = ev["X"], ev["q"]; Fq = feats_q(X, q); _, st = run(p, Fq); H = st[-1].reshape(-1, st[-1].shape[-1]); H1 = first_block_state(p, Fq).reshape(-1, H.shape[-1])
    m = np.asarray(np.minimum(V.eligibility_inv7(X)[np.arange(len(X)), q], 2)).reshape(-1); perf = np.asarray(elig_of(X, q)).reshape(-1).astype(int)
    half = len(H) // 2; pr = {}
    for nm, Z in (("after_last_block", H), ("after_block1_attention", H1)):
        for tn, y in (("m", m), ("perfect", perf)):
            clf = LogisticRegression(max_iter=3000).fit(Z[:half], y[:half]); pr[f"{nm}_{tn}"] = float(clf.score(Z[half:], y[half:]))
            pr[f"{nm}_{tn}_majority"] = float(max(np.bincount(y[half:]) / len(y[half:])))
    res["probes"] = pr
    # interchange on m-subspace after block-1 attention (R1 pairs)
    clf = LogisticRegression(max_iter=3000).fit(H1[:half], m[:half]); W = clf.coef_; Qm, _ = np.linalg.qr(W.T); Qm = Qm[:, :min(3, Qm.shape[1])]
    rng = np.random.default_rng(0); Qr, _ = np.linalg.qr(rng.standard_normal((H1.shape[1], Qm.shape[1])))
    Xb, Xa, q1, v, cba, cbb = sets["F45_R1"]; fb, fa = feats_q(Xb, q1), feats_q(Xa, q1); xa = first_block_state(p, fa)
    lb = run(p, fb)[0]; la = run(p, fa)[0]; cl = lambda c: c - c.mean(-1, keepdims=True); om = lambda l: B.oriented_margin(cl(l), v)
    full = om(la) - om(lb); ic = {}
    for nm, Qx in (("m_subspace", Qm), ("random_subspace", Qr)):
        Pm = jnp.asarray(Qx @ Qx.T); lp = run(p, fb, patch=(jnp.asarray(xa), Pm))[0]; ic[nm] = dict(share=float(((om(lp) - om(lb)) * full).sum() / (full ** 2).sum()), mean_dm=float((om(lp) - om(lb)).mean()))
    ic["full_mean_dm"] = float(full.mean()); res["interchange_R1"] = ic
    out[tag] = res; print(tag, {m_: {k: round(v_["dmM"], 3) for k, v_ in r.items()} for m_, r in res.items() if m_ in ("base", "selfonly", "noperf", "noother")}, res["probes"], ic, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "mech_reader_conf.json" if CONF else "mech_reader.json"), "w"), indent=1)
