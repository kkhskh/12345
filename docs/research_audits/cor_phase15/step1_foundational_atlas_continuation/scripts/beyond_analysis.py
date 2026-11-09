"""Beyond latent reliability (mandate §11): analysis of the trained GATE and TRANS Transformers (seed 0).
GATE: accuracy vs exact Bayes and vs the best no-relation rule; input intervention 'flip the gate' (re-draw x1 so the
      constraint x0 + x1 = 0 mod K switches, all evidence sources kept): does the model switch which evidence pair it follows,
      as Bayes does?  Routing: last-layer attention mass of readout q on sources 2,3 vs 4,5 by gate state.
      Activation patch: the residual at q's gate tokens (0,1) at the last-layer input from the flipped run.
TRANS: accuracy by question type vs exact Bayes and the one-hop (no-composition) reference; input interventions
      (a) mask the C reports of all BC questions (removes tau evidence) and (b) mask the A reports of all AB questions
      (removes sigma evidence); the effect on C-only accuracy for model and Bayes; attention of C-only readouts to other questions.
Streams 9_200_000+.  Writes logs/beyond_analysis.json."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import train_tasks as TT, tasks_ext as T, fwd as F, ckpt as C
K, Q = 3, 6; res = {}
def model(path):
    p = jax.tree_util.tree_map(jnp.asarray, C.load(path)); p.pop("WX", None)
    @jax.jit
    def run(X):
        rec = {}
        def hk(n, a):
            if n in ("pattern2", "resid_pre2"): rec[n] = a
            return a
        o = F.forward(p, X, hk); return jax.nn.softmax(o["answer"], -1), rec["pattern2"], rec["resid_pre2"]
    @jax.jit
    def patched(X, src, mask):
        def hk(n, a):
            return a + (src - a) * mask[None, :, None] if n == "resid_pre2" else a
        return jax.nn.softmax(F.forward(p, X, hk)["answer"], -1)
    return run, patched
CK = os.path.join(HERE, "..", "checkpoints")
if os.path.exists(os.path.join(CK, "gate_s0.npz")):
    run, patched = model(os.path.join(CK, "gate_s0.npz")); rng = np.random.default_rng(9_200_000)
    d = T.sample_gate(6000, rng); X, Y = d["X"], d["Y"]; P, A, Rr = map(np.asarray, run(jnp.asarray(X)))
    pb = T.bayes_gate(X); pn = T.bayes_gate_no_relation(X)
    r = dict(acc=float((P.argmax(-1) == Y).mean()), bayes=float((pb.argmax(-1) == Y).mean()), no_relation=float((pn.argmax(-1) == Y).mean()),
             agree_bayes=float((P.argmax(-1) == pb.argmax(-1)).mean()), KL_to_bayes=float((pb * (np.log(pb + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()))
    # flip the gate in question q only (re-draw x1 to switch the constraint), for each q
    agree_flip, follow = [], []
    for q in range(Q):
        Xf = X.copy(); op = (X[:, q, 0] + X[:, q, 1]) % K == 0
        Xf[:, q, 1] = np.where(op, ((-X[:, q, 0]) % K + 1 + rng.integers(0, K - 1, len(X))) % K, (-X[:, q, 0]) % K)
        Pf = np.asarray(run(jnp.asarray(Xf))[0]); pbf = T.bayes_gate(Xf)
        ch = pb[:, q].argmax(-1) != pbf[:, q].argmax(-1)
        agree_flip.append(float((Pf[:, q].argmax(-1) == pbf[:, q].argmax(-1))[ch].mean()))
        # activation patch: gate tokens 6q, 6q+1 from the flipped run
        src = np.asarray(run(jnp.asarray(Xf))[2]); m = np.zeros(43, np.float32); m[6 * q:6 * q + 2] = 1
        Pp = np.asarray(patched(jnp.asarray(X), jnp.asarray(src), jnp.asarray(m)))
        follow.append(float((Pp[:, q].argmax(-1) == pbf[:, q].argmax(-1))[ch].mean()))
    r["flip_gate_input_agree_with_bayes_on_changed"] = float(np.mean(agree_flip)); r["patch_gate_tokens_IIA_on_changed"] = float(np.mean(follow))
    op = (X[:, :, 0] + X[:, :, 1]) % K == 0
    m23 = np.stack([A[:, :, 36 + q, 6 * q + 2:6 * q + 4].sum(-1) for q in range(Q)], -1); m45 = np.stack([A[:, :, 36 + q, 6 * q + 4:6 * q + 6].sum(-1) for q in range(Q)], -1)
    r["attn_by_head"] = {f"head{h}": dict(open_23=float(m23[:, h][op].mean()), open_45=float(m45[:, h][op].mean()), closed_23=float(m23[:, h][~op].mean()), closed_45=float(m45[:, h][~op].mean())) for h in range(A.shape[1])}
    res["GATE"] = r; print("GATE", json.dumps(r), flush=True)
if os.path.exists(os.path.join(CK, "trans_s0.npz")):
    run, patched = model(os.path.join(CK, "trans_s0.npz")); rng = np.random.default_rng(9_300_000)
    d = T.sample_trans(6000, rng); X, Y, typ = d["X"], d["Y"], d["typ"]; P, A, _ = map(np.asarray, run(jnp.asarray(X)))
    pb = T.bayes_trans(X)["post_y"]; p1 = T.bayes_trans_onehop(X)["post_y"]; own = T.bayes_trans(X)["own_y"]
    r = {}
    for t, nm in enumerate(["AB", "BC", "C_only"]):
        s = typ == t
        r[nm] = dict(model=float((P.argmax(-1) == Y)[s].mean()), bayes=float((pb.argmax(-1) == Y)[s].mean()), onehop=float((p1.argmax(-1) == Y)[s].mean()),
                     own=float((own.argmax(-1) == Y)[s].mean()), agree_bayes=float((P.argmax(-1) == pb.argmax(-1))[s].mean()))
    for nm, fn in [("mask_C_in_BC (no tau evidence)", lambda Z: np.where(((Z[:, :, 0] == T.MASK) & (Z[:, :, 2] != T.MASK))[..., None] & (np.arange(6) >= 4)[None, None], T.MASK, Z)),
                   ("mask_A_in_AB (no sigma evidence)", lambda Z: np.where(((Z[:, :, 4] == T.MASK))[..., None] & (np.arange(6) < 2)[None, None], T.MASK, Z))]:
        Xm = fn(X); Pm = np.asarray(run(jnp.asarray(Xm))[0]); pbm = T.bayes_trans(Xm)["post_y"]; s = typ == 2
        r[nm] = dict(model_Conly=float((Pm.argmax(-1) == Y)[s].mean()), bayes_Conly=float((pbm.argmax(-1) == Y)[s].mean()))
    oth = np.stack([A[:, :, 36 + q, :36].sum(-1) - A[:, :, 36 + q, 6 * q:6 * q + 6].sum(-1) for q in range(Q)], -1)
    r["attn_other_questions_by_type"] = {f"head{h}": {nm: float(oth[:, h][typ == t].mean()) for t, nm in enumerate(["AB", "BC", "C_only"])} for h in range(A.shape[1])}
    res["TRANS"] = r; print("TRANS", json.dumps(r), flush=True)
json.dump(res, open(os.path.join(HERE, "..", "logs", "beyond_analysis.json"), "w"), indent=1)
