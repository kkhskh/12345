"""GATE circuit reconstruction (Workstream B).  Usage: gate_circuit.py CKPT OUT.json
Gate flip: in question q re-draw x1 so that the constraint x0 + x1 = 0 (mod K) switches; evidence sources kept.
Scored set: questions where the Bayes argmax changes and both Bayes argmaxes are unique (same as the continuation).
(1) PATCHES (flipped run -> base run, one component at a set of positions, full downstream recomputation):
    flip-transfer = fraction of scored cases whose patched argmax equals the flipped Bayes answer.
(2) ABLATIONS on natural data (component output replaced by its dataset mean at the stated positions, every question):
    accuracy and agreement with Bayes; matched controls = the other heads of the same layer.
(3) NEGATIVE CONTROLS: flip the gate of a DIFFERENT question k != q and score q; patch from that run.
(4) B2 representation test: is the gate state linearly decodable at a site among 'answer-neutral' questions (both
    evidence pairs agree internally on the SAME value, so either gate state gives the same Bayes answer)?
Positions for question q: gate G = (6q, 6q+1), evidence E = (6q+2..6q+5), readout R = 36+q.  Stream 9_500_000."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import tasks3  # noqa
import tasks_ext as TX, fwd3 as F3, ckpt as C
from sklearn.linear_model import LogisticRegression
ck, outp = sys.argv[1], sys.argv[2]; K, Q, T = 3, 6, 43
p = C.load(ck); p.pop("WX", None); p = jax.tree_util.tree_map(jnp.asarray, p); H = p["layers"][0]["WQ"].shape[0]
SITES = ["resid_pre1", "resid_pre2", "head_out0", "head_out1", "head_out2", "attn_out0", "attn_out1", "attn_out2", "mlp_out0", "mlp_out1", "mlp_out2"]
@jax.jit
def record(X):
    rec = {}
    def hk(n, a):
        if n in SITES: rec[n] = a
        return a
    o = F3.forward(p, X, hk); return jax.nn.softmax(o["answer"], -1), rec
def run_patch(X, specs):
    """specs: list of (site, head or None, positions, source array (same shape as site))"""
    def hk(n, a):
        for site, h, pos, src in specs:
            if n != site: continue
            m = jnp.zeros(T).at[jnp.asarray(pos)].set(1.0)
            if h is None: a = a + (src - a) * m[None, :, None]
            else: a = a.at[:, h].set(a[:, h] + (src[:, h] - a[:, h]) * m[None, :, None])
        return a
    return np.asarray(jax.nn.softmax(F3.forward(p, X, hk)["answer"], -1))
rng = np.random.default_rng(9_500_000); d = TX.sample_gate(3000, rng); X = d["X"]; Y = d["Y"]
res = dict(ckpt=ck, equivalence_max_abs_diff=F3.check_equivalence(jax.tree_util.tree_map(np.asarray, p), jnp.asarray(X[:200])))
pb = TX.bayes_gate(X); P0, R0 = record(jnp.asarray(X)); P0 = np.asarray(P0); R0 = {k: np.asarray(v) for k, v in R0.items()}
res["acc"] = float((P0.argmax(-1) == Y).mean()); res["acc_bayes"] = float((pb.argmax(-1) == Y).mean())
def uniq(P): s = np.sort(P, -1); return s[..., -1] - s[..., -2] > 1e-6
G_ = lambda q: [6 * q, 6 * q + 1]; E_ = lambda q: list(range(6 * q + 2, 6 * q + 6)); R_ = lambda q: [36 + q]
experiments = {"resid_pre1@E": [("resid_pre1", None, E_)], "resid_pre2@R": [("resid_pre2", None, R_)], "resid_pre2@E": [("resid_pre2", None, E_)],
               "attn_out0@E": [("attn_out0", None, E_)], "attn_out0@G": [("attn_out0", None, G_)], "mlp_out0@E": [("mlp_out0", None, E_)],
               "attn_out1@R": [("attn_out1", None, R_)], "mlp_out1@R": [("mlp_out1", None, R_)], "attn_out1@E": [("attn_out1", None, E_)],
               "mlp_out1@E": [("mlp_out1", None, E_)], "attn_out2@R": [("attn_out2", None, R_)], "mlp_out2@R": [("mlp_out2", None, R_)],
               "attn_out1@R+mlp_out1@R": [("attn_out1", None, R_), ("mlp_out1", None, R_)],
               "attn_out0@E+mlp_out0@E": [("attn_out0", None, E_), ("mlp_out0", None, E_)]}
for h in range(H):
    experiments[f"head_out0[{h}]@E"] = [("head_out0", h, E_)]; experiments[f"head_out1[{h}]@R"] = [("head_out1", h, R_)]; experiments[f"head_out1[{h}]@E"] = [("head_out1", h, E_)]
acc = {k: [] for k in experiments}; neg = {"flip_other_question_changes_q_argmax": [], "patch_resid_pre2@R_from_other_question_flip": []}; nsc = 0
for q in range(Q):
    Xf = X.copy(); op = (X[:, q, 0] + X[:, q, 1]) % K == 0
    Xf[:, q, 1] = np.where(op, ((-X[:, q, 0]) % K + 1 + rng.integers(0, K - 1, len(X))) % K, (-X[:, q, 0]) % K)
    pf = TX.bayes_gate(Xf); sel = (pb[:, q].argmax(-1) != pf[:, q].argmax(-1)) & uniq(pb[:, q]) & uniq(pf[:, q]); nsc += int(sel.sum())
    Pf, Rf = record(jnp.asarray(Xf)); Rf = {k: np.asarray(v) for k, v in Rf.items()}
    for name, specs in experiments.items():
        o = run_patch(jnp.asarray(X), [(s, h, pos(q), jnp.asarray(Rf[s])) for s, h, pos in specs])
        acc[name].append((o[:, q].argmax(-1) == pf[:, q].argmax(-1))[sel])
    k = (q + 1) % Q; Xo = X.copy(); opk = (X[:, k, 0] + X[:, k, 1]) % K == 0      # negative control: flip another question
    Xo[:, k, 1] = np.where(opk, ((-X[:, k, 0]) % K + 1 + rng.integers(0, K - 1, len(X))) % K, (-X[:, k, 0]) % K)
    Po, Ro = record(jnp.asarray(Xo)); neg["flip_other_question_changes_q_argmax"].append(np.asarray(Po)[:, q].argmax(-1) != P0[:, q].argmax(-1))
    o = run_patch(jnp.asarray(X), [("resid_pre2", None, R_(q), jnp.asarray(np.asarray(Ro["resid_pre2"])))])
    neg["patch_resid_pre2@R_from_other_question_flip"].append(o[:, q].argmax(-1) != P0[:, q].argmax(-1))
res["n_scored"] = nsc; res["patch_flip_transfer"] = {k: float(np.concatenate(v).mean()) for k, v in acc.items()}
res["negative_controls_change_rate"] = {k: float(np.concatenate(v).mean()) for k, v in neg.items()}
# (2) mean ablations on natural data
abl = {}
def ablate(specs):
    o = run_patch(jnp.asarray(X), [(s, h, pos, jnp.asarray(np.broadcast_to(R0[s].mean(0, keepdims=True), R0[s].shape))) for s, h, pos in specs])
    return dict(acc=float((o.argmax(-1) == Y).mean()), agree_bayes_unique=float((o.argmax(-1) == pb.argmax(-1))[uniq(pb)].mean()))
allpos = list(range(T)); rd = list(range(36, 42)); src = list(range(36))
for l in range(3):
    for h in range(H): abl[f"head_out{l}[{h}]@all"] = ablate([(f"head_out{l}", h, allpos)])
    abl[f"attn_out{l}@all"] = ablate([(f"attn_out{l}", None, allpos)]); abl[f"mlp_out{l}@all"] = ablate([(f"mlp_out{l}", None, allpos)])
    abl[f"mlp_out{l}@readouts"] = ablate([(f"mlp_out{l}", None, rd)]); abl[f"mlp_out{l}@sources"] = ablate([(f"mlp_out{l}", None, src)])
    abl[f"attn_out{l}@readouts"] = ablate([(f"attn_out{l}", None, rd)]); abl[f"attn_out{l}@sources"] = ablate([(f"attn_out{l}", None, src)])
res["mean_ablation"] = abl
# (4) B2: gate-state decodability among answer-neutral questions
neutral = (X[:, :, 2] == X[:, :, 3]) & (X[:, :, 4] == X[:, :, 5]) & (X[:, :, 2] == X[:, :, 4])
openg = (X[:, :, 0] + X[:, :, 1]) % K == 0
dec = {}
for site, posf in (("resid_pre1", "E"), ("resid_pre2", "E"), ("resid_pre2", "R"), ("resid_pre1", "R")):
    Z, t = [], []
    for q in range(Q):
        pos = E_(q) if posf == "E" else R_(q); m = neutral[:, q]
        Z.append(R0[site][m][:, pos].reshape(m.sum(), -1)); t.append(openg[m, q])
    Z = np.concatenate(Z); t = np.concatenate(t).astype(int); n = len(t); tr = np.arange(n) < int(0.7 * n)
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
    dec[f"{site}@{posf}"] = dict(n=int(n), acc=float(LogisticRegression(max_iter=3000).fit((Z[tr] - mu) / sd, t[tr]).score((Z[~tr] - mu) / sd, t[~tr])), majority=float(max(t[~tr].mean(), 1 - t[~tr].mean())))
res["gate_state_decodable_answer_neutral"] = dec
print(json.dumps(res, indent=1)); json.dump(res, open(outp, "w"), indent=1)
