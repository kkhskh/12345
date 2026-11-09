import os, sys, json
B0 = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_mechanism_deepening"
sys.path.insert(0, B0 + "/src")
import numpy as np, jax, jax.numpy as jnp
import tasks3  # noqa
import tasks_ext as TX, fwd3 as F3, ckpt as C
from sklearn.linear_model import LogisticRegression
ck, seed = sys.argv[1], int(sys.argv[2]); K, Q, T = 3, 6, 43
p = C.load(os.path.join(B0, ck)); p.pop("WX", None); p = jax.tree_util.tree_map(jnp.asarray, p)
rng = np.random.default_rng(9_500_000); d = TX.sample_gate(3000, rng); X = d["X"]; Y = d["Y"]; pb = TX.bayes_gate(X)
def uniq(P): s = np.sort(P, -1); return s[..., -1] - s[..., -2] > 1e-6
rec = {}
def hk0(n, a):
    if n in ("head_out1", "head_out2", "attn_out2", "resid_pre1", "resid_pre2", "pattern1", "pattern2"): rec[n] = np.asarray(a)
    return a
F3.forward(p, jnp.asarray(X), hk0)
res = {}
# gate-blind floor: Bayes with gate marginalised (prior 1/2)
Xo = X.copy(); Xo[:, :, 1] = (-X[:, :, 0]) % K; Xc = X.copy(); Xc[:, :, 1] = (1 - X[:, :, 0]) % K
pm = 0.5 * TX.bayes_gate(Xo) + 0.5 * TX.bayes_gate(Xc); u = uniq(pb)
res["gate_blind_floor_agree_bayes_unique"] = float((pm.argmax(-1) == pb.argmax(-1))[u].mean())
ABL = {0: {"L1h2+L2h0": [(1, 2), (2, 0)], "L1h2": [(1, 2)], "L2h0": [(2, 0)]},
       1: {"L2h1+L2h3": [(2, 1), (2, 3)], "L2h1": [(2, 1)], "L2h3": [(2, 3)], "L2all": [(2, h) for h in range(4)]},
       2: {"L2h1": [(2, 1)], "L1h3": [(1, 3)]}}[seed]
perm = np.random.default_rng(1).permutation(len(X))
def run(Xa, heads, mode, src_idx=None):
    def hk(n, a):
        for (l, h) in heads:
            if n == f"head_out{l}":
                R = rec[n]
                rep = np.broadcast_to(R[:, h].mean(0, keepdims=True), R[:, h].shape) if mode == "mean" else R[perm, h]
                a = a.at[:, h].set(jnp.asarray(rep))
        return a
    return np.asarray(F3.forward(p, jnp.asarray(Xa), hk)["answer"]).argmax(-1)
for nm, heads in ABL.items():
    for mode in ("mean", "resample"):
        o = run(X, heads, mode); res[f"{nm}|{mode}|agree_bayes_unique"] = float((o == pb.argmax(-1))[u].mean())
# gate responsiveness under ablation (mean mode): flip gate of q, both runs ablated; fraction of scored cases model follows flip
o0 = {nm: run(X, h, "mean") for nm, h in list(ABL.items()) + [("none", [])]}
fol = {nm: [] for nm in o0}
for q in range(Q):
    Xf = X.copy(); op = (X[:, q, 0] + X[:, q, 1]) % K == 0
    Xf[:, q, 1] = np.where(op, ((-X[:, q, 0]) % K + 1 + rng.integers(0, K - 1, len(X))) % K, (-X[:, q, 0]) % K)
    pf = TX.bayes_gate(Xf); sel = (pb[:, q].argmax(-1) != pf[:, q].argmax(-1)) & uniq(pb[:, q]) & uniq(pf[:, q])
    for nm, h in list(ABL.items()) + [("none", [])]:
        of = run(Xf, h, "mean"); fol[nm].append((of[:, q] == pf[:, q].argmax(-1))[sel])   # NB: 'resample' perm uses base-run recs
res["follow_flipped_bayes_under_mean_ablation_both_runs"] = {k: float(np.concatenate(v).mean()) for k, v in fol.items()}
# readout attention mass on own-question G vs E (decisive heads)
for l in (1, 2):
    A_ = rec[f"pattern{l}"]
    for h in range(4):
        g = np.mean([A_[:, h, 36 + q, 6 * q:6 * q + 2].sum(-1).mean() for q in range(Q)]); e = np.mean([A_[:, h, 36 + q, 6 * q + 2:6 * q + 6].sum(-1).mean() for q in range(Q)])
        res[f"attn_R_L{l}h{h}_G_E"] = (round(float(g), 3), round(float(e), 3))
# B2 generalisation across held-out x0 values (each held-out set = 1 open + 2 closed pairs never seen in training)
neutral = (X[:, :, 2] == X[:, :, 3]) & (X[:, :, 4] == X[:, :, 5]) & (X[:, :, 2] == X[:, :, 4]); openg = (X[:, :, 0] + X[:, :, 1]) % K == 0
for site in ("resid_pre2",):
    Z, t, x0 = [], [], []
    for q in range(Q):
        m = neutral[:, q]; Z.append(rec[site][m][:, 6 * q + 2:6 * q + 6].reshape(m.sum(), -1)); t.append(openg[m, q]); x0.append(X[m, q, 0])
    Z = np.concatenate(Z); t = np.concatenate(t).astype(int); x0 = np.concatenate(x0); accs = []
    for v in range(K):
        tr = x0 != v; mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
        accs.append(LogisticRegression(max_iter=3000).fit((Z[tr] - mu) / sd, t[tr]).score((Z[~tr] - mu) / sd, t[~tr]))
    res[f"B2_{site}@E_heldout_x0_acc"] = [round(a, 3) for a in accs]
print(ck, json.dumps(res, indent=0))
