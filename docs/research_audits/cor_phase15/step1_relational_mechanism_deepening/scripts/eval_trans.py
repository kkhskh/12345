"""TRANS evaluation (Workstream C2-C4, C6).  Usage: eval_trans.py CKPT NAME PAIRS   (PAIRS = all | holdout | train)
Episodes: 6000 from the chosen (sigma, tau) pair set (stream 9_950_000).
 (a) accuracy and cross-entropy by question type, vs exact Bayes (all 36 pairs) and the no-tau-evidence reference.
 (b) Q3-C criteria: C-only accuracy; drop when the C reports of all BC questions are masked; drop when the A reports of all
     AB questions are masked.
 (c) representation: decodability of sigma, tau and the composite tau.sigma from [readout-mean, episode token] of the final
     layer (logistic and a small MLP probe; 4000 train / 2000 test episodes); chance 1/6.
 (d) counterfactual map interventions (valid episodes): replace sigma by sigma' != sigma in the AB questions only (B
     reports regenerated with the same noise draws), keeping BC/C-only questions: the world is (sigma', tau) and the
     C-only Bayes answer changes; likewise replace tau by tau' in the C reports of BC questions.  Score: on C-only questions
     whose (unique) Bayes argmax changes, fraction where the model's argmax follows the new Bayes answer; plus the
     fraction where the model's argmax changes at all.
Writes logs/trans/eval_NAME.json."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import tasks3 as T3, tasks_ext as TX, fwd3 as F3, ckpt as C
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
ck, name, pairs = sys.argv[1], sys.argv[2], sys.argv[3]; K, Q = 3, 6; MASK = TX.MASK
PS = {"all": T3.ALL_PAIRS, "holdout": T3.HOLDOUT_PAIRS, "train": T3.TRAIN_PAIRS}[pairs]
p = C.load(ck); p.pop("WX", None); p = jax.tree_util.tree_map(jnp.asarray, p)
@jax.jit
def run(X):
    rec = {}
    def hk(n, a):
        if n == "final": rec["f"] = a
        return a
    o = F3.forward(p, X, hk); f = rec["f"]; return jax.nn.softmax(o["answer"], -1), jnp.concatenate([f[:, 36:42].mean(1), f[:, 42]], -1)
def apply(X):
    o = [run(jnp.asarray(X[i:i + 2000])) for i in range(0, len(X), 2000)]
    return np.concatenate([np.asarray(a) for a, _ in o]), np.concatenate([np.asarray(b) for _, b in o])
d = T3.sample_trans_pairs(6000, np.random.default_rng(9_950_000), PS); X, Y, typ = d["X"], d["Y"], d["typ"]
P, Z = apply(X); pb = TX.bayes_trans(X)["post_y"]; p1 = TX.bayes_trans_onehop(X)["post_y"]
res = dict(ckpt=ck, pairs=pairs)
ce = lambda Pm, m: float(-np.log(np.take_along_axis(Pm, Y[..., None], -1)[..., 0] + 1e-12)[m].mean())
for t, nm in enumerate(["AB", "BC", "C_only"]):
    m = typ == t
    res[nm] = dict(acc=float((P.argmax(-1) == Y)[m].mean()), bayes=float((pb.argmax(-1) == Y)[m].mean()), no_tau=float((p1.argmax(-1) == Y)[m].mean()),
                   ce=ce(P, m), ce_bayes=ce(pb, m), ce_no_tau=ce(p1, m))
mC = np.where(((X[:, :, 0] == MASK) & (X[:, :, 2] != MASK))[..., None] & (np.arange(6) >= 4)[None, None], MASK, X)
mA = np.where((X[:, :, 4] == MASK)[..., None] & (np.arange(6) < 2)[None, None], MASK, X)
c = typ == 2
res["Q3C"] = dict(C_only=res["C_only"]["acc"], C_only_mask_tau_evidence=float((apply(mC)[0].argmax(-1) == Y)[c].mean()),
                  C_only_mask_sigma_evidence=float((apply(mA)[0].argmax(-1) == Y)[c].mean()))
res["Q3C"]["learned"] = bool(res["Q3C"]["C_only"] >= 0.42 and res["Q3C"]["C_only"] - res["Q3C"]["C_only_mask_tau_evidence"] >= 0.05 and res["Q3C"]["C_only"] - res["Q3C"]["C_only_mask_sigma_evidence"] >= 0.05)
comp = T3.composite_index(d["sigma"], d["tau"]); tr = np.arange(6000) < 4000; mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6; Zs = (Z - mu) / sd
res["probes"] = {}
for nm, t in (("sigma", d["sigma"]), ("tau", d["tau"]), ("composite", comp)):
    lin = LogisticRegression(max_iter=3000).fit(Zs[tr], t[tr]).score(Zs[~tr], t[~tr])
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, random_state=0).fit(Zs[tr], t[tr]).score(Zs[~tr], t[~tr])
    res["probes"][nm] = dict(linear=float(lin), mlp=float(mlp))
# (d) counterfactual map interventions
rng = np.random.default_rng(9_950_500); si, ti = d["sigma"].copy(), d["tau"].copy()
def regen(Xb, which):
    Xn = Xb.copy(); n = len(Xn); new = (si + rng.integers(1, 6, n)) % 6 if which == "sigma" else (ti + rng.integers(1, 6, n)) % 6
    Sm = np.stack([T3.PERMS[i] for i in (new if which == "sigma" else si)]); Tm = np.stack([T3.PERMS[i] for i in (ti if which == "sigma" else new)])
    if which == "sigma":   # AB questions: B := sigma'(A-true Y) -> regenerate B from the A-question's Y with the original noise pattern
        vB = np.take_along_axis(Sm, Y, 1); ab = typ == 0
        for s in (2, 3):
            keep_noise = Xb[:, :, s] != np.take_along_axis(np.stack([T3.PERMS[i] for i in si]), Y, 1)   # where the original report was an error
            Xn[:, :, s] = np.where(ab, np.where(keep_noise, (vB + 1 + rng.integers(0, K - 1, vB.shape)) % K, vB), Xn[:, :, s])
    else:                  # BC questions: C := tau'(B-true value)
        vB = np.take_along_axis(np.stack([T3.PERMS[i] for i in si]), Y, 1); vC = np.take_along_axis(Tm, vB, 1); bc = typ == 1
        vC_old = np.take_along_axis(np.stack([T3.PERMS[i] for i in ti]), vB, 1)
        for s in (4, 5):
            keep_noise = Xb[:, :, s] != vC_old
            Xn[:, :, s] = np.where(bc, np.where(keep_noise, (vC + 1 + rng.integers(0, K - 1, vC.shape)) % K, vC), Xn[:, :, s])
    return Xn
def uniq(Pm): s = np.sort(Pm, -1); return s[..., -1] - s[..., -2] > 1e-6
res["map_interventions"] = {}
for which in ("sigma", "tau"):
    Xn = regen(X, which); Pn = apply(Xn)[0]; pbn = TX.bayes_trans(Xn)["post_y"]
    sel = c & (pb.argmax(-1) != pbn.argmax(-1)) & uniq(pb) & uniq(pbn)
    res["map_interventions"][which] = dict(n=int(sel.sum()), follows_new_bayes=float((Pn.argmax(-1) == pbn.argmax(-1))[sel].mean()) if sel.any() else None,
                                           model_changes=float((Pn.argmax(-1) != P.argmax(-1))[sel].mean()) if sel.any() else None)
print(json.dumps(res, indent=1)); os.makedirs(os.path.join(HERE, "..", "logs", "trans"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "..", "logs", "trans", f"eval_{name}.json"), "w"), indent=1)
