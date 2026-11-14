"""A3 / C4: probe-composition sufficiency test.  Native-state-only at inference: linear probes read (i) each group's count n_g from the
readout token q and (ii) each source's own report at q from its source token; the decoded values are plugged into the exact closed form
(Theorem 1, Investigation VII).  This separates INFORMATION ACCESS (are the ingredients linearly recoverable from native states?) from
COMBINATION (no learned combiner is involved).  Probes are fit on 20,000 fresh generator episodes (all 6 q).  Evaluated with the standard
families.  Variants: UPTO in {2, 3}; 'hard' (argmax decoding) and 'soft' (posterior-weighted over decoded count distributions).
Usage: decode_compose.py PARENT UPTO [EVAL=dev|path] -> logs/decode_compose_<PARENT>_u<UPTO>[_<eval>].json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, native as NV
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(NV.ROOT, "step1_learned_relational_computation", "src")); import v8lib as V8
parent, upto = sys.argv[1], int(sys.argv[2]); evname = sys.argv[3] if len(sys.argv) > 3 else "dev"
pf = NV.load_parent(parent); rng = np.random.default_rng(33_000_000)
Xtr = NV.Bm.sample_fast(NV.ENV, 20000, rng)["X"]; Ztr = NV.states_np(pf, Xtr, upto)
ntr = V8.E.loo(Xtr)                                                    # (n,6,3)
cnt = [[LogisticRegression(C=1.0, max_iter=3000).fit(Ztr[:, 36 + q], ntr[:, q, g]) for g in range(3)] for q in range(6)]   # per position, per group
vote = [LogisticRegression(C=1.0, max_iter=3000).fit(Ztr[:, [q * 6 + s for q in range(6)]].reshape(-1, 64), Xtr[:, :, s].reshape(-1)) for s in range(6)]
def decoded(X):
    Z = NV.states_np(pf, X, upto); n = len(X)
    nh = np.stack([np.stack([cnt[q][g].predict(Z[:, 36 + q]) for g in range(3)], -1) for q in range(6)], 1)
    vh = np.stack([vote[s].predict(Z[:, [q * 6 + s for q in range(6)]].reshape(-1, 64)).reshape(n, 6) for s in range(6)], -1)
    return nh, vh
def bayes_from(nh, vh):
    """exact posterior from (counts, current-question reports) -- Investigation VII closed form (E.bayes_from)."""
    return V8.E.bayes_from(vh, nh)
FALLBACK = []
def fn(X):
    nh, vh = decoded(X); P = bayes_from(nh, vh); bad = ~np.isfinite(P).all(-1) | (P.sum(-1) <= 0)
    FALLBACK.append(float(bad.mean())); P = np.where(bad[..., None], 1.0 / 3, P); return np.log(np.nan_to_num(P) + 1e-9)
ev = NV.load_eval() if evname == "dev" else NV.load_eval(evname)
nh, vh = decoded(ev["X"]); ntrue = V8.E.loo(ev["X"])
acc = dict(count_acc=float((nh == ntrue).mean()), count_4v5_confusion=float(((ntrue == 5) & (nh != 5)).sum() / max((ntrue == 5).sum(), 1)),
           count_false_perfect=float(((ntrue != 5) & (nh == 5)).sum() / max((ntrue != 5).sum(), 1)), vote_acc=float((vh == ev["X"]).mean()))
r = NV.ER.run(fn, ev); s = NV.summarize(r)
# check: true ingredients through the same function = exact Bayes
r0 = NV.ER.run(lambda X: np.log(bayes_from(V8.E.loo(X), X) + 1e-9), ev); s0 = NV.summarize(r0)
out = dict(parent=parent, upto=upto, fallback_frac=FALLBACK, decode=acc, composed=s, true_ingredients_check=s0)
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(os.path.join(HERE, "..", "logs", f"decode_compose_{parent}_u{upto}" + ("" if evname == "dev" else "_" + os.path.basename(evname).split(".")[0]) + ".json"), "w"), indent=1, default=float)
