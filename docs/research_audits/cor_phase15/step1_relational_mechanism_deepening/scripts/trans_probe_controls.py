"""Controls for the TRANS map probes (C3): the same probe pipeline (eval_trans.py, stream 9_950_000, 4000/2000 split) on
(i) a randomly initialised network's [readout-mean, episode token]; (ii) raw input features (one-hot of all 36 reports,
an upper reference for what is linearly available from the input); (iii) the exact posterior over sigma / tau (the
statistical ceiling: accuracy of the MAP map).  Writes logs/trans/probe_controls.json."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import tasks3 as T3, tasks_ext as TX, fwd3 as F3, train3 as TR
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
d = T3.sample_trans_pairs(6000, np.random.default_rng(9_950_000), T3.ALL_PAIRS); X = d["X"]; comp = T3.composite_index(d["sigma"], d["tau"])
p = jax.tree_util.tree_map(jnp.asarray, TR.init("trans", 123)); p.pop("WX", None)
@jax.jit
def run(Xb):
    rec = {}
    def hk(n, a):
        if n == "final": rec["f"] = a
        return a
    F3.forward(p, Xb, hk); f = rec["f"]; return jnp.concatenate([f[:, 36:42].mean(1), f[:, 42]], -1)
Zr = np.concatenate([np.asarray(run(jnp.asarray(X[i:i + 2000]))) for i in range(0, 6000, 2000)])
Zi = np.eye(TX.K + 1)[X].reshape(len(X), -1)
J = TX.trans_joint(X); lM = np.log(J.sum(-1)); post = np.exp(lM.sum(1) - lM.sum(1).max(-1, keepdims=True)); post /= post.sum(-1, keepdims=True)
Wp = post.reshape(-1, 6, 6); out = {"MAP_ceiling": dict(sigma=float((Wp.sum(2).argmax(-1) == d["sigma"]).mean()), tau=float((Wp.sum(1).argmax(-1) == d["tau"]).mean()))}
Wc = np.zeros((6000, 6))
for j, (a, b) in enumerate(T3.ALL_PAIRS): Wc[:, T3.composite_index([a], [b])[0]] += post[:, j]
out["MAP_ceiling"]["composite"] = float((Wc.argmax(-1) == comp).mean())
tr = np.arange(6000) < 4000
for nm, Z in (("random_network", Zr), ("raw_input_onehot", Zi)):
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6; Zs = (Z - mu) / sd; out[nm] = {}
    for tn, t in (("sigma", d["sigma"]), ("tau", d["tau"]), ("composite", comp)):
        out[nm][tn] = dict(linear=float(LogisticRegression(max_iter=3000).fit(Zs[tr], t[tr]).score(Zs[~tr], t[~tr])),
                           mlp=float(MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, random_state=0).fit(Zs[tr], t[tr]).score(Zs[~tr], t[~tr])))
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "..", "logs", "trans", "probe_controls.json"), "w"), indent=1)
