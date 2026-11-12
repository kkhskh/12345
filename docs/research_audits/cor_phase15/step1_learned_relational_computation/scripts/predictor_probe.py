"""Successful-vs-failed mechanistic predictor (frozen on development readers, tested on confirmation readers).
For each Condition-R reader, take the representation immediately before the final readout map:
  mlp: second hidden layer (post-GELU) | attn/attn2: mean-pooled group state | relnet: concat[sum_pairs phi, sum psi] (input of rho)
and fit a multinomial logistic probe (sklearn, C=1, held-out half) for the eligibility vector m = (m_0, m_1, m_2), each in {0,1,2}.
Score = mean held-out accuracy over the three groups (majority baseline reported).  Natural decisions of the eval set, question q.
Usage: [PHASE=conf] predictor_probe.py TAG [TAG ...]  -> logs/predictor_probe[_conf].json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import v8lib as V, readers as RD
from sklearn.linear_model import LogisticRegression
CONF = os.environ.get("PHASE", "dev") == "conf"
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_conf8.npz") if CONF else os.path.join(V.V7, "..", "logs", "eval_dev.npz"), allow_pickle=True))
X, q = ev["X"], ev["q"]; idx = np.arange(len(X))
f = jnp.asarray(RD.features(X)[idx, q]); m = np.minimum(V.eligibility_inv7(X)[idx, q], 2)
def hidden(kind, p, f):
    if kind == "mlp":
        h = f.reshape(len(f), -1)
        for l in p["m"][:-1]: h = jax.nn.gelu(h @ l["W"] + l["b"])
        return h
    if kind.startswith("attn"):
        x = f @ p["emb"]["W"] + p["emb"]["b"]
        for b in p["blocks"]:
            z = RD.lnorm(x); a = jax.nn.softmax(jnp.einsum("ngd,nhd->ngh", z @ b["Wq"], z @ b["Wk"]) / jnp.sqrt(x.shape[-1] * 1.0), -1)
            x = x + jnp.einsum("ngh,nhd->ngd", a, z @ b["Wv"]) @ b["Wo"]; x = x + RD.mlp_apply(b["ff"], RD.lnorm(x))
        return x.mean(-2)
    if kind == "relnet":
        pairs = [(g, h) for g in range(3) for h in range(3) if g != h]
        P = sum(RD.mlp_apply(p["phi"], jnp.concatenate([f[:, g], f[:, h]], -1)) for g, h in pairs)
        S = sum(RD.mlp_apply(p["psi"], f[:, g]) for g in range(3)); return jnp.concatenate([P, S], -1)
out = {}
for tag in sys.argv[1:]:
    kind = tag.split("_")[0]; p = jax.tree_util.tree_map(jnp.asarray, V.load_tree(os.path.join(HERE, "..", "checkpoints", "compose_conf" if CONF else "compose", tag + ".npz")))
    H = np.asarray(hidden(kind, p, f)); n = len(H) // 2; accs, maj = [], []
    for g in range(3):
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(H[:n], m[:n, g]); accs.append(clf.score(H[n:], m[n:, g]))
        maj.append(np.bincount(m[n:, g]).max() / (len(m) - n))
    cj = json.load(open(os.path.join(HERE, "..", "checkpoints", "compose_conf" if CONF else "compose", tag + ".json")))
    last = cj["evals"][str(max(int(k) for k in cj["evals"]))]["families"]["F45"]
    out[tag] = dict(probe_acc=float(np.mean(accs)), majority=float(np.mean(maj)), E_R1=last["R1"]["E"], dm_R1=last["R1"]["dmM"])
    print(tag, {k: round(v, 3) for k, v in out[tag].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "predictor_probe_conf.json" if CONF else "predictor_probe.json"), "w"), indent=1)
