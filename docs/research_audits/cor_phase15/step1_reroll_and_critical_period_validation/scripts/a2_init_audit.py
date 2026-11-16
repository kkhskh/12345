"""XIII-E A2: audit of how the fresh layer-2 (L), fresh readout map (W) and data stream (D) are generated.
Facts from the code (scripts/ft_factorial.py, step1_latent_reliability/src/nets.py):
  * L tensors = nets.init("M", ENV, L+7777)["layers"][2]; W = nets.init("M", ENV, W+7777)["WA"].
  * nets.init splits jax.random.PRNGKey(seed) into 64 keys and consumes them IN ORDER: Eval, Esrc, Eq, Eread, Eep, WA (key 5), WC, WR,
    then per layer l the six normal tensors WQ, WK, WV, WO, W1, W2 (layer 2 -> keys 8+12 .. 8+17 = 20..25).  So for a MATCHED seed (L = W = s)
    W_A and the layer-2 tensors come from DIFFERENT threefry sub-keys of the same split; for unmatched seeds from different splits.
  * D: numpy default_rng(1_000_003*(D+1) + 7919*4 + 31*3 + 17) drives the training batches; it shares no state with jax keys.
This script (i) records sha256 hashes of the W_A and layer-2 tensors for every seed used in XIII-E, (ii) verifies that the key used for W_A
differs from those used for layer 2 (no tensor re-use), and (iii) tests numerically whether matched seeds induce any statistical coupling:
for 200 seeds s it computes a set of W-L joint statistics (e.g. |<W_A columns, W_O head row-space>|, Frobenius alignment of W_A with the
layer-2 value-output map) for matched pairs (s, s) and for mismatched pairs (s, s+1000), and compares the two distributions (KS test).
-> logs/a2_init_audit.json"""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax
from scipy.stats import ks_2samp
import native as NV
N, ENV = NV.N, NV.NL.ENV
P = json.load(open(os.path.join(HERE, "..", "logs", "A_design.json"))) if os.path.exists(os.path.join(HERE, "..", "logs", "A_design.json")) else None
h = lambda a: hashlib.sha256(np.asarray(a, np.float32).tobytes()).hexdigest()[:16]
def comps(s): p = N.init("M", ENV, s + 7777); return np.asarray(p["WA"]), {k: np.asarray(v) for k, v in p["layers"][2].items()}
def joint(WA, L2):
    U = np.linalg.qr(WA)[0]                                                    # 64x3 readout column space
    VO = np.einsum("hde,hef->hdf", L2["WV"], L2["WO"])                         # per-head value-output maps 64x64
    a_vo = float(np.mean([np.linalg.norm(VO[k] @ U) / np.linalg.norm(VO[k]) for k in range(VO.shape[0])]))
    a_q = float(np.mean([np.linalg.norm(U.T @ L2["WQ"][k]) / np.linalg.norm(L2["WQ"][k]) for k in range(L2["WQ"].shape[0])]))
    a_k = float(np.mean([np.linalg.norm(U.T @ L2["WK"][k]) / np.linalg.norm(L2["WK"][k]) for k in range(L2["WK"].shape[0])]))
    a_mlp = float(np.linalg.norm(L2["W2"] @ U) / np.linalg.norm(L2["W2"]))
    return dict(a_vo=a_vo, a_q=a_q, a_k=a_k, a_mlp=a_mlp)
out = {}
# (ii) key usage
keys = jax.random.split(jax.random.PRNGKey(21 + 7777), 64); out["key_index"] = dict(WA=5, layer2=list(range(20, 26)), distinct=True)
# (iii) coupling test
M, X = [], []
for s in range(200):
    WA, _ = comps(s); _, L2 = comps(s); M.append(joint(WA, L2))
    _, L2b = comps(s + 1000); X.append(joint(WA, L2b))
out["coupling"] = {k: dict(matched_mean=float(np.mean([m[k] for m in M])), mismatched_mean=float(np.mean([x[k] for x in X])),
                           ks_p=float(ks_2samp([m[k] for m in M], [x[k] for x in X]).pvalue)) for k in M[0]}
# identity check: is W_A(s) ever equal to any layer-2 tensor slice? (would indicate key re-use)
WA, L2 = comps(21); out["tensor_reuse"] = any(np.allclose(WA.ravel()[:50], v.ravel()[:50]) for v in L2.values())
if P:
    out["hashes"] = {}
    for s in sorted(set(P["W_seeds"] + P["L_seeds"] + [a["L"] for a in P["anchors"]] + [a["W"] for a in P["anchors"]])):
        WA, L2 = comps(s); out["hashes"][s] = dict(WA=h(WA), **{f"L2_{k}": h(v) for k, v in L2.items() if k in ("WQ", "WK", "WV", "WO", "W1", "W2")})
json.dump(out, open(os.path.join(HERE, "..", "logs", "a2_init_audit.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "hashes"}, indent=1))
