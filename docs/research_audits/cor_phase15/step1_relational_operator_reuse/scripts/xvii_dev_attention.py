"""XVII DEVELOPMENT: per-slot attention routing at one checkpoint.  Usage: xvii_dev_attention.py COND CKPT OUT.json
For each layer l, head h and slot q: mean attention (i) from slot q's gate tokens (0..a-1) to the OTHER gate tokens of slot q,
(ii) from slot q's evidence tokens (3,4,5) to slot q's gate tokens, (iii) from slot q's readout to slot q's evidence tokens,
(iv) from slot q's tokens to tokens of other slots (cross-slot leakage)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_cross_family_relational_foundations", "src")); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_relational_mechanism_deepening", "src"))
import numpy as np, jax, jax.numpy as jnp, gatelib as GL, fwd3 as F3, xvitask as T, xvidiag as XD
cond, ck, outp = sys.argv[1], sys.argv[2], sys.argv[3]; a = XD.parse(cond)[0]
X = T.sample(500, np.random.default_rng(5), *XD.parse(cond))["X"]; p = GL.load(ck); rec = {}
def hk(n, v):
    if n.startswith("pattern"): rec[n] = np.asarray(v)
    return v
F3.forward(p, jnp.asarray(X), hk); out = {}
for l in range(3):
    A = rec[f"pattern{l}"].mean(0)                       # H,T,T
    for q in range(6):
        G = [6 * q + s for s in range(a)]; E = [6 * q + s for s in (3, 4, 5)]; R = 36 + q; own = list(range(6 * q, 6 * q + 6)) + [R]
        other = [t for t in range(43) if t not in own]
        gg = [float(np.mean([A[h, g, [x for x in G if x != g]].sum() for g in G])) for h in range(A.shape[0])]
        eg = [float(A[h][np.ix_(E, G)].sum(1).mean()) for h in range(A.shape[0])]
        re_ = [float(A[h, R, E].sum()) for h in range(A.shape[0])]
        xs = [float(A[h][np.ix_(own, other)].sum(1).mean()) for h in range(A.shape[0])]
        out[f"L{l}_q{q}"] = dict(gate_to_gate=gg, evid_to_gate=eg, readout_to_evid=re_, cross_slot=xs)
json.dump(out, open(outp, "w"), indent=1)
for l in range(3):
    for key in ("gate_to_gate", "evid_to_gate", "readout_to_evid", "cross_slot"):
        print(f"L{l} {key:16s}", " ".join(f"q{q}:" + ",".join(f"{v:.2f}" for v in out[f'L{l}_q{q}'][key]) for q in range(6)))
