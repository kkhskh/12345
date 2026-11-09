"""Q3-B3 evaluation exactly as frozen: all pairs of heads among the 8 heads of layers 1-2 (within- and cross-layer);
mean-ablation of each head alone and of each pair (all positions); unique-argmax Bayes agreement on stream 9_500_000.
Usage: gate_headpairs.py CKPT OUT.json"""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import tasks3  # noqa
import tasks_ext as TX, fwd3 as F3, ckpt as C
ck, outp = sys.argv[1], sys.argv[2]
p = C.load(ck); p.pop("WX", None); p = jax.tree_util.tree_map(jnp.asarray, p)
d = TX.sample_gate(3000, np.random.default_rng(9_500_000)); X = jnp.asarray(d["X"]); pb = TX.bayes_gate(d["X"]); s_ = np.sort(pb, -1); uq = s_[..., -1] - s_[..., -2] > 1e-6
rec = {}
def hk0(n, a):
    if n.startswith("head_out"): rec[n] = a
    return a
F3.forward(p, X, hk0); means = {k: jnp.mean(v, 0, keepdims=True) for k, v in rec.items()}
def agree(heads):
    def hk(n, a):
        for (l, h) in heads:
            if n == f"head_out{l}": a = a.at[:, h].set(jnp.broadcast_to(means[n][:, h], a[:, h].shape))
        return a
    o = np.asarray(F3.forward(p, X, hk)["answer"]).argmax(-1); return float((o == pb.argmax(-1))[uq].mean())
H = [(l, h) for l in (1, 2) for h in range(4)]; single = {f"L{l}h{h}": agree([(l, h)]) for l, h in H}; pairs = {}
for a, b in itertools.combinations(H, 2): pairs[f"L{a[0]}h{a[1]}+L{b[0]}h{b[1]}"] = agree([a, b])
qual = [k for k, v in pairs.items() if v <= 1 - 0.05 and all(single[x] >= 1 - 0.01 for x in k.split("+"))]
res = dict(single=single, pairs=pairs, Q3B3_qualifying_pairs=qual, Q3B3_holds=bool(qual)); print(json.dumps(res, indent=0)); json.dump(res, open(outp, "w"), indent=1)
