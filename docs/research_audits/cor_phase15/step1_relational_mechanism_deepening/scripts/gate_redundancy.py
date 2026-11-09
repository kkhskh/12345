"""GATE redundancy tests (Workstream B3), following gate_circuit.py: the layer-1 attention output at the readout is
sufficient (flip-transfer 0.97) but not necessary (mean-ablation harmless).  Tests whether layer-2 attention is a backup:
 (a) joint mean-ablations at the readouts (natural data): accuracy and Bayes agreement;
 (b) with attn_out1 at readout q mean-ablated in BOTH runs, flip-transfer of patching attn_out2 (or head_out2[h]) at q.
Usage: gate_redundancy.py CKPT OUT.json   Stream 9_500_000 (same as gate_circuit)."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import tasks3  # noqa
import tasks_ext as TX, fwd3 as F3, ckpt as C
ck, outp = sys.argv[1], sys.argv[2]; K, Q, T = 3, 6, 43
p = C.load(ck); p.pop("WX", None); p = jax.tree_util.tree_map(jnp.asarray, p); H = p["layers"][0]["WQ"].shape[0]
SITES = ["head_out1", "head_out2", "attn_out1", "attn_out2", "mlp_out1", "mlp_out2"]
def fwd(X, specs):
    rec = {}
    def hk(n, a):
        for site, h, pos, src in specs:
            if n != site: continue
            m = jnp.zeros(T).at[jnp.asarray(pos)].set(1.0)
            if h is None: a = a + (src - a) * m[None, :, None]
            else: a = a.at[:, h].set(a[:, h] + (src[:, h] - a[:, h]) * m[None, :, None])
        if n in SITES: rec[n] = a
        return a
    o = F3.forward(p, X, hk); return np.asarray(jax.nn.softmax(o["answer"], -1)), {k: np.asarray(v) for k, v in rec.items()}
rng = np.random.default_rng(9_500_000); d = TX.sample_gate(3000, rng); X, Y = d["X"], d["Y"]; pb = TX.bayes_gate(X)
P0, R0 = fwd(jnp.asarray(X), []); M = {k: jnp.asarray(np.broadcast_to(v.mean(0, keepdims=True), v.shape)) for k, v in R0.items()}
def uniq(P): s = np.sort(P, -1); return s[..., -1] - s[..., -2] > 1e-6
rd = list(range(36, 42)); res = dict(ckpt=ck, ablation={})
def score(o): return dict(acc=float((o.argmax(-1) == Y).mean()), agree_bayes_unique=float((o.argmax(-1) == pb.argmax(-1))[uniq(pb)].mean()))
combos = {"attn1+attn2@readouts": [("attn_out1", None, rd), ("attn_out2", None, rd)],
          "attn1+attn2+mlp2@readouts": [("attn_out1", None, rd), ("attn_out2", None, rd), ("mlp_out2", None, rd)],
          "attn1+mlp1+attn2+mlp2@readouts": [("attn_out1", None, rd), ("mlp_out1", None, rd), ("attn_out2", None, rd), ("mlp_out2", None, rd)]}
for a in range(H):
    for b in range(H): combos[f"head1[{a}]+head2[{b}]@all"] = [("head_out1", a, list(range(T))), ("head_out2", b, list(range(T)))]
for nm, sp in combos.items():
    res["ablation"][nm] = score(fwd(jnp.asarray(X), [(s, h, pos, M[s]) for s, h, pos in sp])[0])
print(json.dumps(res["ablation"], indent=0), flush=True)
tr = {}
for q in range(Q):
    Xf = X.copy(); op = (X[:, q, 0] + X[:, q, 1]) % K == 0
    Xf[:, q, 1] = np.where(op, ((-X[:, q, 0]) % K + 1 + rng.integers(0, K - 1, len(X))) % K, (-X[:, q, 0]) % K)
    pf = TX.bayes_gate(Xf); sel = (pb[:, q].argmax(-1) != pf[:, q].argmax(-1)) & uniq(pb[:, q]) & uniq(pf[:, q])
    abl = [("attn_out1", None, [36 + q], M["attn_out1"])]
    _, Rf = fwd(jnp.asarray(Xf), abl)                        # flipped run, with the same layer-1 readout ablation
    tests = {"ablate_attn1@R_only": [], "ablate_attn1@R+patch_attn2@R": [("attn_out2", None, [36 + q], jnp.asarray(Rf["attn_out2"]))]}
    for h in range(H): tests[f"ablate_attn1@R+patch_head2[{h}]@R"] = [("head_out2", h, [36 + q], jnp.asarray(Rf["head_out2"]))]
    for nm, extra in tests.items():
        o, _ = fwd(jnp.asarray(X), abl + extra); tr.setdefault(nm, []).append((o[:, q].argmax(-1) == pf[:, q].argmax(-1))[sel])
    o, _ = fwd(jnp.asarray(Xf), abl); tr.setdefault("flipped_run_with_attn1@R_ablated_follows_flipped_bayes", []).append((o[:, q].argmax(-1) == pf[:, q].argmax(-1))[sel])
res["backup_flip_transfer"] = {k: float(np.concatenate(v).mean()) for k, v in tr.items()}
print(json.dumps(res["backup_flip_transfer"], indent=0)); json.dump(res, open(outp, "w"), indent=1)
