"""Post-freeze analysis (exploratory, separate from the frozen mech_native.py): is the downstream attention routing onto a PERFECT group
conditioned on whether ANOTHER group is perfect (cross-group computation), or only on the group's own count (separable)?
Works for pool specs (c1, keep3) and fine-tuned specs (full_c1).  For natural decisions, for each group g with n_g = 5:
  mass(readout q -> g's source tokens)  split by  other_perfect = #(h != g with n_h = 5)  in {0, >=1}
and, for comparison, groups with n_g = 4 split the same way.  Also R1-relevant: edited-group mass before/after on F45 pairs by regime.
Usage: mech_routing2.py PARENT SPEC [STEPS=4000] [SEED=0] [EVAL=dev] -> logs/mech_routing2_<tag>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import native as NV
NL, N = NV.NL, NV.N
parent, spec = sys.argv[1], sys.argv[2]; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
evp = sys.argv[5] if len(sys.argv) > 5 else "dev"; suffix = "" if evp == "dev" else "_" + os.path.basename(evp).split(".")[0]
tag = f"{parent}_{spec}_s{STEPS}_seed{seed}{suffix}"; D = os.path.join(HERE, "..", "checkpoints", "native_train", tag)
pf = NV.load_parent(parent); fresh = lambda s: N.init("M", NL.ENV, s)
tmpl = dict(gf=pf["gf"], bf=pf["bf"], WA=fresh(seed + 7777)["WA"], new=[fresh(seed + 7777)["layers"][2]])
if spec == "full_c1": tmpl["emb"] = {k: pf[k] for k in NL.FROZEN_KEYS}; tmpl["pre"] = [pf["layers"][l] for l in range(2)]
z = np.load(os.path.join(D, f"step{STEPS}.npz")); t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tmpl), [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))])
def states(X):
    if spec == "full_c1":
        x = N.embed(dict(t["emb"]), X)
        for l, lay in enumerate(t["pre"]): x = NL.block(lay, x, l=l)
        return x
    return NV.states(pf, X, 3 if spec.startswith("keep3") else 2)
def pattern(X):
    rec = {}
    def hk(n, a):
        if n == "pattern10": rec["p"] = a
        return a
    NL.block(t["new"][0], states(X), hook=hk, l=10); return rec["p"].mean(1)
patj = jax.jit(pattern)
ev = NV.load_eval() if evp == "dev" else NV.load_eval(evp)
sys.path.insert(0, os.path.join(NV.ROOT, "step1_learned_relational_computation", "src")); import v8lib as V8
def group_mass(X, q):
    A = np.concatenate([np.asarray(patj(jnp.asarray(X[i:i + 500]))) for i in range(0, len(X), 500)]); n = len(X)
    cols = np.array([[qx * 6 + 2 * g + s for qx in range(6) for s in (0, 1)] for g in range(3)])          # 3,12
    rows = A[np.arange(n), 36 + q]                                                                          # n,43
    return rows[:, cols].sum(-1)                                                                            # n,3
X = ev["X"]; q = ev["q"]; idx = np.arange(len(X)); nn = V8.E.loo(X)[idx, q]; M = group_mass(X, q)
perf = nn == 5; oth = perf.sum(1, keepdims=True) - perf
res = {}
for c in (4, 5):
    for o in (0, 1):
        sel = (nn == c) & ((oth >= 1) if o else (oth == 0))
        res[f"n{c}_otherperfect{'>=1' if o else '0'}"] = dict(mass=float(M[sel].mean()) if sel.any() else None, n=int(sel.sum()))
# F45 pairs: edited group's mass before/after, by regime
X1 = ev["X_F45"]; n1 = V8.E.loo(X1)[idx, q]; g = np.argmax(np.abs(n1 - nn), 1); M1 = group_mass(X1, q)
reg = ev["reg_F45"]; orient = ev["orient_F45"].astype(bool); pairs = {}
for r in ("R1", "R2", "R3"):
    s = orient & (reg == r); pairs[r] = dict(n=int(s.sum()), edited_before=float(M[idx, g][s].mean()), edited_after=float(M1[idx, g][s].mean()))
out = dict(tag=tag, routing_by_own_and_other=res, edited_group_mass=pairs)
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "..", "logs", f"mech_routing2_{tag}.json"), "w"), indent=1)
