"""Workstream C1/C2: causal decomposition of a trained native downstream (pool-based specs c1 / keep3 / deep2 / keep3x2: frozen native prefix).
For each F45 pair (before, after) in regime R1/R2/R3, with g = edited group (4 -> 5) and q = scored question:
  INTERVENTIONS on the downstream input (native states resid_pre<UPTO>):  replace a token SET by its value from the 'after' episode, keep the rest
  from 'before' (forward patching; a natural, on-support source state for every patched token):
    ro_q   readout token of question q                 ro_all  all 6 readout tokens
    g_all  all 12 source tokens of the edited group g  g_q     the 2 source tokens of g at question q
    all    every token (= the full natural response)
  report mean response dmM (oriented candidate margin change), its share of the full response, and the Bayes response.
ATTENTION ROUTING (C1): in the first downstream block, attention mass from readout token q onto each group's source tokens, regressed on the
  group's count n_g and perfectness (natural decisions), and its change under the edit.
Usage: mech_native.py PARENT SPEC [STEPS=4000] [SEED=0] [UPTO auto] -> logs/mech_native_<tag>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import native as NV
NL, N = NV.NL, NV.N
parent, spec = sys.argv[1], sys.argv[2]; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
suffix = sys.argv[5] if len(sys.argv) > 5 else ""
tag = f"{parent}_{spec}_s{STEPS}_seed{seed}{suffix}"; D = os.path.join(HERE, "..", "checkpoints", "native_train", tag)
pf = NV.load_parent(parent); UPTO = 3 if spec.startswith("keep3") else 2
fresh = lambda s: N.init("M", NL.ENV, s)
tmpl = dict(gf=pf["gf"], bf=pf["bf"], WA=fresh(seed + 7777)["WA"], new=[fresh(seed + 7777)["layers"][2]] + ([fresh(seed + 8888)["layers"][2]] if spec in ("deep2", "keep3x2") else []))
z = np.load(os.path.join(D, f"step{STEPS}.npz")); leaves = [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))]
t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tmpl), leaves)
rec = {}
def hook(n, a):
    rec[n] = a; return a
def head(x, hk=None):
    for i, lay in enumerate(t["new"]): x = NL.block(lay, x, hook=hk if i == 0 else None, l=10 + i)
    return N.ln(x, t["gf"], t["bf"])[:, 36:42] @ t["WA"]
headj = jax.jit(head)
EVP = os.environ.get("MECH_EVAL", "dev"); ev = NV.load_eval() if EVP == "dev" else NV.load_eval(EVP); X0, X1 = ev["X"], ev["X_F45"]; q = ev["q"]; idx = np.arange(len(X0))
reg = ev["reg_F45"]; vote = ev["vote_F45"].astype(int); orient = ev["orient_F45"].astype(bool)
sys.path.insert(0, os.path.join(NV.ROOT, "step1_learned_relational_computation", "src")); import v8lib as V8
# edited group = group whose count at q changed
n0, n1 = V8.E.loo(X0)[idx, q], V8.E.loo(X1)[idx, q]; g = np.argmax(np.abs(n1 - n0), 1)
Z0, Z1 = NV.states_np(pf, X0, UPTO), NV.states_np(pf, X1, UPTO)
def margin(Lg):
    c = Lg[idx, q]; c = c - c.mean(-1, keepdims=True); other = np.where(np.arange(3)[None] == vote[:, None], -np.inf, c).max(-1)
    return c[idx, vote] - other
cB0, cB1 = ev["cB_base"], ev["cB_F45"]
mB = (cB1[idx, vote] - np.where(np.arange(3)[None] == vote[:, None], -np.inf, cB1).max(-1)) - (cB0[idx, vote] - np.where(np.arange(3)[None] == vote[:, None], -np.inf, cB0).max(-1))
base = np.concatenate([np.asarray(headj(jnp.asarray(Z0[i:i + 1000]))) for i in range(0, len(Z0), 1000)]); m0 = margin(base)
def patched(sel):
    Zp = Z0.copy(); Zp[sel] = Z1[sel]; L = np.concatenate([np.asarray(headj(jnp.asarray(Zp[i:i + 1000]))) for i in range(0, len(Zp), 1000)]); return margin(L) - m0
sets = {}
def mask(fn):
    M = np.zeros(Z0.shape[:2], bool)
    for i in range(len(Z0)): M[i, fn(i)] = True
    return M
sets["all"] = mask(lambda i: np.arange(43))
sets["ro_q"] = mask(lambda i: [36 + q[i]]); sets["ro_all"] = mask(lambda i: np.arange(36, 42))
sets["g_all"] = mask(lambda i: [qq * 6 + 2 * g[i] + s for qq in range(6) for s in (0, 1)]); sets["g_q"] = mask(lambda i: [q[i] * 6 + 2 * g[i], q[i] * 6 + 2 * g[i] + 1])
sets["src_all"] = mask(lambda i: np.arange(36))
res = {}
for nm, M in sets.items():
    dm = patched(M); res[nm] = {}
    for r in ("R1", "R2", "R3"):
        s = orient & (reg == r)
        res[nm][r] = dict(n=int(s.sum()), dmM=float(dm[s].mean()), dmB=float(mB[s].mean()))
full = {r: res["all"][r]["dmM"] for r in ("R1", "R2", "R3")}
for nm in res:
    for r in ("R1", "R2", "R3"): res[nm][r]["share_of_full"] = res[nm][r]["dmM"] / full[r] if abs(full[r]) > 1e-6 else None
# ATTENTION-FREEZE intervention: run the 'after' episode with the first downstream block's attention pattern replaced by the 'before' pattern
# (values, MLP and all other computation from 'after').  Isolates the attention-routing pathway of the response.
def head_pattern(Z, pat=None):
    def hk(n, a):
        if n == "pattern10":
            if pat is None: rec["pat"] = a; return a
            return pat
        return a
    return head(Z, hk=hk)
frz = {}
for r in ("R1", "R2", "R3"):
    s_ = np.where(orient & (reg == r))[0]
    rec.clear(); _ = head_pattern(jnp.asarray(Z0[s_])); pat0 = rec["pat"]
    L1 = np.asarray(head_pattern(jnp.asarray(Z1[s_]), pat0)); Lb = np.asarray(head(jnp.asarray(Z0[s_]))); La = np.asarray(head(jnp.asarray(Z1[s_])))
    def mg(Lg, ss):
        c = Lg[np.arange(len(ss)), q[ss]]; c = c - c.mean(-1, keepdims=True); v = vote[ss]; other = np.where(np.arange(3)[None] == v[:, None], -np.inf, c).max(-1); return c[np.arange(len(ss)), v] - other
    frz[r] = dict(n=int(len(s_)), full=float((mg(La, s_) - mg(Lb, s_)).mean()), attention_frozen=float((mg(L1, s_) - mg(Lb, s_)).mean()), bayes=float(mB[s_].mean()))
# attention routing in the first downstream block
rec.clear(); _ = head(jnp.asarray(Z0[:3000]), hk=hook); A = np.asarray(rec["pattern10"]).mean(1)        # n, T, S (head-averaged)
ii = np.arange(3000); qq = q[:3000]
mass = np.stack([np.stack([A[k, 36 + qq[k], [qx * 6 + 2 * gg + s_ for qx in range(6) for s_ in (0, 1)]].sum() for k in range(3000)]) for gg in range(3)], -1)
massq = np.stack([np.stack([A[ii[k], 36 + qq[k], [qq[k] * 6 + 2 * gg, qq[k] * 6 + 2 * gg + 1]].sum() for k in range(3000)]) for gg in range(3)], -1)
nn = n0[:3000]; route = {}
for c in range(6):
    s = nn == c
    if s.sum() > 20: route[str(c)] = dict(mass_all_q=float(mass[s].mean()), mass_at_q=float(massq[s].mean()), n=int(s.sum()))
out = dict(tag=tag, patching=res, attention_freeze=frz, attention_mass_by_own_count=route, attention_mass_readout_self=float(np.mean([A[k, 36 + qq[k], 36 + qq[k]] for k in range(3000)])))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "..", "logs", f"mech_native_{tag}" + ("" if EVP == "dev" else "_" + os.path.basename(EVP).split(".")[0]) + ".json"), "w"), indent=1)
