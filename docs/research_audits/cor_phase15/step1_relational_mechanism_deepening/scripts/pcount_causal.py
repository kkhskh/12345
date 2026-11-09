"""Workstream A3-A5: causal identification of the transferred count computation in the TRANSFERRED T4 models
(frozen layers 0-1 from pretraining, layer 2 trained on T4).  Usage: pcount_causal.py ARM CKPT OUT.json
Site: input of layer 2 (resid_pre2) = output of the frozen, transferred layers.  Sites compared: readout tokens (36-41)
and source tokens (0-35).
(1) Count-subspace identification WITHOUT outcome information: k-dim contrast basis of activation differences between
    base and 'cond' counterfactual episodes (X_q kept, other questions resampled on the generator) at each site.
(2) Interventions on natural T4 data (stream 9_900_000), each reported as accuracy, drop fraction of the relational gain,
    and KL(own-question Bayes || model) (direction of the damage):
    count-subspace mean-ablation | random k-dim subspace | variance-matched control (top-k PCs orthogonal to the count basis).
(3) Interchange (cond pairs; IIA, KL recovered) of the count subspace at each site, and of the full site.
(4) Count-preserving source change (all six reports of every other question cyclically relabelled: agreement pattern and
    hence all counts preserved, contents changed): model change rate, and the same patch -> selectivity.
(5) Abstract-variable interchange: a linear ENCODER E (one-hot leave-q counts of the 3 groups -> activation) is fitted by
    least squares on natural data at the site; intervention a' = a + E(n_cf) - E(n_base), with n_cf from the counterfactual
    episode.  IIA against the counterfactual run.  Same procedure in every model, so the abstract count variable is the
    shared coordinate system."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import tasks3  # noqa
import fwd3 as F3, ckpt as C, gen_family as G, analysis as A, bayes as B
arm, ck, outp = sys.argv[1], sys.argv[2], sys.argv[3]; Q, T, K = 6, 43, 3
p = jax.tree_util.tree_map(jnp.asarray, C.load(ck)); env = B.Env("T4", K=3, Q=6); f4 = G.Fam(); d = p["Eread"].shape[0]
@jax.jit
def acts(X):
    rec = {}
    def hk(n, a):
        if n == "resid_pre2": rec["r"] = a
        return a
    o = F3.forward(p, X, hk); return rec["r"], jax.nn.softmax(o["answer"], -1)
@jax.jit
def patched(X, tgt, P, mask):
    def hk(n, a):
        return a + ((tgt - a) @ P) * mask[None, :, None] if n == "resid_pre2" else a
    return jax.nn.softmax(F3.forward(p, X, hk)["answer"], -1)
@jax.jit
def added(X, delta, mask):
    def hk(n, a):
        return a + delta * mask[None, :, None] if n == "resid_pre2" else a
    return jax.nn.softmax(F3.forward(p, X, hk)["answer"], -1)
SITES = {"readout": list(range(36, 42)), "source": list(range(36))}
def mk(pos): m = np.zeros(T, np.float32); m[pos] = 1; return jnp.asarray(m)
# (1) bases from cond contrasts (stream 9_890_000)
Di = G.sample(f4, 800, np.random.default_rng(9_890_000))["X"]; ai = np.asarray(acts(jnp.asarray(Di))[0])
dk = {s: [] for s in SITES}
for q in range(Q):
    Xp = A.pairs(env, Di, q, "cond", np.random.default_rng(9_890_010 + q)); ap = np.asarray(acts(jnp.asarray(Xp))[0]); dd = ap - ai
    dk["readout"].append(dd[:, 36 + q]); dk["source"].append(dd[:, [t for t in range(36) if t // 6 != q]].reshape(-1, d))
KD = int(os.environ.get("KDIM", "4"))
bases = {s: A.contrast_basis(np.concatenate(v), KD) for s, v in dk.items()}
D = G.sample(f4, 2000, np.random.default_rng(9_900_000)); X, Y = D["X"], D["Y"]; a0, P0 = map(np.asarray, acts(jnp.asarray(X)))
post = G.posterior(f4, X); own = post["own_y"]; acc_clean = float((P0.argmax(-1) == Y).mean()); acc_own = float((own.argmax(-1) == Y).mean())
res = dict(arm=arm, ckpt=ck, kdim=KD, acc_clean=acc_clean, acc_own_bayes=acc_own, acc_bayes=float((post["post_y"].argmax(-1) == Y).mean()))
def summary(o):
    a = float((o.argmax(-1) == Y).mean())
    return dict(acc=a, drop_frac=float((acc_clean - a) / max(acc_clean - acc_own, 1e-9)), KL_own_to_model=float((own * (np.log(own + 1e-12) - np.log(o + 1e-12))).sum(-1).mean()))
res["KL_own_to_clean"] = float((own * (np.log(own + 1e-12) - np.log(P0 + 1e-12))).sum(-1).mean())
rng = np.random.default_rng(3); res["ablation"] = {}; res["subspace_variance_share"] = {}
for s, pos in SITES.items():
    U = bases[s]; Pc = A.proj(U).astype(np.float32); mean = np.broadcast_to(a0[:, pos].reshape(-1, d).mean(0), a0.shape)
    Ur = A.random_basis(d, KD, rng); Pr = A.proj(Ur).astype(np.float32)
    Z = a0[:, pos].reshape(-1, d); Zc = Z - Z.mean(0); Zo = Zc - Zc @ Pc; _, _, Vt = np.linalg.svd(Zo, full_matrices=False); Pv = A.proj(Vt[:KD].T).astype(np.float32)
    tot = (Zc ** 2).sum(); res["subspace_variance_share"][s] = dict(count=float(((Zc @ Pc) ** 2).sum() / tot), random=float(((Zc @ Pr) ** 2).sum() / tot), varmatched=float(((Zc @ Pv) ** 2).sum() / tot))
    for nm, Pm in (("count_subspace", Pc), ("random", Pr), ("variance_matched", Pv)):
        o = np.asarray(patched(jnp.asarray(X), jnp.asarray(mean, jnp.float32), jnp.asarray(Pm), mk(pos))); res["ablation"][f"{s}|{nm}"] = summary(o)
print(json.dumps(res["ablation"], indent=0), flush=True)
# (3) interchange and (5) abstract-variable interchange
def loo_counts(Z):
    Ag = np.stack([(Z[:, :, a] == Z[:, :, b]) for a, b in G.GROUPS], -1).astype(int); return Ag.sum(1, keepdims=True) - Ag   # N,Q,3
def onehot(n): return np.eye(6)[n].reshape(n.shape[:-1] + (18,))                                  # (...,3) -> (...,18)
Dn = G.sample(f4, 3000, np.random.default_rng(9_910_000))["X"]; an = np.asarray(acts(jnp.asarray(Dn))[0]); nn = loo_counts(Dn)
enc = {}
# readout q: encode all three groups' leave-q counts; source token (q,s): encode the leave-q counts (of all groups) too
Fr = onehot(nn).reshape(-1, 18); Ar = an[:, 36:42].reshape(-1, d); Wr = np.linalg.lstsq(np.c_[np.ones(len(Fr)), Fr], Ar, rcond=None)[0]
Fs = np.repeat(onehot(nn), 6, axis=1).reshape(-1, 18); As = an[:, :36].reshape(-1, d); Ws = np.linalg.lstsq(np.c_[np.ones(len(Fs)), Fs], As, rcond=None)[0]
enc["readout"], enc["source"] = Wr[1:], Ws[1:]
res["interchange"] = {}
for kind in ("cond", "relabel_other_questions"):
    PE, PEp, PP = {}, {}, {}
    for q in range(Q):
        if kind == "cond": Xp = A.pairs(env, X, q, "cond", np.random.default_rng(9_900_100 + q))
        else:
            Xp = X.copy(); oth = [k for k in range(Q) if k != q]; Xp[:, oth] = (X[:, oth] + 1 + (np.arange(len(X)) % 2)[:, None, None]) % K
        ap, pp = map(np.asarray, acts(jnp.asarray(Xp))); n0, n1 = loo_counts(X)[:, q], loo_counts(Xp)[:, q]
        conds = {}
        for s, pos in SITES.items():
            Pc = A.proj(bases[s]).astype(np.float32)
            conds[f"{s}|count_subspace"] = np.asarray(patched(jnp.asarray(X), jnp.asarray(ap), jnp.asarray(Pc), mk(pos)))
            conds[f"{s}|full"] = np.asarray(patched(jnp.asarray(X), jnp.asarray(ap), jnp.eye(d, dtype=jnp.float32), mk(pos)))
            dl = np.zeros((len(X), T, d), np.float32)
            if s == "readout": dl[:, 36 + q] = (onehot(n1) - onehot(n0)) @ enc[s]
            else:   # every source token (k, s) gets the change of ITS OWN question's leave-k counts
                N0, N1 = loo_counts(X), loo_counts(Xp)
                for kq in range(Q): dl[:, 6 * kq:6 * kq + 6] = ((onehot(N1[:, kq]) - onehot(N0[:, kq])) @ enc[s])[:, None]
            conds[f"{s}|abstract_count_variable"] = np.asarray(added(jnp.asarray(X), jnp.asarray(dl), mk(pos)))
        for nm, o in conds.items():
            PE.setdefault(nm, []).append(P0[:, q]); PEp.setdefault(nm, []).append(pp[:, q]); PP.setdefault(nm, []).append(o[:, q])
        if kind != "cond": res.setdefault("relabel_model_change_rate", []).append(float((pp[:, q].argmax(-1) != P0[:, q].argmax(-1)).mean()))
    for nm in PE:
        m = A.interchange_metrics(*map(np.concatenate, (PE[nm], PEp[nm], PP[nm])))
        res["interchange"][f"{kind}|{nm}"] = dict(iia=m["iia"], n_inf=m["n_inf"], kl_rec=m["kl_recovered"], change_rate=m["change_rate"])
res["relabel_model_change_rate"] = float(np.mean(res.get("relabel_model_change_rate", [0])))
print(json.dumps(res["interchange"], indent=0)); json.dump(res, open(outp, "w"), indent=1, default=float)
