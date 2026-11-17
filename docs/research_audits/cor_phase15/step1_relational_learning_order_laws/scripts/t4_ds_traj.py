"""XV Workstream E diagnostics for DeepSets T4 snapshots.  Usage: t4_ds_traj.py RUNDIR SEED STREAM N OUT.json
References (exact, bayes.episode_posterior): Bayes P(Y_q | episode) and the own-question rule P(Y_q | X_q only) (no cross-question
information: the T4 'source stage' reference).  Relational edit: at a random other question q' != q, one source of a random pair is set
equal to its partner (if they disagree) or to a different value (if they agree); edits whose Bayes posterior is undefined are dropped.
G4 = slope of the model's change of centred log-probabilities for question q on the Bayes change (own-question rule: 0 exactly; Bayes 1)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_latent_reliability", "src"))
import numpy as np, jax, jax.numpy as jnp, nets as N, bayes as B
rd, seed, stream, n, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
env = B.Env("T4", K=3, Q=6); rng = np.random.default_rng(stream); d = B.sample_fast(env, n, rng); X, Y = d["X"], d["Y"]; I = np.arange(n)
q = rng.integers(0, 6, n); qq = (q + rng.integers(1, 6, n)) % 6; pr = rng.integers(0, 3, n); Xe = X.copy()
a0, a1 = X[I, qq, 2 * pr], X[I, qq, 2 * pr + 1]
Xe[I, qq, 2 * pr + 1] = np.where(a0 != a1, a0, (a0 + rng.integers(1, 3, n)) % 3)
P0 = B.episode_posterior(env, X); P1 = B.episode_posterior(env, Xe); PB, PBe = P0["post_y"][I, q], P1["post_y"][I, q]; own = P0["own_y"][I, q]
ok = np.isfinite(PBe).all(-1) & (PBe.sum(-1) > 0.5)
def cen(P): L = np.log(P + 1e-12); return L - L.mean(-1, keepdims=True)
dB = cen(PBe) - cen(PB)
tmpl = N.init_base("deepsets", "M", env, seed); tree = jax.tree_util.tree_structure(tmpl); fwd = N.make_forward("deepsets", 3)
def probs(p, Xin): return np.asarray(jax.nn.softmax(fwd(p, jnp.asarray(Xin))["answer"], -1))[I, q]
steps = sorted(int(f[4:-4]) for f in os.listdir(rd) if f.startswith("step") and f.endswith(".npz")); res = dict(run=rd, stream=stream, n=n, n_edits_valid=int(ok.sum()), traj=[])
kl = lambda A, Bq: float((A * (np.log(A + 1e-12) - np.log(Bq + 1e-12))).sum(-1).mean())
res["refs"] = dict(bayes_acc=float((PB.argmax(-1) == Y[I, q]).mean()), own_acc=float((own.argmax(-1) == Y[I, q]).mean()), own_KL=kl(PB, own))
for k in steps:
    z = np.load(os.path.join(rd, f"step{k}.npz")); p = jax.tree_util.tree_unflatten(tree, [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))])
    Pm, Pme = probs(p, X), probs(p, Xe); dm = cen(Pme) - cen(Pm)
    res["traj"].append(dict(step=k, acc=float((Pm.argmax(-1) == Y[I, q]).mean()), KL_bayes=kl(PB, Pm), KL_own=kl(own, Pm),
                            G4=float((dm[ok] * dB[ok]).sum() / (dB[ok] ** 2).sum())))
json.dump(res, open(outp, "w"), indent=1)
