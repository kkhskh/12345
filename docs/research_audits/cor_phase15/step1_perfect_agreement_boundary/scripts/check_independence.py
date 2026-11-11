"""Independence of initialisations across the two curricula (mandate §7; Investigation V's seed-coupling error).
Builds the initial parameters each trainer would use (T1: nets.init("M", T1 env, seed); P-count: train_tasks.init("pcount", seed))
for T1 seeds 70-77 and P-count seeds 80-87, and counts bit-identical non-constant arrays between every cross-curriculum pair and
every within-curriculum pair.  Constant arrays (e.g. LayerNorm scales of ones) are excluded.  Writes logs/independence.json."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
for p_ in ("step1_latent_reliability/src", "step1_relational_computation_atlas/src", "step1_foundational_atlas_continuation/src"):
    sys.path.insert(0, os.path.join(ROOT, p_))
import numpy as np, jax
import nets as N, bayes as Bm, train_tasks as TT
def leaves(p): return [np.asarray(x) for x in jax.tree_util.tree_leaves({k: v for k, v in p.items() if k != "WX"})]
T = {s: leaves(N.init("M", Bm.Env("T1", K=3, Q=6), s)) for s in range(70, 78)}
P = {s: leaves(TT.init("pcount", s, "M")) for s in range(80, 88)}
def ident(a, b): return sum(1 for x, y in zip(a, b) if x.shape == y.shape and x.std() > 0 and np.array_equal(x, y))
out = dict(n_arrays=len(T[70]), cross=max(ident(T[a], P[b]) for a in T for b in P),
           within_T=max(ident(T[a], T[b]) for a, b in itertools.combinations(T, 2)), within_P=max(ident(P[a], P[b]) for a, b in itertools.combinations(P, 2)))
json.dump(out, open(os.path.join(HERE, "..", "logs", "independence.json"), "w"), indent=1); print(out)
