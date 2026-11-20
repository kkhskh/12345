"""Track C, part 3: can a bounded-order trust complex (the repaired O/D's additive energy form) represent T4's exact Bayes rule?

T4 has a LATENT role assignment (which group is reliable, which colludes).  Conditional on the roles the likelihood factorises into
order-2 per-group-per-question factors, so a complex that CONTAINS the latent role is a standard factor graph whose decision quantity is
the marginal free energy -log sum_r exp(-sum_h E_h(x, c, r)) -- i.e. exact Bayes, not an additive D.  The repaired note defines D only in
the additive form D_c = sum_h E_h(x_h, c) over observed inputs.  Question: what hyperedge order would an additive complex need?

Method (exact up to float64 rounding, which is bounded and reported): for a set T of input coordinates, the interaction coefficient of
the Bayes log-odds f(x) = log p(y=c|x) - log p(y=c'|x) on T at base point x0 with alternative values x1 is
   I_T = sum_{U subset T} (-1)^{|T|-|U|} f(x0 with coordinates in U set to x1).
If I_T != 0, every additive representation f = sum_h phi_h(x_h) needs a hyperedge containing T (all phi_h with h not containing T are
annihilated by the alternating sum).  So |T| is a lower bound on the required order.  We evaluate I_T for sets spanning all six
questions (and growing sizes) using genlib.posterior_brute (exact enumeration over roles; no closed form assumed)."""
import itertools, numpy as np


def interaction(GL, env, x0, x1, coords, q=0, c=0, cp=1):
    n = len(coords); subs = np.array(list(itertools.product([0, 1], repeat=n)), bool)
    X = np.repeat(x0[None], len(subs), 0)
    for j, (qq, s) in enumerate(coords): X[subs[:, j], qq, s] = x1[qq, s]
    P = GL.posterior_brute(env, X, np.full(len(X), q)); f = np.log(P[:, c]) - np.log(P[:, cp])
    sign = (-1.0) ** (n - subs.sum(1)); val = float((sign * f).sum()); bound = float(np.abs(f).sum() * 4 * np.finfo(float).eps * n)
    return val, bound


def run(GL, env, seed=0, n_bases=5, sizes=(2, 6, 12, 18)):
    rng = np.random.default_rng(seed); res = []
    for b in range(n_bases):
        d = GL.sample(env, 1, rng); x0 = d["X"][0]; x1 = (x0 + rng.integers(1, env.K, x0.shape)) % env.K
        g_rel, hs = env.roles[int(d["ridx"][0])]; free = [s for s in range(2 * env.G) if s // 2 not in hs]   # never touch the colluding pair:
        for size in sizes:                                                                                     # every flipped input stays on-support
            # coordinates: a target-question source, then non-colluder sources spread over all other questions (round-robin)
            coords = [(0, free[0])] + [(1 + (j % 5), free[(j // 5) % len(free)]) for j in range(size - 1)]
            if len(set(coords)) < len(coords): continue
            val, bound = interaction(GL, env, x0, x1, coords)
            res.append(dict(base=b, size=size, spans_questions=sorted({c[0] for c in coords}), interaction=val, rounding_bound=bound, nonzero=bool(abs(val) > 10 * bound + 1e-12)))
    lb = max([r["size"] for r in res if r["nonzero"]] or [0])
    return dict(results=res, certified_order_lower_bound=lb,
                statement=(f"No additive complex over observed inputs with all hyperedges of order < {lb} reproduces T4's exact Bayes LOG-ODDS (R1-type); "
                           "decision-level (R3-type) representability is not addressed. With order unrestricted a single all-coordinate hyperedge represents the "
                           "log-odds trivially; the only non-trivial completion adds latent roles, which is standard factor-graph marginal inference. "
                           "Certified by nonzero interaction coefficients exceeding a heuristic float64 rounding bound; independently recomputed at 40 digits by "
                           "review B (size 18: -20.8558394526408 and 138.635712661052; tests/independent/t4_highprecision_certificate.py)."))
