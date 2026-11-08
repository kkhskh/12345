"""Review R3 #11: own-question Bayes accuracy when the pairing is UNKNOWN (the correct reference for PP4)."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, gen_family as G
out = {}
for nm, f in {"T1": G.Fam(pc=0.0, perm=True), "T4": G.Fam(pc=1.0, rho=1.0, perm=True)}.items():
    d = G.sample(f, 3000, np.random.default_rng(8_100_000)); X, Y = d["X"], d["Y"]
    fu = G.Fam(pc=f.pc, rho=f.rho)
    Js, pri = [], []
    for P in G.ALL_PAIRINGS:
        Js.append(G.joint(fu, X, list(P))); pri.append(fu.prior / len(G.ALL_PAIRINGS))
    J = np.concatenate(Js, 2); pr = np.concatenate(pri)
    own = (pr[None, None, :, None] * J).sum(2)
    maj = np.stack([(X == k).sum(-1) for k in range(3)], -1) + 1e-6 * np.random.default_rng(0).random(X.shape[:2] + (3,))
    out[nm] = dict(own_question_bayes_unknown_pairing=float((own.argmax(-1) == Y).mean()), majority_vote=float((maj.argmax(-1) == Y).mean()))
json.dump(out, open(os.path.join(HERE, "..", "logs", "pp4_own_unknown.json"), "w"), indent=1); print(out)
