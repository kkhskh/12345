"""POST-HOC (written after the confirmation outcomes; not a frozen test).  Diagnosis of the Prediction II failure.
The frozen decoder alpha_M pools all six decision positions into one classifier per group.  For P-count models the six
readout tokens 36..41 carry position-specific offsets, so a pooled decoder can be mis-specified (the same error class as
Investigation III's shared source encoder).  Here: per-position decoders (one multinomial logistic per (q, group)), same fit
stream and evaluation stream as conf, then the same transport computation for every ordered pair on the cond and single
families.  Writes logs/conf/transport_posthoc.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from sklearn.linear_model import LogisticRegression
import eqv as E
names = ["P2", "P3", "P4", "T2", "T10", "T11"]; FIT, EVAL, NF, NE = 12_300_000, 12_400_000, 4000, 1000
Xf = E.sample(NF, FIT)["X"]; X = E.sample(NE, EVAL)["X"]; cfg = E.sample(NE, EVAL)["cfg"]; n0 = E.loo(X); nf = E.loo(Xf)
fam = {}
for q in range(E.Q):
    r = np.random.default_rng(EVAL + 100 + q)
    fam[("cond", q)] = E.cond_pairs(X, q, r); fam[("single", q)] = E.single_group_edit(X, q, r, cfg)[0]   # identical RNG order to equiv_run
C = {}
for m in names:
    M = E.Model(m); k = "P" if m[0] == "P" else "T"; Af = M.acts(Xf)[0]; A0, L0, _, _ = M.acts(X)
    decs = {}
    for q in range(E.Q):
        pos = E.site_positions(k, q); Fe = Af[:, pos].reshape(NF, -1); mu, sd = Fe.mean(0), Fe.std(0) + 1e-6
        decs[q] = (mu, sd, [LogisticRegression(max_iter=3000, C=1.0).fit((Fe - mu) / sd, nf[:, q, g]) for g in range(3)])
    def dec(A, q, decs=decs, k=k):          # bind this model's decoders (late-binding bug fixed)
        mu, sd, cl = decs[q]; Fe = (A[:, E.site_positions(k, q)].reshape(len(A), -1) - mu) / sd; P = np.zeros((len(A), 3, 6))
        for g in range(3): P[:, g, cl[g].classes_] = cl[g].predict_proba(Fe)
        return P
    acc = float(np.mean([(dec(A0, q).argmax(-1) == n0[:, q]).mean() for q in range(E.Q)]))
    enc = E.fit_encoder("P_full" if k == "P" else "T_own", Xf, Af)
    nat = {key: M.acts(Xp) for key, Xp in fam.items()}
    C[m] = dict(M=M, k=k, dec=dec, acc=acc, enc=enc, A0=A0, L0=L0, nat=nat, d=M.d); print(m, "per-position decoder acc", round(acc, 3), flush=True)


def soft_delta(c, q, P0, P1):
    dl = np.zeros((NE, E.T, c["d"]), np.float32)
    for t in E.site_positions(c["k"], q):
        W = c["enc"][t]
        if c["k"] == "P": dl[:, t] = (P1.reshape(NE, 18) - P0.reshape(NE, 18)) @ W
        else: g = (t % 6) // 2; dl[:, t] = (P1[:, g] - P0[:, g]) @ W
    return dl


out = {m: dict(decoder_acc_per_position=C[m]["acc"]) for m in names}
for D in names:
    for Rm in names:
        cd, cr = C[D], C[Rm]
        for f in ("cond", "single"):
            acc = {"dd": [], "fl": [], "or": []}
            for q in range(E.Q):
                Xp = fam[(f, q)]; P0d, P1d = cd["dec"](cd["A0"], q), cd["dec"](cd["nat"][(f, q)][0], q); sh = np.roll(np.arange(NE), 1)
                b, nt = cr["L0"][:, q], cr["nat"][(f, q)][1][:, q]
                dlo = E.encoder_delta(cr["enc"], "P_full" if cr["k"] == "P" else "T_own", X, n0, E.loo(Xp), cr["d"], E.site_positions(cr["k"], q))
                for nm, dd in (("dd", soft_delta(cr, q, P0d, P1d)), ("fl", soft_delta(cr, q, P0d[sh], P1d[sh])), ("or", dlo)):
                    acc[nm].append((b, nt, cr["M"].run(X, dd)[:, q]))
            s = {nm: E.score(*(np.concatenate([x[i] for x in v]) for i in range(3))) for nm, v in acc.items()}
            out[f"{D}->{Rm}|{f}"] = dict(iia=s["dd"]["iia"], oracle=s["or"]["iia"], floor=s["fl"]["iia"],
                                          fidelity=E.adjusted(s["dd"]["iia"], s["fl"]["iia"], s["or"]["iia"]))
    print(D, "done", flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "conf", "transport_posthoc.json"), "w"), indent=1)
