"""Reviewer check: does P->X transport fidelity rise when the P donor's decoder may read q's source tokens as well as the
readout (P2 readout+q_src decodes n at 0.97 with LDA, r_mech_decoder.out)?  conf streams, cond family (rng EVAL+100+q),
per-position per-group LDA decoders (standardised), soft delta through the recipient's primary encoder exactly as
transport_posthoc.py; fidelity = (IIA - floor)/(oracle - floor), floor = donor deltas of another episode.
Usage: r_mech_transport_dec.py DONOR RECIPIENT ..."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import eqv as E
t0 = time.time(); NF, NE = 4000, 1000
Xf = E.sample(NF, 12_300_000)["X"]; X = E.sample(NE, 12_400_000)["X"]; n0, nf = E.loo(X), E.loo(Xf)
cps = {q: E.cond_pairs(X, q, np.random.default_rng(12_400_000 + 100 + q)) for q in range(E.Q)}
D, recs = sys.argv[1], sys.argv[2:]; sh = np.roll(np.arange(NE), 1)
MD = E.Model(D); AfD = MD.acts(Xf)[0]; A0D = MD.acts(X)[0]; ApD = {q: MD.acts(cps[q])[0] for q in range(E.Q)}
dec = {}
for site in ("readout", "readout+q_src"):
    for q in range(E.Q):
        pos = [36 + q] + (list(range(6 * q, 6 * q + 6)) if site != "readout" else [])
        sc = StandardScaler().fit(AfD[:, pos].reshape(NF, -1)); cl = [LDA().fit(sc.transform(AfD[:, pos].reshape(NF, -1)), nf[:, q, g]) for g in range(3)]
        def f(A, pos=pos, sc=sc, cl=cl):
            P = np.zeros((len(A), 3, 6)); Z = sc.transform(A[:, pos].reshape(len(A), -1))
            for g in range(3): P[:, g, cl[g].classes_] = cl[g].predict_proba(Z)
            return P
        dec[(site, q)] = (f(A0D), f(ApD[q]))
    print(D, site, "decoder acc", round(float(np.mean([(dec[(site, q)][0].argmax(-1) == n0[:, q]).mean() for q in range(E.Q)])), 3), flush=True)
for Rm in recs:
    MR = E.Model(Rm); k = Rm[0]; ek = "P_full" if k == "P" else "T_own"; AfR = MR.acts(Xf)[0]; A0R, L0R, _, _ = MR.acts(X)
    enc = E.fit_encoder(ek, Xf, AfR); res = {}
    def soft(q, P0, P1):
        dl = np.zeros((NE, E.T, MR.d), np.float32)
        for t in E.site_positions(k, q):
            if k == "P": dl[:, t] = (P1.reshape(NE, 18) - P0.reshape(NE, 18)) @ enc[t]
            else: g = (t % 6) // 2; dl[:, t] = (P1[:, g] - P0[:, g]) @ enc[t]
        return dl
    agg = {}
    for q in range(E.Q):
        Lp = MR.acts(cps[q])[1]; b, nt = L0R[:, q], Lp[:, q]
        agg.setdefault("oracle", []).append((b, nt, MR.run(X, E.encoder_delta(enc, ek, X, n0, E.loo(cps[q]), MR.d, E.site_positions(k, q)))[:, q]))
        for site in ("readout", "readout+q_src"):
            P0, P1 = dec[(site, q)]
            agg.setdefault(site, []).append((b, nt, MR.run(X, soft(q, P0, P1))[:, q]))
            agg.setdefault(site + "|floor", []).append((b, nt, MR.run(X, soft(q, P0[sh], P1[sh]))[:, q]))
    s = {nm: E.score(*(np.concatenate([x[i] for x in v]) for i in range(3)))["iia"] for nm, v in agg.items()}
    for site in ("readout", "readout+q_src"):
        res[site] = dict(iia=round(s[site], 4), floor=round(s[site + "|floor"], 4), oracle=round(s["oracle"], 4),
                         fidelity=round(E.adjusted(s[site], s[site + "|floor"], s["oracle"]), 3))
    print(f"{D}->{Rm}", json.dumps(res), f"[{time.time() - t0:.0f}s]", flush=True)
