"""ADJUDICATION CHECK (post hoc; written in response to review finding F12 of R_formal_equivalence.md).
Question: does a P-count site hold V = n, or only (approximately) the posterior pi(n)?
For each confirmation model, at its frozen site (P: readout 36+q; T: q's six source tokens), natural fit stream 12_300_000
(4000 episodes, all q) and held-out stream 12_300_001 (2000 episodes):
  (a) decoder for the pi-class label (43 classes; multinomial logistic)              -> accuracy vs majority baseline
  (b) decoder for the WITHIN-class level: among count vectors in pi-classes with >= 2 members, the rank of n inside its
      class (0 = lowest), fitted and scored only on such decisions, stratified so a class-blind predictor is at chance;
      reported against the best class-aware baseline (predict the most frequent rank given the class label).
  (c) site displacement under natural pi-preserving input edits relative to the displacement under single-group edits
      (median ||a(x') - a(x)|| at the site).
If (b) is at baseline and (c) is ~0, the site holds pi (D5 regime); if (b) is above baseline, the site holds more than pi.
Writes logs/review_checks/adj_pi_vs_n.json."""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from sklearn.linear_model import LogisticRegression
import eqv as E
SUPP = [v for v in itertools.product(range(6), repeat=3) if 5 in v]
P = E.pi_of_n(np.array(SUPP)); cls = {}; key = {}
for v, pv in zip(SUPP, P): key[v] = tuple(np.round(pv, 10))
labels = {k: i for i, k in enumerate(sorted(set(key.values())))}
members = {}
for v in SUPP: members.setdefault(labels[key[v]], []).append(v)
rank = {v: sorted(members[labels[key[v]]]).index(v) for v in SUPP}
def lab(n): return np.array([labels[key[tuple(x)]] for x in n]), np.array([rank[tuple(x)] for x in n]), np.array([len(members[labels[key[tuple(x)]]]) for x in n])
Xa = E.sample(4000, 12_300_000)["X"]; Xb = E.sample(2000, 12_300_001)["X"]
out = {}
for m in ["P2", "P3", "P4", "T2", "T10", "T11"]:
    M = E.Model(m); k = "P" if m[0] == "P" else "T"; Aa, Ab = M.acts(Xa)[0], M.acts(Xb)[0]
    def stack(A, X):
        n = E.loo(X); F, C_, R_, S_ = [], [], [], []
        for q in range(E.Q):
            F.append(A[:, E.site_positions(k, q)].reshape(len(A), -1)); c, r, s = lab(n[:, q]); C_.append(c); R_.append(r); S_.append(s)
        return np.concatenate(F), np.concatenate(C_), np.concatenate(R_), np.concatenate(S_)
    Fa, Ca, Ra, Sa = stack(Aa, Xa); Fb, Cb, Rb, Sb = stack(Ab, Xb); mu, sd = Fa.mean(0), Fa.std(0) + 1e-6; Fa, Fb = (Fa - mu) / sd, (Fb - mu) / sd
    c1 = LogisticRegression(max_iter=3000).fit(Fa, Ca); acc_cls = float((c1.predict(Fb) == Cb).mean())
    maj = float((Cb == np.bincount(Ca).argmax()).mean())
    ma, mb = Sa >= 2, Sb >= 2
    Fa2 = np.c_[Fa[ma], np.eye(len(labels))[Ca[ma]]]; Fb2 = np.c_[Fb[mb], np.eye(len(labels))[Cb[mb]]]      # class label given: tests within-class info
    c2 = LogisticRegression(max_iter=3000).fit(Fa2, Ra[ma]); acc_rank = float((c2.predict(Fb2) == Rb[mb]).mean())
    base = {}
    for c, r in zip(Ca[ma], Ra[ma]): base.setdefault(c, []).append(r)
    pred = np.array([np.bincount(base[c]).argmax() if c in base else 0 for c in Cb[mb]]); acc_base = float((pred == Rb[mb]).mean())
    disp = {}
    for fam in ("pipres", "single"):
        ds = []
        for q in range(E.Q):
            r = np.random.default_rng(99 + q)
            if fam == "pipres": Xp, ok = E.pi_preserving_edit(Xb, q, r)
            else: Xp, _, ok, _ = E.single_group_edit(Xb, q, r)
            Ap = M.acts(Xp)[0]; pos = E.site_positions(k, q)
            ds.append(np.linalg.norm((Ap[:, pos] - Ab[:, pos]).reshape(len(Xb), -1), axis=1)[ok])
        disp[fam] = float(np.median(np.concatenate(ds)))
    out[m] = dict(pi_class_decoder_acc=acc_cls, pi_class_majority=maj, within_class_rank_acc=acc_rank, within_class_rank_baseline=acc_base,
                  n_within=int(mb.sum()), displacement_pipres=disp["pipres"], displacement_single=disp["single"],
                  displacement_ratio=disp["pipres"] / disp["single"])
    print(m, json.dumps({k_: round(v, 3) if isinstance(v, float) else v for k_, v in out[m].items()}), flush=True)
os.makedirs(os.path.join(HERE, "..", "logs", "review_checks"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "review_checks", "adj_pi_vs_n.json"), "w"), indent=1)
