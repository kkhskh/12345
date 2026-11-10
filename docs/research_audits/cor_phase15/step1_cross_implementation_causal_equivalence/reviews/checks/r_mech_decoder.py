"""Reviewer check (mechanistic): is P-count's 'lossy code' (per-component decodability 0.72-0.79) a decoder artefact?
Same fit / eval streams as conf (fit 12_300_000 first 4000, eval 12_400_000 1000).  Per-position (q) decoders of
n^{(-q)} at the readout 36+q from resid_pre2, with several decoder families; also readout + q's six source tokens; also a
joint 91-class Gaussian (LDA) decoder of the whole vector and the fraction of its errors that are pi-equivalent.
Usage: python reviews/checks/r_mech_decoder.py P2 P3 P4 T2"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import eqv as E
t0 = time.time()
Xf = E.sample(4000, 12_300_000)["X"]; X = E.sample(1000, 12_400_000)["X"]; nf, ne = E.loo(Xf), E.loo(X)
def pikey(n):
    p = E.pi_of_n(n); return np.round(np.nan_to_num(p), 6)
out = {}
for m in sys.argv[1:]:
    M = E.Model(m); Af = M.acts(Xf)[0]; Ae = M.acts(X)[0]; res = {}
    for site in os.environ.get("SITES", "readout,readout+qsrc").split(","):
        accs = {}
        for q in range(E.Q):
            pos = [36 + q] + (list(range(6 * q, 6 * q + 6)) if site != "readout" else [])
            Fa, Fe = Af[:, pos].reshape(4000, -1), Ae[:, pos].reshape(1000, -1)
            sc = StandardScaler().fit(Fa); Fa_, Fe_ = sc.transform(Fa), sc.transform(Fe)
            decs = {"logit_C1_std": lambda: LogisticRegression(max_iter=3000, C=1.0),
                    "logit_C100_std": lambda: LogisticRegression(max_iter=5000, C=100.0),
                    "lda": lambda: LDA(),
                    "mlp256": lambda: MLPClassifier((256,), max_iter=400, alpha=1e-4, random_state=0)}
            if site == "readout": decs["qda_reg"] = lambda: QDA(reg_param=0.1)
            for nm, mk in decs.items():
                a = np.mean([(mk().fit(Fa_, nf[:, q, g]).predict(Fe_) == ne[:, q, g]).mean() for g in range(3)])
                accs.setdefault(nm, []).append(a)
            if site == "readout":
                # joint decoder over the whole count vector (91 support classes) + pi-equivalence of errors
                ya = nf[:, q] @ np.array([36, 6, 1]); ye = ne[:, q] @ np.array([36, 6, 1])
                cl = LDA().fit(Fa_, ya); pr = cl.predict(Fe_)
                dv = np.stack([pr // 36, (pr // 6) % 6, pr % 6], -1)
                err = pr != ye
                pieq = (np.abs(pikey(dv) - pikey(ne[:, q])).max(-1) < 1e-6)
                accs.setdefault("joint_lda_exact", []).append(float((~err).mean()))
                accs.setdefault("joint_lda_component", []).append(float((dv == ne[:, q]).mean()))
                accs.setdefault("joint_lda_frac_errors_pi_equiv", []).append(float(pieq[err].mean()) if err.any() else np.nan)
                # error distance: mean |decoded - true| per component among component errors
                d_ = np.abs(dv - ne[:, q]); accs.setdefault("joint_lda_mean_abs_err_given_err", []).append(float(d_[d_ > 0].mean()))
        res[site] = {k: round(float(np.nanmean(v)), 4) for k, v in accs.items()}
    out[m] = res; print(m, json.dumps(res), f"[{time.time()-t0:.0f}s]", flush=True)
