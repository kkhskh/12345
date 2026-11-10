"""Reviewer check: episode-bootstrap uncertainty of VI-bayes (r of each model's random-support response with exact Bayes response),
and approximate sampling SE of the threshold-type statistics that decided IV(a)/IV(b) (binomial approximation; no per-decision data logged)."""
import os, json, numpy as np
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
Z = np.load(os.path.join(B, "logs/conf/responses.npz")); mk = Z["mask|random_support"]; idx = np.where(mk)[0]; dB = Z["dB"]
rng = np.random.default_rng(7); R = json.load(open(os.path.join(B, "logs/conf/results.json")))
for m in ["P2", "P3", "P4", "T2", "T10", "T11"]:
    beta = R["decision_response"]["random_support"][f"{m}|beta"]
    x = Z[f"random_support|{m}"] + beta * dB   # stored response is the Bayes residual (equiv_run.py); rebuild raw response
    pt = np.corrcoef(x[idx].ravel(), dB[idx].ravel())[0, 1]
    bs = [np.corrcoef(x[ii].ravel(), dB[ii].ravel())[0, 1] for ii in (rng.choice(idx, len(idx)) for _ in range(1000))]
    print(f"VI-bayes {m}: r {pt:.3f} CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}] P(r<0.5) {np.mean(np.array(bs)<0.5):.3f}")
N = json.load(open(os.path.join(B, "logs/conf/necessity.json")))
for m in ["P2", "P4"]:
    o = N["per_model"][m]; a0, asp, are = o["clean"]["acc"], o["span_ablate"]["acc"], o["span_ablate_restore_true"]["acc"]
    n = 6000; f = o["span_ablate"]["frac_decisions_changed"]
    se_num = np.sqrt(f / n); den = a0 - asp
    print(f"IV(b) {m}: restoration {(are-asp)/den:.3f}; denominator {den:.3f}; rough SE (paired, binomial on changed decisions) ~{se_num/den:.3f}")
