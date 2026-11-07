"""R4 check 4: an exact residual-basis rotation gauge using only existing parameters.
R orthogonal with R 1 = 1 commutes with LN centring/scaling; LN gains are folded into the
next matrices; the LN1 bias is re-solved so that query biases match (key-bias changes cancel
in softmax; value-bias changes are absorbed into bO)."""
import sys, copy; sys.path.insert(0, "../../src")
import numpy as np, warnings; warnings.filterwarnings("ignore")
import learned_analysis as LA, learned as Lm
def rot(d, seed):
    r = np.random.default_rng(seed); one = np.ones((d, 1)) / np.sqrt(d)
    U = np.linalg.qr(np.column_stack([one, r.normal(size=(d, d - 1))]))[0][:, 1:]
    Q = np.linalg.qr(r.normal(size=(d - 1, d - 1)))[0]
    return one @ one.T + U @ Q @ U.T
def gauge_rot(P, seed=7):
    P = {k: (np.asarray(v) if k != "layers" else v) for k, v in P.items()}
    P = copy.deepcopy(P); d = P["Wval"].shape[1]; R = rot(d, seed)
    for k in ("Wval", "Eid", "Wp"): P[k] = np.asarray(P[k]) @ R
    for lay in P["layers"]:
        lay.update({k: np.asarray(v) for k, v in lay.items()})
        g1, b1 = lay["g1"], lay["b1"]
        WQ = np.stack([R.T @ np.diag(g1) @ w for w in lay["WQ"]]); WK = np.stack([R.T @ np.diag(g1) @ w for w in lay["WK"]])
        WV = np.stack([R.T @ np.diag(g1) @ w for w in lay["WV"]])
        Qc_old = np.concatenate(list(lay["WQ"]), 1); Qc_new = np.concatenate(list(WQ), 1)
        b1n = np.linalg.lstsq(Qc_new.T, (b1 @ Qc_old), rcond=None)[0]
        dV = [b1n @ WV[h] - b1 @ lay["WV"][h] for h in range(len(WV))]
        lay["bO"] = (lay["bO"] - sum(dV[h] @ lay["WO"][h] for h in range(len(WV)))) @ R
        lay["WO"] = np.stack([w @ R for w in lay["WO"]]); lay["WQ"], lay["WK"], lay["WV"] = WQ, WK, WV
        lay["g1"], lay["b1"] = np.ones(d), b1n
        if "W1" in lay:
            lay["c1"] = lay["c1"] + lay["b2"] @ lay["W1"]; lay["W1"] = R.T @ np.diag(lay["g2"]) @ lay["W1"]
            lay["g2"], lay["b2"] = np.ones(d), np.zeros(d)
            lay["W2"] = lay["W2"] @ R; lay["c2"] = lay["c2"] @ R
    P["bU"] = P["bU"] + P["bf"] @ P["WU"]; P["WU"] = R.T @ np.diag(P["gf"]) @ P["WU"]
    P["gf"], P["bf"] = np.ones(d), np.zeros(d)
    return P
if __name__ == "__main__":
    for task, arch in [("T1", "TF1"), ("T3", "TF2")]:
        P0 = LA.load(task, arch, 0); P1 = gauge_rot(P0)
        val = Lm.sample(task, 4000, np.random.default_rng(5))
        l0 = np.asarray(Lm.forward(P0, task, Lm.to_inp(task, val))); l1 = np.asarray(Lm.forward(P1, task, Lm.to_inp(task, val)))
        print(task, arch, "max logit diff", np.abs(l0 - l1).max())
        r0 = LA.analyse_binary(task, arch, 0, [1], P=P0); r1 = LA.analyse_binary(task, arch, 0, [1], P=P1)
        pr = max(abs(r0["probes"][s][k] - r1["probes"][s][k]) for s in r0["probes"] for k in r0["probes"][s])
        it = {k: (round(r0["interchange"][k]["decision_to_source"], 3), round(r1["interchange"][k]["decision_to_source"], 3)) for k in r0["interchange"]}
        print("  probe maxdiff", round(pr, 4), " n_rel", r0["units_summary"]["n_rel"], r1["units_summary"]["n_rel"],
              " sign hits", r0["units_summary"]["sign_prediction"], r1["units_summary"]["sign_prediction"])
        print("  interchange (orig, rotated):", it)
