"""Stage-6 functional test battery (mandate §8 E2; pre-review C3: the Claim-2 test that RPC is not).  Route-free: only input-output
behaviour on controlled T4 episodes, scored against exact Bayes (genlib.posterior_closed, ENV1 = T4: 3 groups of 2 sources, one reliable
(alpha 0.8), one colluding, one noise; K = 3; Q = 6).  Episodes are sampled from the T4 generator with their latent roles and binned into
cells defined by the latent structure at the target question q:
  agree_conflict   reliable pair agrees at q, colluding pair agrees on a different answer
  agree_consensus  reliable pair agrees at q, colluders agree on the same answer
  split            reliable pair disagrees at q (one source wrong)
  weak_roles       reliable pair agrees on <= 3 of the 5 other questions (weak role evidence), any q pattern
  strong_roles     reliable pair agrees on all 5 other questions
Per cell and model: KL(Bayes || model), Bayes-argmax agreement, colluder capture (model argmax = colluders' answer where Bayes' argmax
differs), and the calibration slope of the model's centred log-odds on Bayes' centred log-odds.  All cells are on-support (natural
generator samples); none is off-support.  Usage: func_battery19.py STREAM OUT.json DIR@PARENT@STEP ..."""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src"))
import numpy as np, jax, jax.numpy as jnp
import xviiim as XM
GL, RP, F = XM.GL, XM.RP, XM.F
def cells(stream, n=60000):
    rng = np.random.default_rng(stream); d = GL.sample(RP.ENV1, n, rng); X, ridx = d["X"], d["ridx"]; q = rng.integers(0, 6, n); i = np.arange(n)
    roles = np.array([RP.ENV1.roles[r][0] for r in ridx]); coll = np.array([RP.ENV1.roles[r][1][0] for r in ridx])
    a = X.reshape(n, 6, 3, 2); rq = a[i, q, roles]; cq = a[i, q, coll]
    ragree = rq[:, 0] == rq[:, 1]; cans = cq[:, 0]
    eq = (a[..., 0] == a[..., 1]); rel_other = eq[i, :, roles].sum(1) - eq[i, q, roles]
    C = dict(agree_conflict=ragree & (cans != rq[:, 0]), agree_consensus=ragree & (cans == rq[:, 0]), split=~ragree, weak_roles=rel_other <= 3, strong_roles=rel_other == 5)
    out = {}
    for k, m in C.items():
        idx = np.where(m)[0][:3000]; out[k] = dict(X=X[idx], q=q[idx], col=cans[idx], PB=GL.posterior_closed(RP.ENV1, X[idx], q[idx]))
    return out
def measure(M, S):
    f = jax.jit(lambda X_: F.forward(M, X_)); R = {}
    for k, c in S.items():
        L = np.asarray(F.batched(f, c["X"]))[np.arange(len(c["q"])), c["q"]]; P = RP.sm(L); PB = c["PB"]
        kl = float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()); am = P.argmax(-1); bm = PB.argmax(-1)
        cap = (bm != c["col"]); capture = float((am[cap] == c["col"][cap]).mean()) if cap.any() else float("nan")
        lm = np.log(P + 1e-9); lm -= lm.mean(-1, keepdims=True); lb = np.log(PB + 1e-9); lb -= lb.mean(-1, keepdims=True)
        slope = float((lm * lb).sum() / (lb * lb).sum())
        R[k] = dict(n=len(c["q"]), KL=kl, acc=float((am == bm).mean()), colluder_capture=capture, slope=slope)
    return R
if __name__ == "__main__":
    stream, out = int(sys.argv[1]), sys.argv[2]; S = cells(stream); res = json.load(open(out)) if os.path.exists(out) else {}
    for spec in sys.argv[3:]:
        d, parent, step = spec.split("@"); key = os.path.basename(os.path.normpath(d)) + f"@{step}"
        if key in res: continue
        res[key] = measure(XM.load_dir(d, parent, int(step)), S); json.dump(res, open(out, "w"), indent=1)
        print(key[:60], {k: (round(v["KL"], 3), round(v["slope"], 2), round(v["colluder_capture"], 3)) for k, v in res[key].items()}, flush=True)
