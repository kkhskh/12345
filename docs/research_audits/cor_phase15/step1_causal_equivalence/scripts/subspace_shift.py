"""Does a model's learned eligibility (E) subspace respond to the INPUT changes that define eligibility?  (development diagnostic)
For model M with its DAS E-subspace R (L2/q7/k6, logs/das/<tag>_R.npy), measure the displacement of the q7 state inside the subspace,
  d = || Q^T (x_after - x_before) ||  (Q = orthonormalised R),
for natural on-support edits that change e_h through DIFFERENT inputs:
  D2  : create agreement at q of a perfect group (e_h 0 -> 1 via the q-reports)            [A1 'perfect-only' rule ignores this]
  D2c : control -- change the q-report of a NON-perfect group so it starts agreeing (e unchanged)
  CNT : F45 R2/R3 pairs (a group's count 4 -> 5 while agreeing at q: e_h 0 -> 1 via the counts)
  CNTc: F34 R1 pairs (count 3 -> 4: e unchanged)
Ratio D2/CNT near 1 => the subspace reads eligibility from both inputs; << 1 => it reads perfectness (counts) but not q-agreement.
Usage: subspace_shift.py SPEC...  (parent:spec:seed:suffix) -> logs/subspace_shift.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
EVP = os.environ.get("SHIFT_EVAL", "dev"); ev = NV.load_eval() if EVP == "dev" else NV.load_eval(EVP); X = ev["X"]; q = ev["q"]; n = len(X); idx = np.arange(n)
cnt = GL.counts(X, 3)[idx, q]; a_ = X.reshape(n, 6, 3, 2)[idx, q]; ag = a_[..., 0] == a_[..., 1]; rng = np.random.default_rng(91)
def edits(kind):
    b, Xe = [], []
    for i in range(n):
        if kind == "D2": cand = [h for h in range(3) if cnt[i, h] == 5 and not ag[i, h]]
        else: cand = [h for h in range(3) if cnt[i, h] <= 3 and not ag[i, h]]
        if not cand: continue
        h = cand[rng.integers(len(cand))]; x = X[i].copy(); x[q[i], 2 * h + 1] = x[q[i], 2 * h]; b.append(i); Xe.append(x)
    b = np.array(b[:600]); return X[b], np.array(Xe[:600]), q[b]
fams = {"D2": edits("D2"), "D2c": edits("D2c")}
s = np.isin(ev["reg_F45"], ["R2", "R3"]); fams["CNT"] = (X[s][:600], ev["X_F45"][s][:600], q[s][:600])
s = (ev["reg_F34"] == "R1") if "reg_F34" in ev else np.ones(n, bool); fams["CNTc"] = (X[s][:600], ev["X_F34"][s][:600], q[s][:600])
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    R = np.load(os.path.join(HERE, "..", "logs", "das", f"{M['tag']}_E_L2_q7_k6" + os.environ.get("SHIFT_RSUFFIX", "") + "_R.npy")); Qm, _ = np.linalg.qr(R)
    rec = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), r["resid2"])[1])({}))
    res = {}
    for f, (Xb, Xa, qq) in fams.items():
        i = np.arange(len(qq)); tok = np.concatenate([(36 + qq)[:, None], qq[:, None] * 6 + np.arange(6)[None]], 1)
        zb = np.asarray(rec(jnp.asarray(Xb)))[i[:, None], tok].reshape(len(qq), -1); za = np.asarray(rec(jnp.asarray(Xa)))[i[:, None], tok].reshape(len(qq), -1)
        dv = za - zb; res[f] = dict(n=len(qq), in_subspace=float(np.linalg.norm(dv @ Qm, axis=1).mean()), total=float(np.linalg.norm(dv, axis=1).mean()))
    res["ratio_D2_over_CNT"] = res["D2"]["in_subspace"] / res["CNT"]["in_subspace"]
    res["ratio_D2c_over_D2"] = res["D2c"]["in_subspace"] / res["D2"]["in_subspace"]; res["ratio_CNTc_over_CNT"] = res["CNTc"]["in_subspace"] / res["CNT"]["in_subspace"]
    out[sp] = res; print(sp, {k: (round(v["in_subspace"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in res.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "subspace_shift" + os.environ.get("SHIFT_TAG", "") + ".json"), "w"), indent=1)
