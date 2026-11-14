"""Workstream B1/B2: how is the eligibility representation constructed inside the fine-tuned layers 0-1?
For model M (full_c1 or c1) with its Investigation X E-subspace Q (DAS, L2/q7/k6), and natural on-support edit pairs:
  D2   : create agreement at q of a perfect group (n_h = 5)        -> e_h 0 -> 1   [via agreement-at-q]
  D2c  : create agreement at q of a group with n_h <= 3            -> e unchanged  [agreement control]
  CNT  : F45 R2/R3 (count 4 -> 5 of a group agreeing at q)          -> e_h 0 -> 1   [via perfectness]
  CNTc : F34 R1 (count 3 -> 4)                                      -> e unchanged  [perfectness control]
(1) EXACT DIRECT-WRITE DECOMPOSITION.  resid2 = emb + sum_{l in 0,1} (sum_h head_{l,h} + mlp_l)  (residual stream is additive), so the
    E-subspace displacement at the q7 tokens, Q^T d resid2, splits exactly into per-component contributions; share_c = <Q^T dz_c, Q^T dz>/|Q^T dz|^2.
(2) CONJUNCTION DECODABILITY by stage (natural decisions; held-out linear probes on the concatenated q7 token states of the target question):
    targets for each group h: agree_h(q), perf_h (n_h = 5), e_h = agree & perf; and the XOR-free control 'agree OR perf'.
    stages: resid0 (embeddings), resid0 + layer-0 attention (mid0), resid1, resid1 + layer-1 attention (mid1), resid2.
(3) NECESSITY / SUFFICIENCY of each layer-0/1 component for the D2 and CNT effects (total-effect variant, everything downstream recomputed):
    block_c : run AFTER with c's output taken from BEFORE  -> fraction of the E-subspace displacement and of the decision response that remains
    only_c  : run BEFORE with c's output taken from AFTER  -> fraction recovered
Usage: mech_construct.py RSUFFIX SPEC...  (SPEC = parent:spec:seed:suffix; RSUFFIX '' or _ev62500000) -> logs/mech_construct<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; specs = sys.argv[2:]
X_DAS = os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das")
EVP = os.environ.get("MC_EVAL", "dev"); ev = NV.load_eval() if EVP == "dev" else NV.load_eval(EVP)
X = ev["X"]; q = ev["q"]; n = len(X); idx = np.arange(n)
cnt = GL.counts(X, 3)[idx, q]; a_ = X.reshape(n, 6, 3, 2)[idx, q]; ag = a_[..., 0] == a_[..., 1]; rng = np.random.default_rng(1102)
def edits(kind, nmax=500):
    b, Xe, hs = [], [], []
    for i in range(n):
        cand = [h for h in range(3) if not ag[i, h] and ((cnt[i, h] == 5) if kind == "D2" else (cnt[i, h] <= 3))]
        if not cand: continue
        h = cand[rng.integers(len(cand))]; x = X[i].copy(); x[q[i], 2 * h + 1] = x[q[i], 2 * h]; b.append(i); Xe.append(x); hs.append(h)
        if len(b) >= nmax: break
    b = np.array(b); return X[b], np.array(Xe), q[b], np.array(hs), np.array([x[q[i], 2 * h] for x, i, h in zip(Xe, b, hs)])
fams = {"D2": edits("D2"), "D2c": edits("D2c")}
def from_eval(mask, Xa_key, nmax=500):
    s = np.where(mask)[0][:nmax]; Xb, Xa = X[s], ev[Xa_key][s]; i = np.arange(len(s))
    h = np.argmax(np.abs(GL.counts(Xa, 3)[i, q[s]] - cnt[s]), 1); return Xb, Xa, q[s], h, Xa.reshape(len(s), 6, 3, 2)[i, q[s], h, 0]
fams["CNT"] = from_eval(np.isin(ev["reg_F45"], ["R2", "R3"]), "X_F45"); fams["CNTc"] = from_eval(ev["reg_F34"] == "R1", "X_F34")
COMPS = ["emb"] + [f"h{l}.{h}" for l in (0, 1) for h in range(4)] + ["mlp0", "mlp1"]
def comp_outputs(M, Xb):
    rec = {}; L = F.forward(M, jnp.asarray(Xb), rec=rec); r = {"emb": rec["resid0"]}
    for l in (0, 1):
        for h in range(4): r[f"h{l}.{h}"] = rec[f"heads{l}"][:, h]
        r[f"mlp{l}"] = rec[f"mlp_out{l}"]
    return np.asarray(L), {k: np.asarray(v) for k, v in r.items()}, rec
def q7(Z, qq): i = np.arange(len(qq)); tok = np.concatenate([(36 + qq)[:, None], qq[:, None] * 6 + np.arange(6)[None]], 1); return Z[i[:, None], tok].reshape(len(qq), -1)
def margin(L, qq, vv):
    i = np.arange(len(qq)); c = F.centred(L[i, qq]); oth = np.where(np.arange(3)[None] == vv[:, None], -np.inf, c).max(-1); return c[i, vv] - oth
out = {}
for sp in specs:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    Q, _ = np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy"))); res = {"decomposition": {}, "block_only": {}}
    for f, (Xb, Xa, qq, hh, vv) in fams.items():
        Lb, cb, recb = comp_outputs(M, Xb); La, ca, reca = comp_outputs(M, Xa)
        dz = q7(np.asarray(reca["resid2"]) - np.asarray(recb["resid2"]), qq) @ Q
        parts = {c: q7(ca[c] - cb[c], qq) @ Q for c in COMPS}
        recon = sum(parts.values()); tot2 = (dz * dz).sum(1)
        share = {c: float(((parts[c] * dz).sum(1) / np.maximum(tot2, 1e-12)).mean()) for c in COMPS}
        res["decomposition"][f] = dict(n=len(qq), E_disp=float(np.sqrt(tot2).mean()), reconstruction_err=float(np.abs(recon - dz).max()), shares=share,
                                       abs_contrib={c: float(np.linalg.norm(parts[c], axis=1).mean()) for c in COMPS})
        if f in ("D2", "CNT"):
            dm_full = margin(La, qq, vv) - margin(Lb, qq, vv); bo = {}
            for c in COMPS[1:]:
                l, hd = (int(c[1]), int(c[3])) if c.startswith("h") else (int(c[3]), None)
                m = jnp.ones((len(qq), 43), bool)
                def run(Xrun, src):
                    if hd is not None: return F.forward(M, jnp.asarray(Xrun), head_patch={(l, hd): (m, jnp.asarray(src[c]))}, rec=(r := {})), r
                    return F.forward(M, jnp.asarray(Xrun), mlp_patch={l: (m, jnp.asarray(src[c]))}, rec=(r := {})), r
                Lblk, rblk = run(Xa, cb); Lonly, ronly = run(Xb, ca)
                e_blk = (q7(np.asarray(rblk["resid2"]) - np.asarray(recb["resid2"]), qq) @ Q * dz).sum(1) / np.maximum(tot2, 1e-12)
                e_only = (q7(np.asarray(ronly["resid2"]) - np.asarray(recb["resid2"]), qq) @ Q * dz).sum(1) / np.maximum(tot2, 1e-12)
                dm_blk = margin(np.asarray(Lblk), qq, vv) - margin(Lb, qq, vv); dm_only = margin(np.asarray(Lonly), qq, vv) - margin(Lb, qq, vv)
                bo[c] = dict(E_remaining_when_blocked=float(e_blk.mean()), E_recovered_only=float(e_only.mean()),
                             dm_remaining_when_blocked=float(dm_blk.mean() / dm_full.mean()), dm_recovered_only=float(dm_only.mean() / dm_full.mean()))
            res["block_only"][f] = dict(dm_full=float(dm_full.mean()), comps=bo)
    # (2) decodability by stage on natural decisions
    nd = 4000; Xn, qn = X[:nd], q[:nd]; rec = {}; F.forward(M, jnp.asarray(Xn), rec=rec)
    mid0 = np.asarray(rec["resid0"]) + np.asarray(rec["heads0"]).sum(1); mid1 = np.asarray(rec["resid1"]) + np.asarray(rec["heads1"]).sum(1)
    stages = {"resid0": np.asarray(rec["resid0"]), "mid0": mid0, "resid1": np.asarray(rec["resid1"]), "mid1": mid1, "resid2": np.asarray(rec["resid2"])}
    perf = (cnt[:nd] == 5); agq = ag[:nd]; e_ = perf & agq; dec = {}
    for s, Z in stages.items():
        H = q7(Z, qn); d2 = {}
        for nm, T in (("agree", agq), ("perf", perf), ("e", e_), ("agree_or_perf", agq | perf)):
            accs = []
            for h in range(3):
                y = T[:, h].astype(int); clf = LogisticRegression(C=1.0, max_iter=2000).fit(H[:3000], y[:3000]); accs.append(clf.score(H[3000:], y[3000:]))
            d2[nm] = float(np.mean(accs))
        d2["majority_e"] = float(np.mean([max(e_[3000:, h].mean(), 1 - e_[3000:, h].mean()) for h in range(3)])); dec[s] = d2
    res["decodability"] = dec; out[sp] = res
    print(sp, {f: {c: round(v, 2) for c, v in d["shares"].items() if abs(v) > 0.08} for f, d in res["decomposition"].items()}, flush=True)
    print("   decod e:", {s: round(v["e"], 3) for s, v in dec.items()}, "agree:", {s: round(v["agree"], 3) for s, v in dec.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "mech_construct" + os.environ.get("MC_TAG", "") + ".json"), "w"), indent=1)
