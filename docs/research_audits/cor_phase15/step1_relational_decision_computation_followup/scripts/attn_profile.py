"""Workstream A1/A2 (XIII-B): where does the READOUT token look, per layer and head?  Natural episodes (stream 75_100_000).
Per (layer, head): mass on QS (6 source tokens of q), OS (other source tokens), RO-tokens; per-group normalised mass on QS and on ALL of the group's
tokens (12 per group); Pearson correlation of the per-group mass with Bayes responsibility r_g (B), A1 responsibility, W_g alone, and with the
per-group perfect-agreement indicator pf_g; attention entropy.  Also per source token of q: which tokens does it attend to at layer 0
(own pair partner at the same question?  same source at other questions?).  -> logs/attn_profile<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
env = GL.Env(3, 1); rng = np.random.default_rng(int(os.environ.get("AP_STREAM", 75_100_000))); N_ = 2000
d = NV.Bm.sample_fast(NV.ENV, N_, rng); X = d["X"]; q = rng.integers(0, 6, N_); ii = np.arange(N_)
rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); lr = np.log(relm / noi); W = np.exp(lr.sum(1) - lr[ii, q]); Lg = rel[ii, q] / noi
a_ = X.reshape(N_, 6, 3, 2)[ii, q]; pf = (GL.counts(X, 3)[ii, q] == 5).astype(float); e = pf * (a_[..., 0] == a_[..., 1]); S = e.sum(1, keepdims=True)
nrm = lambda w: w / np.maximum(w.sum(1, keepdims=True), 1e-300)
R = dict(B=nrm(W * (S - e) * Lg.sum(-1)), A1=nrm(W * np.maximum(pf.sum(1, keepdims=True) - pf, 0) * Lg.sum(-1) + 1e-300), W=nrm(W), pf=pf, e=e)
corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
qs = q[:, None] * 6 + np.arange(6)[None]; grp_all = [np.array([qq * 6 + 2 * g + s for qq in range(6) for s in range(2)]) for g in range(3)]
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    rec = {}; F.forward(M, jnp.asarray(X), rec=rec); res = {}
    for l in range(3):
        A = np.asarray(rec[f"pattern{l}"])
        for h in range(4):
            row = A[ii, h, 36 + q]                                   # (N,43)
            mq = np.take_along_axis(row, qs, 1); pg_q = nrm(np.stack([mq[:, 2 * g:2 * g + 2].sum(1) for g in range(3)], 1) + 1e-12)
            pg_a = nrm(np.stack([row[:, grp_all[g]].sum(1) for g in range(3)], 1) + 1e-12)
            ent = float(-(row * np.log(row + 1e-12)).sum(1).mean())
            res[f"L{l}H{h}"] = dict(mass_QS=float(mq.sum(1).mean()), mass_OS=float(row[:, :36].sum(1).mean() - mq.sum(1).mean()), mass_RO=float(row[:, 36:].sum(1).mean()), entropy=ent,
                                   corr_QS={k: corr(pg_q, v) for k, v in R.items()}, corr_ALL={k: corr(pg_a, v) for k, v in R.items()})
    # layer-0: what does a source token attend to?
    A0 = np.asarray(rec["pattern0"]); prof = {}
    for h in range(4):
        tok = q * 6 + 0                                            # source 0 of group 0 at q (representative)
        row = A0[ii, h, tok]; partner = row[ii, tok + 1]; same_src_other_q = sum(row[ii, ((q + k) % 6) * 6 + 0] for k in range(1, 6)); own = row[ii, tok]
        prof[f"H{h}"] = dict(partner_same_q=float(partner.mean()), same_source_other_q=float(same_src_other_q.mean()), self=float(own.mean()), readout=float(row[:, 36:].sum(1).mean()))
    res["layer0_source_profile"] = prof; out[sp] = res
    print(sp[:6])
    for l in range(3):
        for h in range(4):
            r = res[f"L{l}H{h}"]; print(f"   L{l}H{h} QS {r['mass_QS']:.2f} OS {r['mass_OS']:.2f} RO {r['mass_RO']:.2f} | corrQS B {r['corr_QS']['B']:.2f} A1 {r['corr_QS']['A1']:.2f} W {r['corr_QS']['W']:.2f} | corrALL B {r['corr_ALL']['B']:.2f} W {r['corr_ALL']['W']:.2f} pf {r['corr_ALL']['pf']:.2f}")
    print("   L0 source profile", {h: {k: round(v, 2) for k, v in x.items()} for h, x in prof.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "attn_profile" + os.environ.get("AP_TAG", "") + ".json"), "w"), indent=1)
