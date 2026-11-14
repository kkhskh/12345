"""Workstream B1: WHERE is the conjunction e = agree_q AND perfect computed?  (A linear probe decoding e is not evidence: AND is linearly separable.)
Interaction test.  For an additive (non-interacting) representation z = a*u_agree + p*u_perf + ..., the change of z caused by flipping agreement
at q does not depend on perfectness.  A representation in which the two ingredients INTERACT responds to the same agreement flip differently
when the group is perfect.  At each stage s (resid0, mid0 = resid0 + L0 attn, resid1, mid1 = resid1 + L1 attn, resid2), at the q7 tokens:
   I_s = || mean_pairs dz_s(D2) - mean_pairs dz_s(D2c) || / ( || mean dz_s(D2) || + || mean dz_s(D2c) || )         (agreement route)
computed in the per-group-aligned frame (pairs are aligned by placing the edited group's tokens in a canonical slot via the source ordering,
so means are over comparable coordinates), with D2: edited group perfect (n=5), D2c: edited group n<=3; both create agreement at q.
Also the PER-PAIR version (no averaging): ratio of the mean norm of the component of dz(D2) that is not explained by the best linear map from
dz(D2c)-type responses ... (omitted for simplicity; the mean-difference version is reported).  Random-split control: the same statistic
between two random halves of D2 (should be ~0 + noise).  Usage: stage_interaction.py SPEC...  -> logs/stage_interaction<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
EVP = os.environ.get("SI_EVAL", "dev"); ev = NV.load_eval() if EVP == "dev" else NV.load_eval(EVP)
X = ev["X"]; q = ev["q"]; n = len(X); idx = np.arange(n)
cnt = GL.counts(X, 3)[idx, q]; a_ = X.reshape(n, 6, 3, 2)[idx, q]; ag = a_[..., 0] == a_[..., 1]; rng = np.random.default_rng(1103)
def edits(kind, nmax=600):
    b, Xe, hs = [], [], []
    for i in range(n):
        cand = [h for h in range(3) if not ag[i, h] and ((cnt[i, h] == 5) if kind == "D2" else (cnt[i, h] <= 3))]
        if not cand: continue
        h = cand[rng.integers(len(cand))]; x = X[i].copy(); x[q[i], 2 * h + 1] = x[q[i], 2 * h]; b.append(i); Xe.append(x); hs.append(h)
        if len(b) >= nmax: break
    b = np.array(b); return X[b], np.array(Xe), q[b], np.array(hs)
fams = {"D2": edits("D2"), "D2c": edits("D2c")}
def canon(Z, qq, hh):
    """q7 tokens with the edited group's two source tokens first, then the other groups' (in order), then the readout token."""
    i = np.arange(len(qq)); rows = []
    for k in range(len(qq)):
        g = [hh[k]] + [x for x in range(3) if x != hh[k]]; toks = [qq[k] * 6 + 2 * gg + s for gg in g for s in (0, 1)] + [36 + qq[k]]; rows.append(toks)
    return Z[i[:, None], np.array(rows)].reshape(len(qq), -1)
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf); r = {}
    dz = {}
    for f, (Xb, Xa, qq, hh) in fams.items():
        rb, ra = {}, {}; F.forward(M, jnp.asarray(Xb), rec=rb); F.forward(M, jnp.asarray(Xa), rec=ra)
        st = lambda rr: {"resid0": np.asarray(rr["resid0"]), "mid0": np.asarray(rr["resid0"]) + np.asarray(rr["heads0"]).sum(1), "resid1": np.asarray(rr["resid1"]),
                         "mid1": np.asarray(rr["resid1"]) + np.asarray(rr["heads1"]).sum(1), "resid2": np.asarray(rr["resid2"])}
        sb, sa = st(rb), st(ra); dz[f] = {s: canon(sa[s], qq, hh) - canon(sb[s], qq, hh) for s in sb}
    for s in dz["D2"]:
        a, b = dz["D2"][s].mean(0), dz["D2c"][s].mean(0); I = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12))
        h1, h2 = dz["D2"][s][::2].mean(0), dz["D2"][s][1::2].mean(0); Ic = float(np.linalg.norm(h1 - h2) / (np.linalg.norm(h1) + np.linalg.norm(h2) + 1e-12))
        r[s] = dict(interaction=I, random_split_control=Ic, norm_D2=float(np.linalg.norm(a)), norm_D2c=float(np.linalg.norm(b)))
    out[sp] = r; print(sp, {s: (round(v["interaction"], 3), round(v["random_split_control"], 3)) for s, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "stage_interaction" + os.environ.get("SI_TAG", "") + ".json"), "w"), indent=1)
