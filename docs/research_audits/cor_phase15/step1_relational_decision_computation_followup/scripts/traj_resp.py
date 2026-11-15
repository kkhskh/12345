"""Workstream D4 (XIII-B, exploratory): which candidate-evidence WEIGHTING emerges when, along full_c1 training?  Input-intervention (class I)
response laws at every saved checkpoint, on relational pairs (relpairs.relational_pairs, stream 75_600_000), split into
  W-only pairs (the edit changes the agreement count W_h but no eligibility e_g)  and  e-change pairs (the edit flips some e_g);
response slope = fidelity slope of the model's change on the Bayes change (1 = correct weighting, 0 = insensitive).  Also KL(Bayes || model) on
natural episodes and the D1/D2 behavioural gap (distance-to-A2 minus distance-to-B; algorithm_discrimination families on eval_conf13).
Question: does correct eligibility weighting (e-change slope) or correct reliability weighting (W-only slope) lead the behavioural A2 -> B
separation?  t50 = first checkpoint reaching half of the final rise.  -> logs/traj_resp<TAG>.json"""
import json, os, sys, re, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
src = open(os.path.join(HERE, "algorithm_discrimination.py")).read(); src = src[:src.index("out = dict(n=")]
sys.argv_ = sys.argv; sys.argv = ["x", "logs/eval_conf13.npz"]; g = {"__file__": os.path.join(HERE, "algorithm_discrimination.py"), "__name__": "alg"}; exec(compile(src, "alg[def]", "exec"), g); sys.argv = sys.argv_
fams, pred = g["fams"], g["pred"]
rng = np.random.default_rng(75_600_000); PP = RP.relational_pairs(1500, rng); iP = np.arange(len(PP["q"])); dB = RP.cen(PP["PBd"]) - RP.cen(PP["PB"]); ech = PP["e_change"]
Xn = NV.Bm.sample_fast(NV.ENV, 1500, rng)["X"]; qn = rng.integers(0, 6, 1500); PBn = GL.posterior_closed(RP.ENV1, Xn, qn)
out = {}
for sp in sys.argv[1:]:
    parent, spec, seed, suf = sp.split(":"); D = F.model_dir(f"{parent}_{spec}_s4000_seed{seed}{suf}")
    steps = sorted(int(re.findall(r"step(\d+)", os.path.basename(f))[0]) for f in glob.glob(os.path.join(D, "step*.npz"))); rows = []
    for k in steps:
        M = F.load_model(parent, spec, seed=int(seed), suffix=suf, step=k); fj = jax.jit(lambda X_: F.forward(M, X_))
        Lb = np.asarray(F.batched(fj, PP["Xb"]))[iP, PP["q"]]; Ld = np.asarray(F.batched(fj, PP["Xd"]))[iP, PP["q"]]; o = RP.cen(RP.sm(Ld)) - RP.cen(RP.sm(Lb))
        Ln = np.asarray(F.batched(fj, Xn))[np.arange(1500), qn]; P = RP.sm(Ln)
        r = dict(step=k, slope_W=RP.fid(o[~ech], dB[~ech])["slope"], slope_e=RP.fid(o[ech], dB[ech])["slope"], KL=float((PBn * (np.log(PBn + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()))
        gaps = []
        for f in ("D1", "D2"):
            Xb, Xa, qq = fams[f]; i = np.arange(len(qq)); D_ = F.centred(F.batched(fj, Xa)[i, qq]) - F.centred(F.batched(fj, Xb)[i, qq])
            gaps.append(float(np.abs(D_ - pred[f]["A2"]).mean() - np.abs(D_ - pred[f]["B"]).mean()))
        r["gap"] = float(np.mean(gaps)); rows.append(r)
    def t50(v, up=True):
        v = np.array(v); lo, hi = v[0], v[-1]
        if abs(hi - lo) < 1e-6: return None
        return int(steps[int(np.argmax((v - lo) / (hi - lo) >= 0.5))])
    out[sp] = dict(rows=rows, t50={k: t50([r[k] for r in rows]) for k in ("slope_W", "slope_e", "KL", "gap")})
    print(sp[:6], out[sp]["t50"], [(r["step"], round(r["slope_W"], 2), round(r["slope_e"], 2), round(r["KL"], 3), round(r["gap"], 3)) for r in rows][2:9], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "traj_resp" + os.environ.get("TR_TAG", "") + ".json"), "w"), indent=1)
