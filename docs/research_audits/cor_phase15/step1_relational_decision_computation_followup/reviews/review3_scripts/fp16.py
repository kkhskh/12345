"""Reviewer check: adapt2 layer-2-side arms train/evaluate on float16-stored resid2; early01 arms use float32 end-to-end. Size of the precision loss."""
import os, sys
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation_followup"
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, genlib as GL, relpairs as RP
for sp in sys.argv[1:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    X = GL.sample(GL.Env(3, 2, beta=0.6), 1500, np.random.default_rng(1))["X"]
    r = {}; Lfull = np.asarray(F.forward(M, jnp.asarray(X), rec=r)); z = np.asarray(r["resid2"])
    def head(zz):
        x = F.block_heads(M["layers"][2], jnp.asarray(zz), None, 2, None, None); return np.asarray(F.N.ln(x, M["gf"], M["bf"])[:, 36:42] @ M["WA"])
    L32 = head(z); L16 = head(z.astype(np.float16).astype(np.float32))
    c = lambda L: RP.cen(RP.sm(L))
    print(sp[:6], "resid2 |z| mean", round(float(np.abs(z).mean()), 3), "max |dz| fp16", float(np.abs(z - z.astype(np.float16)).max()),
          "max |dcenlogp| head32 vs full", float(np.abs(c(L32) - c(Lfull)).max()), "fp16 vs fp32: mean", float(np.abs(c(L16) - c(L32)).mean()), "max", float(np.abs(c(L16) - c(L32)).max()),
          "decision flips", float((L16.argmax(-1) != L32.argmax(-1)).mean()))
