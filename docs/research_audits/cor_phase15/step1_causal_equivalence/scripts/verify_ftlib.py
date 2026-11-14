"""Check ftlib.forward == Investigation IX evaluation logits for the stored full_c1 / c1 models (max abs diff), and patch-identity checks."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, ftlib as F, native as NV
ev = NV.load_eval(); X = ev["X"][:500]; out = {}
for parent, spec, suf in (("xP81", "full_c1", ""), ("yP130", "full_c1", "_eval_conf9"), ("xP81", "c1", "")):
    M = F.load_model(parent, spec, suffix=suf); L = np.asarray(F.forward(M, jnp.asarray(X)))
    # reference: Investigation IX path (nnlib.block, N.embed)
    x = NV.N.embed(dict(M["emb"]), jnp.asarray(X))
    for l, lay in enumerate(M["layers"]): x = NV.NL.block(lay, x, l=l if l < 2 else 10)
    R = np.asarray(NV.N.ln(x, M["gf"], M["bf"])[:, 36:42] @ M["WA"])
    rec = {}; F.forward(M, jnp.asarray(X), rec=rec)
    hp = {(1, 2): (jnp.ones((500, 43), bool), rec["heads1"][:, 2])}                  # patch head with itself -> identity
    L2 = np.asarray(F.forward(M, jnp.asarray(X), head_patch=hp))
    out[M["tag"]] = dict(max_diff_vs_nnlib=float(np.abs(L - R).max()), self_patch_diff=float(np.abs(L2 - L).max()))
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "..", "logs", "verify_ftlib.json"), "w"), indent=1)
