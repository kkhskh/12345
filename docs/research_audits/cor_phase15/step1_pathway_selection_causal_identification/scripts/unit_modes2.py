"""XVIII unit checks of the modes added after pre-review B (gsv, shc, vrg), on the parent-initialised E (own0 edges) and M (own2) models.
  gsv: logits == unmasked logits; gradient differs from plain AND from gsg (value gradient through T removed as well).
  (the constant-c share clamp 'shc' was superseded by the adaptive clamp 'sha', checked in unit_modes3.py / unit_sha_far.py)
  vrg: logits == vrs logits exactly; gradient differs from vrs.
Writes logs/unit_modes2.json."""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); SRC = open(os.path.join(H, "xviii_train.py")).read().split("rng = np.random.default_rng(1_000_003")[0]
SRC = SRC.replace("OUT = os.path.join(HERE, \"..\", \"checkpoints\", \"ft\", tag); os.makedirs(OUT, exist_ok=True)", "OUT = None")
import numpy as np
def env(args, **kw):
    for k in ("FX_INTERV", "FX_EDGES", "FX_B"): os.environ.pop(k, None)
    os.environ.update(kw); sys.argv = ["x"] + args; g = {"__file__": os.path.join(H, "xviii_train.py"), "__name__": "unit"}; exec(SRC, g); return g
res = {}
for base, args, edges in (("E", ["uP160", "21", "21", "801"], "own0"), ("M", ["uP160", "23", "23", "801"], "own2")):
    g0 = env(args); import jax, jax.numpy as jnp
    d = g0["NL"].Bm.sample_fast(g0["NL"].ENV, 128, np.random.default_rng(97_400_000)); X, Y = jnp.asarray(d["X"]), jnp.asarray(d["Y"])
    flat = lambda G: np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(G)])
    gr = lambda g, m=True: jax.grad(lambda a: g["lossf"](a, X, Y, m))(g["t"]); lg = lambda g, m=True: np.asarray(g["logits_X"](g["t"], X, m))
    L0, G0 = lg(g0, False), flat(gr(g0, False))
    gs = env(args, FX_INTERV="gsg:100", FX_EDGES=edges); Gs = flat(gr(gs))
    gv = env(args, FX_INTERV="gsv:100", FX_EDGES=edges); Gv = flat(gr(gv))
    r = dict(gsv=dict(logits_vs_plain=float(np.abs(lg(gv) - L0).max()), grad_rel_vs_plain=float(np.linalg.norm(Gv - G0) / np.linalg.norm(G0)), grad_rel_vs_gsg=float(np.linalg.norm(Gv - Gs) / np.linalg.norm(Gs))))
    gvr = env(args, FX_INTERV="vrs:100", FX_EDGES=edges); gvg = env(args, FX_INTERV="vrg:100", FX_EDGES=edges)
    r["vrg"] = dict(logits_vs_vrs=float(np.abs(lg(gvg) - lg(gvr)).max()), grad_rel_vs_vrs=float(np.linalg.norm(flat(gr(gvg)) - flat(gr(gvr))) / np.linalg.norm(flat(gr(gvr)))))
    res[base] = r; print(base, json.dumps(r), flush=True)
json.dump(res, open(os.path.join(H, "..", "logs", "unit_modes2.json"), "w"), indent=1)
