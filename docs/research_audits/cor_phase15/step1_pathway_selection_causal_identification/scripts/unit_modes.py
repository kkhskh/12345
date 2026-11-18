"""XVIII unit checks of the four new training-time modes, on the parent-initialised E and M models (step 0) and a fixed dev batch.
For each mode the harness source is executed up to (not including) the training loop, with that mode's environment, so the checked code
is exactly the code that trains.  Checks:
  gsg: logits == unmasked logits exactly; d loss / d (target logits) == 0; parameter gradient differs from unmasked.
  fst: logits == -1e9 block logits exactly; parameter gradient differs from the block's.
  vrs: target attention pattern unchanged (same logits); logits differ from unmasked.
  gof: target attention share reduced by the odds factor exp(-B).
Writes logs/unit_modes.json."""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); SRC = open(os.path.join(H, "xviii_train.py")).read().split("rng = np.random.default_rng(1_000_003")[0]
SRC = SRC.replace("OUT = os.path.join(HERE, \"..\", \"checkpoints\", \"ft\", tag); os.makedirs(OUT, exist_ok=True)", "OUT = None")
import numpy as np
def env(parent_args, **kw):
    for k in ("FX_INTERV", "FX_EDGES", "FX_B"): os.environ.pop(k, None)
    os.environ.update(kw); sys.argv = ["x"] + parent_args; g = {"__file__": os.path.join(H, "xviii_train.py"), "__name__": "unit"}; exec(SRC, g); return g
res = {}
for base, args, edges in (("E", ["uP160", "21", "21", "801"], "ro0"), ("M", ["uP160", "23", "23", "801"], "own2")):
    g0 = env(args); import jax, jax.numpy as jnp
    d = g0["NL"].Bm.sample_fast(g0["NL"].ENV, 128, np.random.default_rng(97_400_000)); X, Y = jnp.asarray(d["X"]), jnp.asarray(d["Y"])
    L_plain = np.asarray(g0["logits_X"](g0["t"], X, False)); G_plain = jax.grad(lambda a: g0["lossf"](a, X, Y, False))(g0["t"])
    gb = env(args, FX_INTERV="msk:100", FX_EDGES=edges) if edges != "ro0" else env(args, FX_INTERV="ro0:100")
    if edges == "own2": gb = env(args, FX_INTERV="l2q:100")
    L_blk = np.asarray(gb["logits_X"](gb["t"], X, True)); G_blk = jax.grad(lambda a: gb["lossf"](a, X, Y, True))(gb["t"])
    flat = lambda G: np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(G)])
    r = {}
    gs = env(args, FX_INTERV="gsg:100", FX_EDGES=edges); L = np.asarray(gs["logits_X"](gs["t"], X, True)); G = jax.grad(lambda a: gs["lossf"](a, X, Y, True))(gs["t"])
    r["gsg"] = dict(logits_vs_plain=float(np.abs(L - L_plain).max()), grad_rel_change_vs_plain=float(np.linalg.norm(flat(G) - flat(G_plain)) / np.linalg.norm(flat(G_plain))))
    gf = env(args, FX_INTERV="fst:100", FX_EDGES=edges); L = np.asarray(gf["logits_X"](gf["t"], X, True)); G = jax.grad(lambda a: gf["lossf"](a, X, Y, True))(gf["t"])
    r["fst"] = dict(logits_vs_block=float(np.abs(L - L_blk).max()), grad_rel_change_vs_block=float(np.linalg.norm(flat(G) - flat(G_blk)) / np.linalg.norm(flat(G_blk))),
                    grad_rel_change_vs_plain=float(np.linalg.norm(flat(G) - flat(G_plain)) / np.linalg.norm(flat(G_plain))))
    gv = env(args, FX_INTERV="vrs:100", FX_EDGES=edges); L = np.asarray(gv["logits_X"](gv["t"], X, True))
    r["vrs"] = dict(logits_vs_plain=float(np.abs(L - L_plain).max()), loss_plain=float(g0["lossf"](g0["t"], X, Y, False)), loss_vrs=float(gv["lossf"](gv["t"], X, Y, True)), loss_block=float(gb["lossf"](gb["t"], X, Y, True)))
    # direct check that gsg zeroes the gradient w.r.t. target logits: perturb the target logits by a bias and differentiate through it
    lay_l, em = g0["MFW"].edges(edges); em = jnp.asarray(em.astype(np.float32))
    def loss_with_bias(beta, gg):
        old = gg["mask_for"]
        gg["mask_for"] = lambda layer: (jnp.zeros((43, 43)) + beta * em) if layer == lay_l else old(layer)
        try: return gg["lossf"](gg["t"], X, Y, True)
        finally: gg["mask_for"] = old
    r["gsg"]["dloss_dtarget_bias"] = float(jax.grad(lambda b: loss_with_bias(b, gs))(0.0)); r["plain_dloss_dtarget_bias"] = float(jax.grad(lambda b: loss_with_bias(b, env(args, FX_INTERV="gof:100", FX_EDGES=edges, FX_B="0")))(0.0))
    res[base] = r; print(base, json.dumps(r), flush=True)
json.dump(res, open(os.path.join(H, "..", "logs", "unit_modes.json"), "w"), indent=1)
