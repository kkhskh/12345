"""XVIII unit checks of the modes added after pre-review B (gsv, shc, vrg), on the parent-initialised E (own0 edges) and M (own2) models.
  gsv: logits == unmasked logits; gradient differs from plain AND from gsg (value gradient through T removed as well).
  shc: target attention mass in every target row == FX_SHC; logits differ from plain only through the rescaling.
  vrg: logits == vrs logits exactly; gradient differs from vrs.
Writes logs/unit_modes2.json."""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); SRC = open(os.path.join(H, "xviii_train.py")).read().split("rng = np.random.default_rng(1_000_003")[0]
SRC = SRC.replace("OUT = os.path.join(HERE, \"..\", \"checkpoints\", \"ft\", tag); os.makedirs(OUT, exist_ok=True)", "OUT = None")
import numpy as np
def env(args, **kw):
    for k in ("FX_INTERV", "FX_EDGES", "FX_B", "FX_SHC"): os.environ.pop(k, None)
    os.environ.update(kw); sys.argv = ["x"] + args; g = {"__file__": os.path.join(H, "xviii_train.py"), "__name__": "unit"}; exec(SRC, g); return g
res = {}
for base, args, edges, c in (("E", ["uP160", "21", "21", "801"], "own0", "0.134"), ("M", ["uP160", "23", "23", "801"], "own2", "0.141")):
    g0 = env(args); import jax, jax.numpy as jnp
    d = g0["NL"].Bm.sample_fast(g0["NL"].ENV, 128, np.random.default_rng(97_400_000)); X, Y = jnp.asarray(d["X"]), jnp.asarray(d["Y"])
    flat = lambda G: np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(G)])
    gr = lambda g, m=True: jax.grad(lambda a: g["lossf"](a, X, Y, m))(g["t"]); lg = lambda g, m=True: np.asarray(g["logits_X"](g["t"], X, m))
    L0, G0 = lg(g0, False), flat(gr(g0, False))
    gs = env(args, FX_INTERV="gsg:100", FX_EDGES=edges); Gs = flat(gr(gs))
    gv = env(args, FX_INTERV="gsv:100", FX_EDGES=edges); Gv = flat(gr(gv))
    r = dict(gsv=dict(logits_vs_plain=float(np.abs(lg(gv) - L0).max()), grad_rel_vs_plain=float(np.linalg.norm(Gv - G0) / np.linalg.norm(G0)), grad_rel_vs_gsg=float(np.linalg.norm(Gv - Gs) / np.linalg.norm(Gs))))
    gc = env(args, FX_INTERV="shc:100", FX_EDGES=edges, FX_SHC=c)
    # target mass after clamping, recomputed from the clamped weights of the intervened layer
    lay_l, em = gc["MFW"].edges(edges); rec = {}
    x = gc["N"].embed(dict(gc["t"]["emb"]), X)
    lays = list(gc["t"]["pre"]) + list(gc["t"]["new"])
    for l in range(lay_l): x = gc["NL"].block(lays[l], x, l=l if l < 2 else 10)
    lay = lays[lay_l]; z = gc["N"].ln(x, lay["g1"], lay["b1"]); q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"])
    a = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1]), -1); XM = jnp.asarray(em.astype(np.float32)); mT = (a * XM).sum(-1)[:, :, 36:42]
    cc = float(c); a2 = a * XM * (cc / (a * XM).sum(-1, keepdims=True)) + a * (1 - XM) * ((1 - cc) / (1 - (a * XM).sum(-1, keepdims=True)))
    r["shc"] = dict(target_mass_before_mean=float(mT.mean()), target_mass_before_range=[float(mT.min()), float(mT.max())], target_mass_after=float((a2 * XM).sum(-1)[:, :, 36:42].mean()),
                    logits_vs_plain=float(np.abs(lg(gc) - L0).max()), rows_sum_to_1=float(np.abs(a2.sum(-1) - 1).max()))
    gvr = env(args, FX_INTERV="vrs:100", FX_EDGES=edges); gvg = env(args, FX_INTERV="vrg:100", FX_EDGES=edges)
    r["vrg"] = dict(logits_vs_vrs=float(np.abs(lg(gvg) - lg(gvr)).max()), grad_rel_vs_vrs=float(np.linalg.norm(flat(gr(gvg)) - flat(gr(gvr))) / np.linalg.norm(flat(gr(gvr)))))
    res[base] = r; print(base, json.dumps(r), flush=True)
json.dump(res, open(os.path.join(H, "..", "logs", "unit_modes2.json"), "w"), indent=1)
