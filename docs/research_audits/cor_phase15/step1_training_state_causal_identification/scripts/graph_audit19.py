"""XIX Stage 1: measured computational-graph audit of every training-time mode against plain training, at fixed parameters.
For each base (E: T = own0, layer 0; M: T = own2, layer 2), each audit state (step 0 = parent init; step 50 of the matched baseline run, with
its Adam state) and each mode, on a fixed batch of 256 episodes (stream 97_900_000):
  forward  : max |logits_mode - logits_plain|, loss difference, forward target share (readout rows, head mean)
  gradient : per parameter tensor, ||g_mode||, ||g_plain||, relative difference ||g_mode - g_plain|| / ||g_plain||, cosine
  logits   : d loss / d (bias on T edges) and d loss / d (bias on the complement edges of the same rows)
  update   : per tensor, relative difference of the one-step AdamW update from the step-50 optimizer state (state 'step50' only)
  optimizer: the same for the updated Adam first/second moments (mu, nu)
Writes logs/graph_audit19.json (machine-readable table) and prints a summary.  Usage: graph_audit19.py"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); SRC = open(os.path.join(H, "xix_train.py")).read().split("rng = np.random.default_rng(1_000_003")[0]
SRC = SRC.replace('OUT = os.path.join(HERE, "..", "checkpoints", "ft", tag); os.makedirs(OUT, exist_ok=True)', "OUT = None")
import numpy as np, pickle
MODES = {"block": dict(FX_INTERV="gof:100", FX_B="1e9"), "gof2.3": dict(FX_INTERV="gof:100", FX_B="2.3"), "gsg": dict(FX_INTERV="gsg:100"),
         "gsv": dict(FX_INTERV="gsv:100"), "vsg": dict(FX_INTERV="vsg:100"), "vsgo": dict(FX_INTERV="vsgo:100"), "vrg": dict(FX_INTERV="vrg:100"),
         "vrs": dict(FX_INTERV="vrs:100"), "fst": dict(FX_INTERV="fst:100")}
def base_modes(base):     # v2 (pre-review B5 / A8): parameter freezes, parameter noise and the development same-row controls
    P = os.path.join(H, "..", "checkpoints", "ft", f"uP160_fx_L{21 if base == 'E' else 23}_W{21 if base == 'E' else 23}_D801")
    m = dict(MODES); fz = "L0_QK" if base == "E" else "L2_QK"
    m["frz"] = dict(FX_INTERV="frz:100", FX_FREEZE=fz); m["frzp"] = dict(FX_INTERV="frzp:100", FX_FREEZE=fz)
    m["pnz"] = dict(FX_INTERV="pnz:100", FX_PNZ_REF=f"{P}_stop100_s100dev:{P}_gof{'own0' if base == 'E' else 'own2'}b1e9T100_stop100_s100dev", FX_PNZ_SEED="1")
    m["lbl"] = dict(FX_INTERV="lbl:100", FX_LBL_P="0.5", FX_LBL_SEED="1")
    m["ctl_same_row"] = dict(FX_INTERV="msk:100", FX_EDGES="oq0") if base == "E" else dict(FX_INTERV="gof:100", FX_EDGES="oq2", FX_B="1.0")
    return m
def env(args, **kw):
    for k in list(os.environ):
        if k.startswith("FX_"): os.environ.pop(k)
    os.environ.update(kw); sys.argv = ["x"] + args; g = {"__file__": os.path.join(H, "xix_train.py"), "__name__": "audit"}; exec(SRC, g); return g
def names(t, jax):
    return ["/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in p) for p, _ in jax.tree_util.tree_flatten_with_path(t)[0]]
out = {}
for base, args, edges in (("E", ["uP160", "21", "21", "801"], "own0"), ("M", ["uP160", "23", "23", "801"], "own2")):
    g0 = env(args); import jax, jax.numpy as jnp, optax
    d = g0["NL"].Bm.sample_fast(g0["NL"].ENV, 256, np.random.default_rng(97_900_000)); X, Y = jnp.asarray(d["X"]), jnp.asarray(d["Y"])
    nm = names(g0["t"], jax); lay_l, em = g0["MFW"].edges(edges); em = jnp.asarray(em.astype(np.float32)); rowmask = (em.max(-1, keepdims=True) > 0) * (1 - em)
    states = {"step0": (g0["t"], None)}
    bdir = os.path.join(H, "..", "checkpoints", "ft", f"uP160_fx_L{args[1]}_W{args[2]}_D801_stop100_s100dev")
    if os.path.exists(os.path.join(bdir, "opt_step50.pkl")):
        z = np.load(os.path.join(bdir, "step50.npz")); t50 = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(g0["t"]), [jnp.asarray(z[f"arr_{k}"]) for k in range(len(z.files))])
        R = pickle.load(open(os.path.join(bdir, "opt_step50.pkl"), "rb")); st50 = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(g0["st"]), [jnp.asarray(a) for a in R["opt"]])
        states["step50"] = (t50, st50)
    res = {}
    for sname, (tt, st) in states.items():
        Lp = np.asarray(g0["logits_X"](tt, X, False)); lp = float(g0["lossf"](tt, X, Y, False)); Gp = jax.grad(lambda a: g0["lossf"](a, X, Y, False))(tt)
        if st is not None: Up, Sp = g0["opt"].update(Gp, st, tt)
        st0 = st
        rr = {}
        if st is None: st = g0["opt"].init(tt)                       # v2: step-0 update from a fresh AdamW state
        t2p, s2p, _ = g0["step_plain"](tt, st, X, Y); Up_ = [np.asarray(a) - np.asarray(b) for a, b in zip(jax.tree_util.tree_leaves(t2p), jax.tree_util.tree_leaves(tt))]
        for mname, kw in base_modes(base).items():
            gm = env(args, **dict(dict(FX_EDGES=edges), **kw)); i50 = jnp.int32(50)
            Yu = Y                                                     # v2: lbl is audited with its noised labels (pre-review B R2-2)
            if mname == "lbl":
                _r = np.random.default_rng(1); _Yn = np.asarray(Y).copy(); _mk = _r.random(_Yn.shape) < 0.5; _Yn[_mk] = _r.integers(0, int(g0["t"]["WA"].shape[1]), int(_mk.sum())); Yu = jnp.asarray(_Yn)
            Lm = np.asarray(gm["logits_X"](tt, X, True, i50)); lm = float(gm["lossf"](tt, X, Yu, True, i50)); Gm = jax.grad(lambda a: gm["lossf"](a, X, Yu, True, i50))(tt)
            rec = dict(forward_max_logit_diff=float(np.abs(Lm - Lp).max()), loss_diff=lm - lp, tensors={})
            for n_, a, b in zip(nm, jax.tree_util.tree_leaves(Gm), jax.tree_util.tree_leaves(Gp)):
                a, b = np.asarray(a).ravel(), np.asarray(b).ravel(); nb = float(np.linalg.norm(b)) + 1e-30
                rec["tensors"][n_] = dict(norm_mode=float(np.linalg.norm(a)), norm_plain=float(np.linalg.norm(b)), rel_diff=float(np.linalg.norm(a - b)) / nb, cos=float(a @ b / (np.linalg.norm(a) * nb + 1e-30)))
            old = gm["mask_for"]
            def lb(beta, m):
                def mf(layer):
                    o = old(layer)
                    return ((jnp.zeros((43, 43)) if o is None else o) + beta * m) if layer == lay_l else o
                gm["mask_for"] = mf
                try: return gm["lossf"](tt, X, Yu, True, i50)
                finally: gm["mask_for"] = old
            rec["dloss_dbias_T"] = float(jax.grad(lambda b_: lb(b_, em))(0.0)); rec["dloss_dbias_complement"] = float(jax.grad(lambda b_: lb(b_, rowmask))(0.0))
            # v2: the mode's ACTUAL training step (includes freezes); per-tensor update and per-tensor mu / nu differences from plain
            t2m, s2m, _ = gm["step_mask"](tt, st, X, Yu, i50); Um_ = [np.asarray(a) - np.asarray(b) for a, b in zip(jax.tree_util.tree_leaves(t2m), jax.tree_util.tree_leaves(tt))]
            rec["step_update_rel_diff"] = {n_: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)) for n_, a, b in zip(nm, Um_, Up_)}
            mo = [e for e in s2m if hasattr(e, "mu")][0]; po = [e for e in s2p if hasattr(e, "mu")][0]
            for fld in ("mu", "nu"):
                rec[f"step_{fld}_rel_diff"] = {n_: float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (np.linalg.norm(np.asarray(b)) + 1e-30)) for n_, a, b in zip(nm, jax.tree_util.tree_leaves(getattr(mo, fld)), jax.tree_util.tree_leaves(getattr(po, fld)))}
            if mname == "pnz": rec["pnz_noise_norm_over_plain_update"] = {n_: float(gm["PNZ_SIG"][k] * np.sqrt(Up_[k].size) / (np.linalg.norm(Up_[k]) + 1e-30)) for k, n_ in enumerate(nm)}
            if sname == "step50":
                Um, Sm = gm["opt"].update(Gm, st, tt); rec["update_rel_diff"] = {n_: float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (np.linalg.norm(np.asarray(b)) + 1e-30)) for n_, a, b in zip(nm, jax.tree_util.tree_leaves(Um), jax.tree_util.tree_leaves(Up))}
                lm_, lp_ = jax.tree_util.tree_leaves(Sm), jax.tree_util.tree_leaves(Sp)
                rec["optstate_rel_diff_max"] = float(max(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (np.linalg.norm(np.asarray(b)) + 1e-30) for a, b in zip(lm_, lp_) if np.asarray(b).size > 1))
            rr[mname] = rec
        res[sname] = rr
    out[base] = dict(target_layer=int(lay_l), states=res, tensor_names=nm)
    for sname, rr in res.items():
        for mname, rec in rr.items():
            ch = [n_ for n_, v in rec["tensors"].items() if v["rel_diff"] > 1e-5]
            print(base, sname, mname.ljust(7), "fwd", f"{rec['forward_max_logit_diff']:.1e}", "dL", f"{rec['loss_diff']:+.4f}", "dT", f"{rec['dloss_dbias_T']:+.4f}", "dC", f"{rec['dloss_dbias_complement']:+.4f}",
                  "changed grads:", len(ch), sorted(set(c.split("/")[0] + ("/" + c.split("/")[1] if c.split("/")[0] in ("pre", "new") else "") for c in ch)), flush=True)
json.dump(out, open(os.path.join(H, "..", "logs", "graph_audit19.json"), "w"), indent=1)
