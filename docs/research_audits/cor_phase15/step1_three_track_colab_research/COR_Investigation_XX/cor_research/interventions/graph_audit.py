"""Measured computational-graph audit of the XX training-time interventions (adapted from XIX scripts/graph_audit19.py to the XX harness).
python -m cor_research.interventions.graph_audit OUT.json [PARENT LSEED_E LSEED_M D]   (default uP160 21 23 801; parent init state, step 0)
For each base (E: T = own0 at layer 0; M: T = own2 at layer 2) and mode, on a fixed batch of 256 episodes (stream 97_900_000):
  forward_max_logit_diff vs plain; dloss/dbias on the T edges and on the complement edges of the same readout rows (target layer);
  per-tensor gradient relative difference vs plain.
Every mode also reports the tensors that receive EXACTLY zero gradient (review A m9; e.g. rsg on M freezes layer-2 WQ/WK).
Expected (asserted in `checks`):  rsg  forward identical; dL/dbias_T = 0 AND dL/dbias_complement = 0 (whole readout rows: no routing
gradient); gradients still change (value / other-path learning intact elsewhere is NOT asserted, it is reported)
  vsgo forward identical; dL/dbias_T != 0 (routing gradient intact);   gsg forward identical; dL/dbias_T = 0, complement != 0;
  block (gof 1e9) forward changes;   lbl forward identical (labels change, not the graph)."""
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E
E.configure(); E.legacy_paths()
import numpy as np
SRC = open(E.HARNESS).read().split("rng = np.random.default_rng(1_000_003")[0]
SRC = SRC.replace('OUT = os.path.join(RUNROOT, "ft", tag); os.makedirs(OUT, exist_ok=True)', "OUT = None")
MODES = {"block": dict(FX_INTERV="gof:100", FX_B="1e9"), "gsg": dict(FX_INTERV="gsg:100"), "vsgo": dict(FX_INTERV="vsgo:100"), "rsg": dict(FX_INTERV="rsg:100"),
         "lbl": dict(FX_INTERV="lbl:100", FX_LBL_P="0.5", FX_LBL_SEED="1")}


def env(args, **kw):
    for k in list(os.environ):
        if k.startswith("FX_"): os.environ.pop(k)
    os.environ.update(kw); sys.argv = ["x"] + args; g = {"__file__": E.HARNESS, "__name__": "audit"}; exec(SRC, g); return g


def names(t, jax): return ["/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in p) for p, _ in jax.tree_util.tree_flatten_with_path(t)[0]]


def audit(parent="uP160", lE="21", lM="23", D="801"):
    E.patch_parents(); out = {}
    for base, L, edges in (("E", lE, "own0"), ("M", lM, "own2")):
        args = [parent, L, L, D]; g0 = env(args); import jax, jax.numpy as jnp
        d = g0["NL"].Bm.sample_fast(g0["NL"].ENV, 256, np.random.default_rng(97_900_000)); X, Y = jnp.asarray(d["X"]), jnp.asarray(d["Y"])
        nm = names(g0["t"], jax); lay_l, em = g0["MFW"].edges(edges); em = jnp.asarray(em.astype(np.float32)); rowmask = (em.max(-1, keepdims=True) > 0) * (1 - em)
        tt = g0["t"]; Lp = np.asarray(g0["logits_X"](tt, X, False)); Gp = jax.grad(lambda a: g0["lossf"](a, X, Y, False))(tt); res = {}
        for mname, kw in MODES.items():
            gm = env(args, **dict(dict(FX_EDGES=edges), **kw)); i50 = jnp.int32(50); Yu = Y
            if mname == "lbl":
                r_ = np.random.default_rng(1); Yn = np.asarray(Y).copy(); mk = r_.random(Yn.shape) < 0.5; Yn[mk] = r_.integers(0, int(tt["WA"].shape[1]), int(mk.sum())); Yu = jnp.asarray(Yn)
            Lm = np.asarray(gm["logits_X"](tt, X, True, i50)); Gm = jax.grad(lambda a: gm["lossf"](a, X, Yu, True, i50))(tt)
            rec = dict(forward_max_logit_diff=float(np.abs(Lm - Lp).max()), tensors_changed=[], zero_gradient_tensors=[])
            for n_, a, b in zip(nm, jax.tree_util.tree_leaves(Gm), jax.tree_util.tree_leaves(Gp)):
                a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
                if np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30) > 1e-5: rec["tensors_changed"].append(n_)
                if np.linalg.norm(a) == 0 and np.linalg.norm(b) > 0: rec["zero_gradient_tensors"].append(n_)   # review A m9: report exact zero gradients
            old = gm["mask_for"]
            def lb(beta, m):
                def mf(layer):
                    o = old(layer); return ((jnp.zeros((43, 43)) if o is None else o) + beta * m) if layer == lay_l else o
                gm["mask_for"] = mf
                try: return gm["lossf"](tt, X, Yu, True, i50)
                finally: gm["mask_for"] = old
            rec["dloss_dbias_T"] = float(jax.grad(lambda b_: lb(b_, em))(0.0)); rec["dloss_dbias_complement"] = float(jax.grad(lambda b_: lb(b_, rowmask))(0.0)); res[mname] = rec
        out[base] = dict(target_layer=int(lay_l), modes=res)
    z = lambda v: abs(v) < 1e-7
    ck = {}
    for b, o in out.items():
        m = o["modes"]
        ck[f"{b}:rsg_forward_identical"] = m["rsg"]["forward_max_logit_diff"] < 1e-6; ck[f"{b}:rsg_no_routing_gradient_T"] = z(m["rsg"]["dloss_dbias_T"]); ck[f"{b}:rsg_no_routing_gradient_complement"] = z(m["rsg"]["dloss_dbias_complement"])
        ck[f"{b}:vsgo_forward_identical"] = m["vsgo"]["forward_max_logit_diff"] < 1e-5   # float32 re-association of the split T / complement sum (~1e-6)
        ck[f"{b}:vsgo_routing_gradient_intact"] = not z(m["vsgo"]["dloss_dbias_T"])
        ck[f"{b}:gsg_T_zero_complement_nonzero"] = z(m["gsg"]["dloss_dbias_T"]) and not z(m["gsg"]["dloss_dbias_complement"])
        ck[f"{b}:block_changes_forward"] = m["block"]["forward_max_logit_diff"] > 1e-3; ck[f"{b}:lbl_forward_identical"] = m["lbl"]["forward_max_logit_diff"] < 1e-6
    zg = {b: {m: o["modes"][m]["zero_gradient_tensors"] for m in o["modes"]} for b, o in out.items()}
    ck["M:rsg_zero_grad_layer2_QK_disclosed"] = any("WQ" in t or "WK" in t for t in zg["M"]["rsg"])   # rsg on M = layer-2 QK frozen (review A B2): must be visible
    return dict(audit=out, checks=ck, all_pass=all(ck.values()), zero_gradient_tensors=zg)


if __name__ == "__main__":
    dst = sys.argv[1]; a = sys.argv[2:6] if len(sys.argv) >= 6 else []          # env() rewrites sys.argv: read it first
    r = audit(*a) if a else audit(); E.atomic_json(r, dst); print(json.dumps(r["checks"], indent=1)); sys.exit(0 if r["all_pass"] else 1)
