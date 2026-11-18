"""XVIII unit checks after pre-review A: adaptive share clamp (sha) and the logged block (gof with B = 1e9).
  sha: per-head forward target share equals the schedule A*_h(i) within 1e-3 (i = 0 and i = 99 of the baseline slog, on that step's batch
       is not available here, so a fixed dev batch with the schedule row is used); beta finite; d loss / d (target logits) nonzero;
       the rolled-content variant differs from real content only on target edges (C-only rows unchanged).
  gof B=1e9: logits identical to the -1e9 block (msk own0 for E; l2q for M).
Writes logs/unit_modes3.json."""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); SRC = open(os.path.join(H, "xviii_train.py")).read().split("rng = np.random.default_rng(1_000_003")[0]
SRC = SRC.replace("OUT = os.path.join(HERE, \"..\", \"checkpoints\", \"ft\", tag); os.makedirs(OUT, exist_ok=True)", "OUT = None")
import numpy as np
def env(args, **kw):
    for k in ("FX_INTERV", "FX_EDGES", "FX_B", "FX_SHA_SCHED", "FX_SHA_LO", "FX_SHA_CONTENT", "FX_GLOG", "FX_SLOG"): os.environ.pop(k, None)
    os.environ.update(kw); sys.argv = ["x"] + args; g = {"__file__": os.path.join(H, "xviii_train.py"), "__name__": "unit"}; exec(SRC, g); return g
res = {}
for base, args, edges, blk in (("E", ["uP160", "21", "21", "801"], "own0", dict(FX_INTERV="msk:100", FX_EDGES="own0")), ("M", ["uP160", "23", "23", "801"], "own2", dict(FX_INTERV="l2q:100"))):
    sched = os.path.join(H, "..", "checkpoints", "ft", f"uP160_fx_L{args[1]}_W{args[2]}_D801_stop100_slog", "slog.json")
    g0 = env(args); import jax, jax.numpy as jnp
    d = g0["NL"].Bm.sample_fast(g0["NL"].ENV, 256, np.random.default_rng(97_400_000)); X, Y = jnp.asarray(d["X"]), jnp.asarray(d["Y"])
    r = {}
    gb = env(args, **blk); gg = env(args, FX_INTERV="gof:100", FX_EDGES=edges, FX_B="1e9")
    r["gof1e9_vs_block_logits"] = float(np.abs(np.asarray(gg["logits_X"](gg["t"], X, True)) - np.asarray(gb["logits_X"](gb["t"], X, True))).max())
    for lo in ("0", "1"):
        for content in ("real", "rolled"):
            gs = env(args, FX_INTERV="sha:100", FX_EDGES=edges, FX_SHA_SCHED=sched, FX_SHA_LO=lo, FX_SHA_CONTENT=content)
            lay_l = gs["MFW"].edges(edges)[0]; XMj = gs["XM"]; out = {}
            for i in (0, 99):
                sraw = gs["_att_at"](gs["t"], X, lay_l, True, jnp.int32(i)); b = gs["sha_beta"](sraw, jnp.int32(i)); a = jax.nn.softmax(sraw + b[None, :, None, None] * XMj, -1)
                ach = np.asarray((a[:, :, 36:42] * XMj[None, None, 36:42]).sum(-1).mean((0, 2))); tgt = np.asarray(gs["SHA_A"][i])
                out[f"step{i}"] = dict(target=tgt.round(4).tolist(), achieved=ach.round(4).tolist(), max_abs_err=float(np.abs(ach - tgt).max()), beta=np.asarray(b).round(3).tolist())
            em = jnp.asarray(gs["MFW"].edges(edges)[1].astype(np.float32))
            old = gs["mask_for"]
            def lb(beta):
                gs["mask_for"] = lambda layer: (jnp.zeros((43, 43)) + beta * em) if layer == lay_l else old(layer)
                try: return gs["lossf"](gs["t"], X, Y, True, jnp.int32(50))
                finally: gs["mask_for"] = old
            out["dloss_dtarget_logit_bias"] = float(jax.grad(lb)(0.0))
            r[f"sha_lo{lo}_{content}"] = out
    # rolled content changes only T-edge reads: outputs on rows that have no T edges (source tokens) unchanged at the intervened layer
    res[base] = r; print(base, json.dumps(r)[:900], flush=True)
json.dump(res, open(os.path.join(H, "..", "logs", "unit_modes3.json"), "w"), indent=1)
