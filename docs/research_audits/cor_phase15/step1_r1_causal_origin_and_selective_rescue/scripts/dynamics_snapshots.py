"""Learning dynamics on Investigation VI's archived transfer snapshots (steps 250-4000; plus step 0 = pretrained layers with the
freshly initialised layer 2 / inherited readout, i.e. the transfer starting point): behaviour on the dev evaluation set and the
input-route decomposition (readout-token copy vs other-token copy) of the R1/R3 responses.  Usage: dynamics_snapshots.py MODEL [...]"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import nnlib as NL, r1lib as L, evalrun as ER
N, B = NL.N, L.B
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_dev.npz"), allow_pickle=True)); out = {}
def route_split(p, Xb, Xa, q, v):
    """margin change from switching only the readout-token copy (z_t in residual, query and its own key) vs only the other tokens' copies."""
    fz = jax.jit(lambda X: NL.frozen_z(p, X)); zb, za = fz(jnp.asarray(Xb)), fz(jnp.asarray(Xa)); t = 36 + q; n = len(q)
    mask = np.zeros((n, 43, 1), bool); mask[np.arange(n), t] = True; m = jnp.asarray(mask)
    def logits(z):
        x = NL.block(p["layers"][2], z); return np.asarray(N.ln(x, p["gf"], p["bf"]) @ p["WA"])[np.arange(n), t]
    mg = lambda lg: B.oriented_margin(lg - lg.mean(-1, keepdims=True), v)
    b, a = mg(logits(zb)), mg(logits(za)); zt = mg(logits(jnp.where(m, za, zb))); ot = mg(logits(jnp.where(m, zb, za)))
    return dict(total=float((a - b).mean()), readout_token_only=float((zt - b).mean()), other_tokens_only=float((ot - b).mean()))
for mname in sys.argv[1:]:
    seed = int(mname[2:]); res = {}
    for st in (0, 250, 500, 1000, 2000, 4000):
        if st == 0:
            p = NL.load(NL.dev_ckpt(mname, 0)); fresh = N.init("M", NL.ENV, seed + 7777); p = dict(p); p["layers"] = list(p["layers"]); p["layers"][2] = fresh["layers"][2]
        else: p = NL.load(NL.dev_ckpt(mname, st))
        r = ER.run(NL.full_model_logits(p), ev); h = ER.headline(r); rs = {}
        for rg in ("R1", "R3"):
            s = ev["orient_F45"] & (ev["reg_F45"] == rg); rs[rg] = route_split(p, ev["X"][s], ev["X_F45"][s], ev["q"][s], ev["vote_F45"][s])
        res[st] = dict(headline=h, routes=rs, WA_norm=float(jnp.linalg.norm(p["WA"])), gf_mean=float(p["gf"].mean()))
        print(mname, st, " ".join(f"{k} {v:+.3f}" for k, v in h.items() if k in ("KL", "acc", "R1", "dm_R1", "R3", "F34R1")),
              "| R1 routes tot %+.3f zt %+.3f oth %+.3f | R3 routes zt %+.3f oth %+.3f" % (rs["R1"]["total"], rs["R1"]["readout_token_only"], rs["R1"]["other_tokens_only"], rs["R3"]["readout_token_only"], rs["R3"]["other_tokens_only"]), flush=True)
    out[mname] = res
    json.dump(out, open(os.path.join(HERE, "..", "logs", f"dynamics_snapshots_{sys.argv[1]}.json"), "w"), indent=1)
