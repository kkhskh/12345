"""Frozen P7 (confirmation): input-route split of the ORIGINAL-transfer (C1) R1 response for each confirmation parent, on eval_conf.
readout_token_only = margin change when only the readout token's layer-2 input copy (residual, query and own key/value) takes its
after-edit value; other_tokens_only = only the other 42 tokens.  -> logs/conf_mech.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, nnlib as NL, r1lib as L
N, B = NL.N, L.B
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_conf.npz"), allow_pickle=True)); out = {}
for s_ in range(90, 98):
    f = os.path.join(HERE, "..", "checkpoints", "conf", f"yP{s_}_C1_r0_full.npz")
    if not os.path.exists(f): out[f"yP{s_}"] = None; continue
    p = NL.load(f); sel = ev["orient_F45"] & (ev["reg_F45"] == "R1"); Xb, Xa, q, v = ev["X"][sel], ev["X_F45"][sel], ev["q"][sel], ev["vote_F45"][sel]
    fz = jax.jit(lambda X: NL.frozen_z(p, X)); zb, za = fz(jnp.asarray(Xb)), fz(jnp.asarray(Xa)); t = 36 + q; n = len(q)
    mask = np.zeros((n, 43, 1), bool); mask[np.arange(n), t] = True; m = jnp.asarray(mask)
    def lg(z):
        x = NL.block(p["layers"][2], z); return np.asarray(N.ln(x, p["gf"], p["bf"]) @ p["WA"])[np.arange(n), t]
    mg = lambda l: B.oriented_margin(l - l.mean(-1, keepdims=True), v)
    b, a = mg(lg(zb)), mg(lg(za)); out[f"yP{s_}"] = dict(n=int(n), total=float((a - b).mean()), readout_token_only=float((mg(lg(jnp.where(m, za, zb))) - b).mean()),
                                                         other_tokens_only=float((mg(lg(jnp.where(m, zb, za))) - b).mean()))
    print(s_, out[f"yP{s_}"], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "conf_mech.json"), "w"), indent=1)
