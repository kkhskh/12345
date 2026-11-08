"""Forward pass identical to latent-reliability nets.forward, with additional per-head hooks:
  head_out{l} : (N,H,T,d) each head's contribution a_h V_h W_O,h  (summed over h it equals attn_out{l})
  pattern{l}, resid_pre{l}, resid_post{l}, final as in nets.forward.
Equivalence with nets.forward is checked in scripts/mechanism_atlas.py (max |diff| reported)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "step1_latent_reliability", "src"))
import jax, jax.numpy as jnp
import nets as N


def forward(p, X, hook=None):
    hk = (lambda n, a: a) if hook is None else hook
    x = N.embed(p, X); Q = X.shape[1]
    for l, lay in enumerate(p["layers"]):
        x = hk(f"resid_pre{l}", x)
        z = N.ln(x, lay["g1"], lay["b1"])
        q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
        s = jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1])
        a = hk(f"pattern{l}", jax.nn.softmax(s, -1))
        ho = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
        ho = hk(f"head_out{l}", ho)
        x = x + ho.sum(1)
        hmid = jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True)
        x = hk(f"resid_post{l}", x + hmid @ lay["W2"] + lay["c2"])
    zf = hk("final", N.ln(x, p["gf"], p["bf"]))
    rd = zf[:, 6 * Q:6 * Q + Q]
    return dict(answer=rd @ p["WA"], conf=jax.nn.sigmoid((rd @ p["WC"])[..., 0]), rel=zf[:, -1] @ p["WR"])
