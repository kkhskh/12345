"""Forward pass identical to latent-reliability nets.forward (pre-LN, bidirectional, GELU MLP, final LN, readout logits),
with every intermediate exposed to a hook:
  resid_pre{l} (N,T,d) | head_out{l} (N,H,T,d) per-head contribution a_h V_h W_O,h | attn_out{l} (N,T,d) |
  resid_mid{l} | mlp_hidden{l} (N,T,4d) | mlp_out{l} (N,T,d) | resid_post{l} | final (after final LN)
A hook may return a modified array.  `check_equivalence` verifies equality with nets.forward."""
import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_latent_reliability", "src"))
import jax, jax.numpy as jnp, numpy as np
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
        ho = hk(f"head_out{l}", jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"]))
        o = hk(f"attn_out{l}", ho.sum(1))
        x = hk(f"resid_mid{l}", x + o)
        hmid = hk(f"mlp_hidden{l}", jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True))
        mo = hk(f"mlp_out{l}", hmid @ lay["W2"] + lay["c2"])
        x = hk(f"resid_post{l}", x + mo)
    zf = hk("final", N.ln(x, p["gf"], p["bf"]))
    rd = zf[:, 6 * Q:6 * Q + Q]
    return dict(answer=rd @ p["WA"])


def check_equivalence(p, X):
    a = np.asarray(forward(p, X)["answer"]); b = np.asarray(N.forward(p, X)["answer"]); return float(np.abs(a - b).max())
