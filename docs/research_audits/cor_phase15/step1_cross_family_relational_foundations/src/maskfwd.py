"""XIV: forward pass of a fine-tuned T4 unit with additive attention biases / logit noise (math identical to ft_factorial_b.block_m).
masks: {layer: (43,43) bias}; lnoise: (layer, sigma, keymask (43,43) bool, jax PRNGKey) adds sigma*N(0,1) to the logits on keymask."""
import jax, jax.numpy as jnp, numpy as np
import native as NV
N, NL = NV.N, NV.NL
NEG = -1e9
def edges(kind):
    """(layer, bool (43,43) edge set) for the named edge sets used in Stage I."""
    m = np.zeros((43, 43), bool)
    if kind in ("ro0", "own0", "own2"):          # readout q -> own-question sources (ro0 = all sources, as in XIII-E)
        pass
    if kind == "ro0": m[36:42, 0:36] = True; return 0, m
    if kind in ("own0", "own2"):
        for q in range(6): m[36 + q, q * 6:(q + 1) * 6] = True
        return (0 if kind == "own0" else 2), m
    if kind in ("oq0", "oq2"):                   # readout q -> sources of OTHER questions
        for q in range(6): m[36 + q, 0:36] = True; m[36 + q, q * 6:(q + 1) * 6] = False
        return (0 if kind == "oq0" else 2), m
    if kind in ("rd0", "rd2"):                   # readout -> readout and episode tokens
        m[36:42, 36:43] = True; return (0 if kind == "rd0" else 2), m
    raise ValueError(kind)
def block_m(lay, x, bias=None, noise=None, rec=None, l=0):
    z = N.ln(x, lay["g1"], lay["b1"])
    q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
    s = jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1])
    if bias is not None: s = s + bias
    if noise is not None: sig, km, key = noise; s = s + sig * jnp.asarray(km) * jax.random.normal(key, s.shape)
    a = jax.nn.softmax(s, -1)
    if rec is not None: rec[f"pattern{l}"] = a
    x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
    h = jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); return x + h @ lay["W2"] + lay["c2"]
def forward(M, X, biases=None, noise=None, rec=None):
    """biases {layer: bool edge mask}; noise (layer, sigma, keymask, key)."""
    x = N.embed(dict(M["emb"]), jnp.asarray(X))
    for l, lay in enumerate(M["layers"]):
        b = None if not biases or l not in biases else jnp.asarray(np.where(biases[l], NEG, 0.0).astype(np.float32))
        nz = (noise[1], noise[2], noise[3]) if noise is not None and noise[0] == l else None
        x = block_m(lay, x, b, nz, rec, l)
    return N.ln(x, M["gf"], M["bf"])[:, 36:42] @ M["WA"]
