"""Investigation X: fine-tuned native models (Investigation IX native_train.py checkpoints) with intervention hooks.
A model = (parent pf, spec in {c1, full_c1}, trained tree t).  Layers: L0, L1 (pretrained; trainable in full_c1), L2 (fresh).  43 tokens:
source tokens q*6+s (s = source 0..5, group s//2), readout tokens 36+q, episode token 42.
Interventions supported in `forward`:
  head_patch = {(layer, head): (token_mask (n,43) bool, replacement per-head output (n,43,64))}   -- per-head output patching
  resid_patch = {site_layer: fn(x) -> x}   applied to the residual stream BEFORE layer `site_layer` (0..3; 3 = before final LN)
All math identical to nnlib.block (verified in scripts/verify_ftlib.py)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, jax, jax.numpy as jnp
import native as NV
NL, N = NV.NL, NV.N
N9 = NV.N9; PKG = NV.PKG


def model_dir(tag):
    for base in (os.path.join(PKG, "checkpoints", "native_train"), os.path.join(N9, "checkpoints", "native_train")):
        if os.path.isdir(os.path.join(base, tag)): return os.path.join(base, tag)
    raise FileNotFoundError(tag)


def load_model(parent, spec, seed=0, steps=4000, suffix="", step=None):
    tag = f"{parent}_{spec}_s{steps}_seed{seed}{suffix}"; D = model_dir(tag); pf = NV.load_parent(parent)
    fresh = lambda s: N.init("M", NL.ENV, s)
    tmpl = dict(gf=pf["gf"], bf=pf["bf"], WA=fresh(seed + 7777)["WA"], new=[fresh(seed + 7777)["layers"][2]])
    if spec == "full_c1": tmpl["emb"] = {k: pf[k] for k in NL.FROZEN_KEYS}; tmpl["pre"] = [pf["layers"][l] for l in range(2)]
    z = np.load(os.path.join(D, f"step{step or steps}.npz"))
    t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tmpl), [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))])
    emb = t["emb"] if spec == "full_c1" else {k: pf[k] for k in NL.FROZEN_KEYS}
    layers = (t["pre"] if spec == "full_c1" else [pf["layers"][0], pf["layers"][1]]) + [t["new"][0]]
    return dict(tag=tag, parent=parent, spec=spec, seed=seed, emb=emb, layers=layers, gf=t["gf"], bf=t["bf"], WA=t["WA"], dir=D)


def block_heads(lay, x, head_patch=None, l=0, rec=None, mlp_patch=None):
    """nnlib.block with per-head attention outputs exposed/patchable.  head_patch: {head: (mask (n,43), repl (n,43,64))}."""
    z = N.ln(x, lay["g1"], lay["b1"])
    q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
    a = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1]), -1)
    oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])          # n,H,T,d per-head output
    if rec is not None: rec[f"heads{l}"] = oh; rec[f"pattern{l}"] = a
    if head_patch:
        for h, (m, r) in head_patch.items(): oh = oh.at[:, h].set(jnp.where(m[..., None], r, oh[:, h]))
    x = x + oh.sum(1)
    hid = jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True)
    mo = hid @ lay["W2"] + lay["c2"]
    if rec is not None: rec[f"mlp_out{l}"] = mo
    if mlp_patch is not None: m, r = mlp_patch; mo = jnp.where(m[..., None], r, mo)
    return x + mo


def forward(M, X, head_patch=None, resid_patch=None, rec=None, mlp_patch=None):
    """logits (n,6,3).  head_patch {(l,h): (mask, repl)}; resid_patch {l: fn}; mlp_patch {l: (mask, repl)} replaces the MLP output."""
    x = N.embed(dict(M["emb"]), X)
    for l, lay in enumerate(M["layers"]):
        if rec is not None: rec[f"resid{l}"] = x
        if resid_patch and l in resid_patch: x = resid_patch[l](x)
        hp = {h: v for (ll, h), v in (head_patch or {}).items() if ll == l}
        x = block_heads(lay, x, hp, l, rec, (mlp_patch or {}).get(l))
    if rec is not None: rec["resid3"] = x
    if resid_patch and 3 in resid_patch: x = resid_patch[3](x)
    return N.ln(x, M["gf"], M["bf"])[:, 36:42] @ M["WA"]


def batched(fn, X, bs=1000):
    return np.concatenate([np.asarray(fn(jnp.asarray(X[i:i + bs]))) for i in range(0, len(X), bs)])


def centred(L): return L - L.mean(-1, keepdims=True)
