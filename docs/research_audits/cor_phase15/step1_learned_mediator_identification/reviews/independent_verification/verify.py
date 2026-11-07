"""Independent re-implementation (pure numpy) of the key mediator measurements.
Does NOT import src/mediator.py or src/run_protocol.py. src/model.py is used only in check_forward()."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "checkpoints" / "historical"

def load(name):
    z = np.load(CK / f"{name}.npz")
    p = {k: z[k].astype(np.float64) for k in z.files if not k.startswith("layers.")}
    L = sorted({int(k.split(".")[1]) for k in z.files if k.startswith("layers.")})
    p["layers"] = [{k.split(".", 2)[2]: z[k].astype(np.float64) for k in z.files if k.startswith(f"layers.{l}.")} for l in L]
    return p

def ln(x, g, b):
    mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + 1e-5) * g + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def embed(p, task, a, b, pp):
    toks = [a[:, None] * p["Wval"][0] + p["Eid"][0], b[:, None] * p["Wval"][0] + p["Eid"][1]]
    if task == "T3":
        toks.append(pp[:, None] * p["Wp"][0] + p["Eid"][2])
    toks.append(np.broadcast_to(p["Eid"][3], toks[0].shape))
    return np.stack(toks, 1)

def attn(x, lay):
    z = ln(x, lay["g1"], lay["b1"])
    q = np.einsum("ntd,hde->nhte", z, lay["WQ"]); k = np.einsum("ntd,hde->nhte", z, lay["WK"])
    v = np.einsum("ntd,hde->nhte", z, lay["WV"])
    s = np.einsum("nhte,nhse->nhts", q, k) / np.sqrt(8.0)
    T = x.shape[1]; s = np.where(np.tril(np.ones((T, T), bool)), s, -1e9)
    s = s - s.max(-1, keepdims=True); w = np.exp(s); w /= w.sum(-1, keepdims=True)
    return x + np.einsum("nhts,nhse,hed->ntd", w, v, lay["WO"]) + lay["bO"]

def mlp(x, lay):
    if "W1" not in lay: return x
    return x + gelu(ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"]) @ lay["W2"] + lay["c2"]

def logits_from_x(p, x):
    return ln(x[:, -1], p["gf"], p["bf"]) @ p["WU"] + p["bU"]

def forward(p, task, a, b, pp):
    x = embed(p, task, a, b, pp)
    for lay in p["layers"]:
        x = mlp(attn(x, lay), lay)
    return logits_from_x(p, x)

# site = resid_mid{L-1} at last position (TF1: mid0, TF2: mid1)
def site(p, task, a, b, pp):
    x = embed(p, task, a, b, pp)
    for lay in p["layers"][:-1]:
        x = mlp(attn(x, lay), lay)
    return attn(x, p["layers"][-1])[:, -1]

def readout_from_site(p, h):
    """continue from patched last-position resid_mid of the final layer (only last position feeds the readout)."""
    x = h[:, None, :]
    x = mlp(x, p["layers"][-1])
    return logits_from_x(p, x)

def dec(lg): return np.where(lg[:, 0] > lg[:, 1], 1, -1)

def check_forward():
    sys.path.insert(0, str(ROOT / "src"))
    import model, ckpt  # model.forward used ONLY for this equality check
    import jax.numpy as jnp
    rng = np.random.default_rng(12345); worst = 0.0
    for name in ["T1_TF1_s4", "T1_TF2_s2", "T4_TF1_s2", "T3_TF1_s2"]:
        task = name.split("_")[0]
        a, b = rng.uniform(-3, 3, 256), rng.uniform(-3, 3, 256); pp = rng.choice([-1.0, 1.0], 256)
        ref = np.asarray(model.forward(ckpt.load(CK / f"{name}.npz"), task,
                                       dict(a=jnp.asarray(a, jnp.float32), b=jnp.asarray(b, jnp.float32), p=jnp.asarray(pp, jnp.float32))))
        mine = forward(load(name), task, a, b, pp)
        d = float(np.abs(ref - mine).max()); worst = max(worst, d)
        print(f"forward check {name}: max |dlogit| = {d:.2e}")
    return worst

def pairs(task, n, rng):
    """my own matched pairs: same z (and p), one ab>0, one ab<0, both in [-3,3]^2, |z| in [0.2,2.5]."""
    z = rng.uniform(0.2, 2.5, n) * rng.choice([-1, 1], n)
    # agree: a,b same sign as z: a = t*z, t in (0.01,0.99)
    t = rng.uniform(0.01, 0.99, n); aA = t * z; bA = z - aA
    # conflict: a opposite sign to z or b opposite: a = z + u*sign(z), u in (0.01, 3-|z|), b = z - a
    u = rng.uniform(0.01, 1, n) * (3 - np.abs(z) - 0.01)
    flip = rng.random(n) < 0.5
    aC = np.where(flip, z + u * np.sign(z), -u * np.sign(z)); bC = z - aC
    pp = rng.choice([-1.0, 1.0], n) if task == "T3" else np.zeros(n)
    assert np.allclose(aA + bA, aC + bC) and (aA * bA > 0).all() and (aC * bC < 0).all()
    assert max(map(lambda v: np.abs(v).max(), (aA, bA, aC, bC))) <= 3
    return (aA, bA, pp), (aC, bC, pp)

def proj(V): return V @ V.T

def iia(p, hb, hs, P, ds):
    hp = hb - hb @ P + hs @ P
    return float((dec(readout_from_site(p, hp)) == ds).mean())

def additive_R2(p, task, V):
    g = np.linspace(-3, 3, 61); A, B = np.meshgrid(g, g, indexing="ij")
    out = {}; ss_res_all = ss_tot_all = 0.0
    for pv in ([1.0, -1.0] if task == "T3" else [0.0]):
        S = site(p, task, A.ravel(), B.ravel(), np.full(A.size, pv)) @ V   # (3721, k)
        S = S.reshape(61, 61, -1)
        fit = S.mean(1, keepdims=True) + S.mean(0, keepdims=True) - S.mean((0, 1), keepdims=True)
        r = ((S - fit) ** 2).sum(); t = ((S - S.mean((0, 1), keepdims=True)) ** 2).sum()
        out[f"p={pv:+.0f}" if task == "T3" else "all"] = float(1 - r / t)
        ss_res_all += r; ss_tot_all += t
    out["pooled"] = float(1 - ss_res_all / ss_tot_all)
    return out

def analyse(name, seed):
    task = name.split("_")[0]; p = load(name); rng = np.random.default_rng(seed)
    (fA, fC) = pairs(task, 800, rng)
    D = site(p, task, *fA) - site(p, task, *fC)
    sv = np.linalg.svd(D, full_matrices=False)
    en = sv[1] ** 2 / (sv[1] ** 2).sum()
    k = int(min(8, np.searchsorted(np.cumsum(en), 0.9) + 1))
    V = sv[2][:k].T; P = proj(V)
    eA, eC = pairs(task, 500, rng)
    hA, hC = site(p, task, *eA), site(p, task, *eC)
    dA, dC = dec(forward(p, task, *eA)), dec(forward(p, task, *eC))
    m = dA != dC
    hA, hC, dA, dC = hA[m], hC[m], dA[m], dC[m]
    I = np.eye(hA.shape[1])
    both = lambda P: 0.5 * (iia(p, hA, hC, P, dC) + iia(p, hC, hA, P, dA))
    rand = []
    for _ in range(10):
        Q = np.linalg.qr(rng.normal(size=(hA.shape[1], k)))[0]; rand.append(both(proj(Q)))
    mean = (hA + hC) / 2
    nA = dec(readout_from_site(p, hA - hA @ P + mean @ P)); nC = dec(readout_from_site(p, hC - hC @ P + mean @ P))
    sA = dec(readout_from_site(p, hA @ P + hC @ (I - P))); sC = dec(readout_from_site(p, hC @ P + hA @ (I - P)))
    return dict(n_model_differs=int(m.sum()), k=k, sv_energy=en[:4].tolist(),
                iia_full_site=both(I), iia_S_rel=both(P),
                iia_S_rel_dirs=[iia(p, hA, hC, P, dC), iia(p, hC, hA, P, dA)],
                iia_S_rand_mean=float(np.mean(rand)), iia_S_rand_max=float(np.max(rand)),
                necessity_flip_after=float((nA != nC).mean()),
                sufficiency=float(0.5 * ((sA == dA).mean() + (sC == dC).mean())),
                additive_R2=additive_R2(p, task, V))

if __name__ == "__main__":
    worst = check_forward()
    res = {"forward_max_abs_diff": worst}
    for name, seed in [("T1_TF1_s4", 90001), ("T1_TF2_s2", 90002), ("T4_TF1_s2", 90003), ("T3_TF1_s2", 90004)]:
        res[name] = analyse(name, seed); print(name, json.dumps(res[name]))
    (Path(__file__).parent / "results.json").write_text(json.dumps(res, indent=1))
