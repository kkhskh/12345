"""EXPLORATORY (not preregistered; added after the dev-seed results were seen).

E-add   additive share of each binary model's margin on G-full: R2_add = 1 - ||m - (f(a)+g(b))||^2 / Var m
        (two-way ANOVA main effects; for T3 per prior value).
E-B-add relation-free source-additive baseline, same data/optimizer/steps as the preregistered baselines:
        binary: m = f(a, p) + g(b, p);   T5: logit_k = f_k(zA, p) + g_k(zB, p)
        (f, g: MLPs 2x32 GELU; p one-hot/scalar). No term can read both sources.
E-int   interchange restricted to matched pairs whose base and source decisions differ.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learned as Lm  # noqa: E402
import learned_analysis as LA  # noqa: E402


def badd_train(task, seed):
    K = Lm.n_classes(task)
    vin = 3 if task.startswith("T5") else 1
    pin = 3 if task == "T5b" else 1
    k = iter(jax.random.split(jax.random.PRNGKey(seed + 4242), 16))
    def mlp_p(nin):
        return dict(W1=0.3 * jax.random.normal(next(k), (nin, 32)), c1=jnp.zeros(32),
                    W2=0.3 * jax.random.normal(next(k), (32, 32)), c2=jnp.zeros(32),
                    W3=0.3 * jax.random.normal(next(k), (32, K)), c3=jnp.zeros(K))
    def mlp(p, x):
        h = jax.nn.gelu(x @ p["W1"] + p["c1"], approximate=True)
        h = jax.nn.gelu(h @ p["W2"] + p["c2"], approximate=True)
        return h @ p["W3"] + p["c3"]
    P = dict(f=mlp_p(vin + pin), g=mlp_p(vin + pin))
    def feats(inp):
        if task.startswith("T5"):
            pp = jax.nn.one_hot(inp["p"], 3) if task == "T5b" else jnp.zeros((inp["zA"].shape[0], 1))
            return jnp.concatenate([inp["zA"], pp], 1), jnp.concatenate([inp["zB"], pp], 1)
        pp = inp["p"][:, None]
        return jnp.concatenate([inp["a"][:, None], pp], 1), jnp.concatenate([inp["b"][:, None], pp], 1)
    def f(P, inp):
        xa, xb = feats(inp)
        return mlp(P["f"], xa) + mlp(P["g"], xb)
    opt = optax.adam(optax.cosine_decay_schedule(Lm.LR, Lm.STEPS)); st = opt.init(P)
    @jax.jit
    def step(P, st, inp, tgt):
        l, g = jax.value_and_grad(lambda P: -(tgt * jax.nn.log_softmax(f(P, inp))).sum(-1).mean())(P)
        up, st = opt.update(g, st, P)
        return optax.apply_updates(P, up), st, l
    rng = np.random.default_rng(30_000 * (seed + 1) + Lm.TASKS.index(task))
    for _ in range(Lm.STEPS):
        bt = Lm.sample(task, Lm.BATCH, rng)
        P, st, l = step(P, st, Lm.to_inp(task, bt), Lm.soft_target(task, bt))
    val = Lm.sample(task, 20000, np.random.default_rng(1000))
    lg = np.asarray(f(P, Lm.to_inp(task, val)))
    if task.startswith("T5"):
        return float((lg.argmax(1) == val.y).mean())
    return float((np.where(lg[:, 0] > lg[:, 1], 1.0, -1.0) == val.y).mean())


def additive_share(task, arch, seed):
    P = LA.load(task, arch, seed)
    A, B, g = LA.grid()
    n = len(g)
    out = {}
    for pv in ([1, -1] if task == "T3" else [0]):
        m = LA.fwd_margin(P, task, LA.inp_bin(A, B, np.full_like(A, float(pv))))
        Z = m.reshape(n, n)
        res = Z - LA.additive_fit(Z)
        # also: share of the sign pattern reproduced by the additive fit
        sgn = float(np.mean(np.sign(LA.additive_fit(Z)) == np.sign(Z)))
        out[f"p={pv:+d}"] = dict(R2_add=float(1 - res.var() / Z.var()), sign_agreement_of_additive_fit=sgn)
    return out


def interchange_differing(task, arch, seed):
    P = LA.load(task, arch, seed)
    L = len(P["layers"])
    agp, cfp = LA.matched_pairs()
    pv = 1.0
    base_in = LA.inp_bin(agp[:, 0], agp[:, 1], np.full(len(agp), pv))
    src_in = LA.inp_bin(cfp[:, 0], cfp[:, 1], np.full(len(cfp), pv))
    mb = LA.fwd_margin(P, task, base_in); ms = LA.fwd_margin(P, task, src_in)
    diff = np.sign(mb) != np.sign(ms)
    if diff.sum() == 0:
        return {"n_differing": 0}
    from sklearn.linear_model import LogisticRegression
    bt = LA.sample_binary(task, 4000, np.random.default_rng(3000))
    _, st = LA.capture(P, task, LA.inp_bin(bt.a, bt.b, bt.p))
    _, st_src = LA.capture(P, task, src_in)
    lab = (bt.a * bt.b > 0).astype(int)
    rnd = np.random.default_rng(999).normal(size=Lm.D_MODEL); rnd /= np.linalg.norm(rnd)
    out = {"n_differing": int(diff.sum())}
    for site in [f"resid_mid{l}" for l in range(L)] + [f"resid_post{l}" for l in range(L)]:
        w = LogisticRegression(C=10.0, max_iter=3000).fit(st[site][:2000, -1], lab[:2000]).coef_[0]
        w = w / np.linalg.norm(w)
        for nm, u in (("probe", w), ("random", rnd)):
            sc = st_src[site][:, -1] @ u
            def hk(name, arr, site=site, u=u, sc=sc):
                if name == site:
                    a_ = np.asarray(arr).copy(); a_[:, -1] = a_[:, -1] + np.outer(sc - a_[:, -1] @ u, u); return jnp.asarray(a_)
                return arr
            mp = LA.fwd_margin(P, task, base_in, hk)
            out[f"{site}/{nm}"] = float(np.mean(np.sign(mp[diff]) == np.sign(ms[diff])))
    return out


def main(tag, seeds):
    res = {}
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    for task in Lm.TASKS:
        for seed in seeds:
            acc = badd_train(task, seed)
            res[f"{task}/B-add/s{seed}"] = acc
            log(f"{task} B-add (relation-free source-additive) seed {seed}: val acc {acc:.4f}")
        if task in LA.BIN:
            for arch in Lm.ARCH:
                for seed in seeds:
                    a = additive_share(task, arch, seed)
                    i = interchange_differing(task, arch, seed)
                    res[f"{task}/{arch}/s{seed}/additive"] = a
                    res[f"{task}/{arch}/s{seed}/interchange_differing"] = i
                    log(f"{task} {arch} s{seed}: additive {json.dumps(a)}; interchange(differing pairs) {json.dumps(i)}")
    root = HERE.parent
    (root / "logs" / f"learned_posthoc_{tag}.txt").write_text("\n".join(lines) + "\n")
    (root / "logs" / f"learned_posthoc_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], [int(s) for s in sys.argv[2].split(",")])
