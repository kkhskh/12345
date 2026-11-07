"""Post-review checks on the learned models (R2 FATAL-2 / MAJOR-6, R5 H3 / M1, R7 point 2). EXPLORATORY.

K1  cosine between the sign(ab) probe direction at the final residual and the readout direction
    gamma_f * (W_U[c] - W_U[c']) (R2/R5: final-site interchange = readout alignment).
K2  readout-direction interchange at the final site (control: transfers whatever the label is).
K3  ablation method: mean vs resample ablation of the flagged units; and ablation of the NON-flagged units.
K4  graded-output single-cancellation test (T-RF') on the 61x61 grid margin: row pairs (a,a') whose order reverses
    across b by more than 0.05 SD of the margin.
K5  stricter relation-free baselines: B-add3 f(a)+g(b)+h(p) (no prior in f, g) and B-signs (linear on per-source
    sign one-hots + prior), same optimizer/steps/data.
"""
import itertools, json, sys, warnings
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, optax
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learned as Lm
import learned_analysis as LA


def k1_k2(task, arch, seed):
    P = LA.load(task, arch, seed); L = len(P["layers"])
    bt = LA.sample_binary(task, 4000, np.random.default_rng(3000))
    _, st = LA.capture(P, task, LA.inp_bin(bt.a, bt.b, bt.p))
    site = f"resid_post{L - 1}"
    lab = (bt.a * bt.b > 0).astype(int)
    w = LogisticRegression(C=10.0, max_iter=3000).fit(st[site][:2000, -1], lab[:2000]).coef_[0]
    w = w / np.linalg.norm(w)
    rd = np.asarray(P["gf"] * (P["WU"][:, 0] - P["WU"][:, 1])); rd = rd / np.linalg.norm(rd)
    # final-site interchange along the readout direction on matched pairs with differing decisions
    agp, cfp = LA.matched_pairs()
    base_in = LA.inp_bin(agp[:, 0], agp[:, 1], np.ones(len(agp))); src_in = LA.inp_bin(cfp[:, 0], cfp[:, 1], np.ones(len(cfp)))
    mb = LA.fwd_margin(P, task, base_in); ms = LA.fwd_margin(P, task, src_in)
    diff = np.sign(mb) != np.sign(ms)
    _, ss = LA.capture(P, task, src_in)
    sc = ss[site][:, -1] @ rd
    def hk(name, arr):
        if name == site:
            a_ = np.asarray(arr).copy(); a_[:, -1] = a_[:, -1] + np.outer(sc - a_[:, -1] @ rd, rd); return jnp.asarray(a_)
        return arr
    mp = LA.fwd_margin(P, task, base_in, hk)
    tr = float(np.mean(np.sign(mp[diff]) == np.sign(ms[diff]))) if diff.any() else None
    return dict(abs_cos_probe_readout=float(abs(w @ rd)), readout_dir_interchange_transfer=tr, n_differing=int(diff.sum()))


def k3(task, arch, seed, sel):
    P = LA.load(task, arch, seed); L = len(P["layers"])
    val = Lm.sample(task, 20000, np.random.default_rng(1000))
    inp = Lm.to_inp(task, val)
    _, st = LA.capture(P, task, inp)
    rng = np.random.default_rng(77)
    perm = rng.permutation(len(val.y))
    def acc(hk):
        lg = np.asarray(Lm.forward(P, task, inp, hk))
        return float((np.where(lg[:, 0] > lg[:, 1], 1.0, -1.0) == val.y).mean())
    def make(kind, units):
        def hk(name, arr):
            for l, us in units.items():
                if name == f"mlp_hidden{l}" and len(us):
                    a_ = np.asarray(arr).copy()
                    src = st[f"mlp_hidden{l}"][:, -1]
                    a_[:, -1, us] = src[:, us].mean(0) if kind == "mean" else src[perm][:, us]
                    return jnp.asarray(a_)
            return arr
        return hk
    W = P["layers"][0]["W1"].shape[1]
    non = {l: [u for u in range(W) if u not in sel[l]] for l in sel}
    out = dict(acc_full=acc(None), mean_flagged=acc(make("mean", sel)), resample_flagged=acc(make("resample", sel)),
               mean_nonflagged=acc(make("mean", non)), resample_nonflagged=acc(make("resample", non)),
               frac_flagged=float(sum(len(v) for v in sel.values()) / (W * len(sel))))
    return out


def k4(task, arch, seed, pv=1.0):
    P = LA.load(task, arch, seed)
    g = np.linspace(-3, 3, 31)
    A, B = np.meshgrid(g, g, indexing="ij")
    m = LA.fwd_margin(P, task, LA.inp_bin(A.ravel(), B.ravel(), np.full(A.size, pv))).reshape(31, 31)
    tol = 0.05 * m.std()
    worst = 0.0; nrev = 0; tot = 0
    for i, j in itertools.combinations(range(31), 2):
        d = m[i] - m[j]
        tot += 1
        if (d > tol).any() and (d < -tol).any():
            nrev += 1
            worst = max(worst, min(d.max(), -d.min()) / m.std())
    return dict(reversing_row_pairs=nrev, total=tot, max_reversal_in_SD=float(worst))


def strict_baselines(task, seed):
    K = Lm.n_classes(task)
    k = iter(jax.random.split(jax.random.PRNGKey(seed + 999), 32))
    def mlp_p(nin):
        return dict(W1=0.3 * jax.random.normal(next(k), (nin, 32)), c1=jnp.zeros(32), W2=0.3 * jax.random.normal(next(k), (32, 32)),
                    c2=jnp.zeros(32), W3=0.3 * jax.random.normal(next(k), (32, K)), c3=jnp.zeros(K))
    def mlp(p, x):
        h = jax.nn.gelu(x @ p["W1"] + p["c1"], approximate=True); h = jax.nn.gelu(h @ p["W2"] + p["c2"], approximate=True)
        return h @ p["W3"] + p["c3"]
    vin = 3 if task.startswith("T5") else 1
    def parts(inp):
        if task.startswith("T5"):
            pp = jax.nn.one_hot(inp["p"], 3) if task == "T5b" else jnp.zeros((inp["zA"].shape[0], 3))
            return inp["zA"], inp["zB"], pp
        return inp["a"][:, None], inp["b"][:, None], inp["p"][:, None]
    res = {}
    # B-add3
    P = dict(f=mlp_p(vin), g=mlp_p(vin), h=mlp_p(3 if task.startswith("T5") else 1))
    f3 = lambda P, inp: (lambda xa, xb, xp: mlp(P["f"], xa) + mlp(P["g"], xb) + mlp(P["h"], xp))(*parts(inp))
    # B-signs: linear on per-source support one-hots + prior
    def sfeat(inp):
        xa, xb, xp = parts(inp)
        if task.startswith("T5"):
            return jnp.concatenate([jax.nn.one_hot(xa.argmax(1), 3), jax.nn.one_hot(xb.argmax(1), 3), xp], 1)
        return jnp.concatenate([jnp.sign(xa), jnp.sign(xb), xp], 1)
    nin = 9 if task.startswith("T5") else 3
    Ps = dict(W=0.1 * jax.random.normal(next(k), (nin, K)), c=jnp.zeros(K))
    fs = lambda P, inp: sfeat(inp) @ P["W"] + P["c"]
    val = Lm.sample(task, 20000, np.random.default_rng(1000))
    for name, f, P0 in (("B-add3", f3, P), ("B-signs", fs, Ps)):
        opt = optax.adam(optax.cosine_decay_schedule(Lm.LR, Lm.STEPS)); st = opt.init(P0)
        @jax.jit
        def step(P, st, inp, tgt):
            l, g_ = jax.value_and_grad(lambda P: -(tgt * jax.nn.log_softmax(f(P, inp))).sum(-1).mean())(P)
            up, st = opt.update(g_, st, P); return optax.apply_updates(P, up), st, l
        rng = np.random.default_rng(40_000 * (seed + 1) + Lm.TASKS.index(task))
        Pc = P0
        for _ in range(Lm.STEPS):
            bt = Lm.sample(task, Lm.BATCH, rng)
            Pc, st, _ = step(Pc, st, Lm.to_inp(task, bt), Lm.soft_target(task, bt))
        lg = np.asarray(f(Pc, Lm.to_inp(task, val)))
        res[name] = float((lg.argmax(1) == val.y).mean()) if task.startswith("T5") else \
            float((np.where(lg[:, 0] > lg[:, 1], 1.0, -1.0) == val.y).mean())
    return res


def main():
    out = {}; lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    an = {**json.load(open(HERE.parent / "logs" / "learned_analysis_dev.json")),
          **json.load(open(HERE.parent / "logs" / "learned_analysis_heldout.json"))}
    for task in ["T1", "T2", "T3", "T4"]:
        for arch in ["TF1", "TF2", "TF2AO"]:
            for seed in range(5):
                key = f"{task}/{arch}/s{seed}"
                r = dict(K1K2=k1_k2(task, arch, seed), K4=k4(task, arch, seed))
                if arch != "TF2AO":
                    sel = {int(l): [x["unit"] for x in v] for l, v in an[key]["units"].items()}
                    r["K3"] = k3(task, arch, seed, sel)
                out[key] = r
                log(f"{key}: {json.dumps(r)}")
    for task in Lm.TASKS:
        for seed in range(3):
            b = strict_baselines(task, seed)
            out[f"{task}/strict/s{seed}"] = b
            log(f"{task} strict relation-free baselines seed {seed}: {json.dumps(b)}")
    (HERE.parent / "logs" / "learned_review_checks.txt").write_text("\n".join(lines) + "\n")
    (HERE.parent / "logs" / "learned_review_checks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
