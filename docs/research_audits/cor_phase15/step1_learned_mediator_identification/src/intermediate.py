"""PROTOCOL_ADDENDUM_XOR3.md: is agreement used as an intermediate variable distinct from the target?"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ckpt, generators as G, mediator as MD, model as M
from run_protocol import PRIMARY

TASK = "REL-XOR3"


def pairs_opp_p(n, seed, flip_p=True):
    A, C = G.matched_pairs(TASK, n, np.random.default_rng(seed))
    if flip_p:   # source member: opposite agreement AND opposite p, same z
        C["p"] = -A["p"]; C["y"], C["soft"] = G.target(TASK, C["a"], C["b"], C["p"])
    # randomise which member is base (agree or conflict)
    rng = np.random.default_rng(seed + 1); sw = rng.random(n) < 0.5
    B, S = {}, {}
    for k in ("a", "b", "p"):
        B[k] = np.where(sw, C[k], A[k]); S[k] = np.where(sw, A[k], C[k])
    for X in (B, S):
        X["y"], X["soft"] = G.target(TASK, X["a"], X["b"], X["p"]); X["z"] = X["a"] + X["b"]; X["s"] = (X["a"] * X["b"] > 0).astype(int)
    return B, S


def predictions(B, S):
    sg = np.sign
    return {"H_AB": sg(S["a"] * S["b"]) * B["p"], "H_y": S["y"], "H_AP": sg(S["a"] * S["p"]) * sg(B["b"]),
            "H_BP": sg(S["b"] * S["p"]) * sg(B["a"])}


def patched(site, B, S, U):
    _, hb = site.get(B); _, hs = site.get(S)
    return MD.decision(TASK, site.run_with(B, hb + (hs - hb) @ U @ U.T))


def das_to(site, B, S, labels, k, steps=600, seed=0, lr=0.03):
    P, code, hook, pos = site.P, site.code, site.hook, site.pos
    _, hb = site.get(B); _, hs = site.get(S)
    inB = MD.to_inp(TASK, B)
    y = jnp.asarray(labels); Y = jnp.stack([(y > 0).astype(jnp.float32), (y < 0).astype(jnp.float32)], 1)
    W0 = jnp.asarray(np.random.default_rng(seed).normal(size=(hb.shape[1], k)), jnp.float32)
    def loss(W):
        U, _ = jnp.linalg.qr(W)
        new = jnp.asarray(hb) + (jnp.asarray(hs) - jnp.asarray(hb)) @ U @ U.T
        def hk(name, arr):
            if name == hook:
                arr = arr.at[:, pos, :].set(new.reshape(arr.shape[0], len(pos), -1))
            return arr
        return -(Y * jax.nn.log_softmax(M.forward(P, code, inB, hk))).sum(-1).mean()
    opt = optax.adam(lr); st = opt.init(W0); g = jax.jit(jax.value_and_grad(loss)); W = W0
    for _ in range(steps):
        l, gr = g(W); up, st = opt.update(gr, st); W = optax.apply_updates(W, up)
    return MD.orth(np.asarray(W))


def main(seeds, archs, tag):
    out = {}
    for arch in archs:
        for seed in seeds:
            f = ROOT / "checkpoints/new" / f"T7_{arch}_s{seed}.npz"
            P = ckpt.load(f)
            val = M.sample("T7", 20000, np.random.default_rng(1000))
            acc = M.accuracy(P, "T7", val)
            for hook, pos in PRIMARY[arch]:
                site = MD.Site(P, TASK, hook, pos)
                A, C = G.matched_pairs(TASK, 800, np.random.default_rng(5001))
                U, k, _ = MD.s_rel(site, A, C)
                Bf, Sf = pairs_opp_p(800, 7001)
                Bd, Sd = pairs_opp_p(600, 7002)
                Bh, Sh = pairs_opp_p(600, 7003)
                Bc, Sc = pairs_opp_p(600, 7004, flip_p=False)
                pf = predictions(Bf, Sf)
                Uab = das_to(site, Bf, Sf, pf["H_AB"], k); Uy = das_to(site, Bf, Sf, pf["H_y"], k)
                r = dict(acc=acc, k=k)
                for nm, V, (Bx, Sx) in (("S_rel/D", U, (Bd, Sd)), ("S_rel/C", U, (Bc, Sc)),
                                        ("S_das_AB/heldoutD", Uab, (Bh, Sh)), ("S_das_y/heldoutD", Uy, (Bh, Sh))):
                    d = patched(site, Bx, Sx, V); pr = predictions(Bx, Sx)
                    r[nm] = {h: float((d == v).mean()) for h, v in pr.items()}
                full = patched(site, Bd, Sd, None if False else np.eye(site.get(Bd)[1].shape[1]))
                r["full_state/D"] = {h: float((full == v).mean()) for h, v in predictions(Bd, Sd).items()}
                rel, ab, y = r["S_rel/D"], r["S_das_AB/heldoutD"]["H_AB"], r["S_das_y/heldoutD"]["H_y"]
                r["verdict"] = ("AGREEMENT_INTERMEDIATE" if rel["H_AB"] >= 0.9 or (ab >= 0.9 and ab - y >= 0.2) else
                                "TARGET_ONLY" if rel["H_y"] >= 0.9 and ab < 0.9 else
                                "OTHER_INTERMEDIATE" if max(rel["H_AP"], rel["H_BP"]) >= 0.9 else "UNRESOLVED")
                out[f"{arch}/s{seed}/{hook}@{pos}"] = r
                print(arch, seed, f"{hook}@{pos}", "acc", round(acc, 4), "k", k, r["verdict"],
                      {kk: {h: round(v, 3) for h, v in vv.items()} for kk, vv in r.items() if isinstance(vv, dict)}, flush=True)
    (ROOT / "logs" / f"xor3_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1].split(",")], sys.argv[2].split(","), sys.argv[3])
