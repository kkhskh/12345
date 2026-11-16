"""XIV Stage III: trajectory evaluation of GATE models against the exact rules (used for development on s1_stop400 and the
dev finals; the frozen X1 scorer calls `evaluate`).  Usage: g_traj_eval.py OUT.json STREAM N CKPTDIR_or_FINAL...
Per checkpoint: cross-entropy to Bayes (expected NLL), KL(rule||model) for bayes / gatefree / sourcewise, accuracy vs labels,
gate-flip following (Bayes-decisive flips), swap(x3,x5) following of the gate-free rule (gatefree-decisive swaps; the source-wise
rule is exactly invariant to this swap), and the same following statistics for the three rules themselves (references)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, gatelib as GL
def evaluate(p, D):
    X, Xf, Xw = D["X"], D["Xf"], D["Xw"]; P0, Pf, Pw = GL.probs(p, X), GL.probs(p, Xf), GL.probs(p, Xw); PB = D["R"]["bayes"]
    r = dict(NLL=float(-(PB * np.log(P0 + 1e-12)).sum(-1).mean()), acc=float((P0.argmax(-1) == D["Y"]).mean()))
    for n in ("bayes", "gatefree", "sourcewise"): r[f"KL_{n}"] = GL.kl(D["R"][n], P0)
    r["gateflip_follow"], r["n_gateflip"] = GL.following(P0, Pf, D["R"]["bayes"], D["Rf"]["bayes"])
    r["swap_follow_gatefree"], r["n_swap"] = GL.following(P0, Pw, D["R"]["gatefree"], D["Rw"]["gatefree"])
    r["swap_follow_bayes"], _ = GL.following(P0, Pw, D["R"]["bayes"], D["Rw"]["bayes"])
    return r
def dataset(stream, n):
    rng = np.random.default_rng(stream); d = GL.TX.sample_gate(n, rng); X = d["X"]; Xf = GL.gate_flip(X, rng); Xw = GL.swap35(X)
    D = dict(X=X, Y=d["Y"], Xf=Xf, Xw=Xw); D["R"] = {k: GL.rule(k, X) for k in ("bayes", "gatefree", "sourcewise")}
    D["Rf"] = {k: GL.rule(k, Xf) for k in ("bayes", "gatefree", "sourcewise")}; D["Rw"] = {k: GL.rule(k, Xw) for k in ("bayes", "gatefree", "sourcewise")}
    ref = {}
    for k in ("bayes", "gatefree", "sourcewise"):
        R0 = D["R"][k]; ref[k] = dict(NLL=float(-(D["R"]["bayes"] * np.log(R0)).sum(-1).mean()), gateflip_follow=GL.following(R0, D["Rf"][k], D["R"]["bayes"], D["Rf"]["bayes"])[0],
                                     swap_follow_gatefree=GL.following(R0, D["Rw"][k], D["R"]["gatefree"], D["Rw"]["gatefree"])[0])
    D["ref"] = ref; return D
if __name__ == "__main__":
    out, stream, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); D = dataset(stream, n); res = dict(ref=D["ref"], models={})
    print("references", {k: {a: round(b, 3) for a, b in v.items()} for k, v in D["ref"].items()}, flush=True)
    for src in sys.argv[4:]:
        items = [(f"{os.path.basename(src)}@{k}", os.path.join(src, f"step{k}.npz")) for k in GL.snap_steps(src)] if os.path.isdir(src) else [(os.path.basename(src), src)]
        for name, path in items:
            r = evaluate(GL.load(path), D); res["models"][name] = r; print(name, {a: round(b, 3) for a, b in r.items()}, flush=True)
    json.dump(res, open(out, "w"), indent=1)
