"""EXPLORATORY follow-ups (after the frozen verdicts; not used to change any verdict).
X1  REL-XNOR TF1 models whose pre-MLP mediator was source-additive: run the same protocol at mlp_hidden0@R.
X2  Interaction form of every RELATION_COMPUTING mediator: R^2 of the interaction residual (S-coords minus best
    additive fit, on the grid) explained by candidate functions: sign(ab), ab, sign(ab)*sign(a+b), target y.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ckpt, generators as G, mediator as MD, run_protocol as RP

out = {"X1": {}, "X2": {}}
# X1
RP.PRIMARY["TF1"] = [("mlp_hidden0", "R")]
for tag, ckd, seeds in (("E1", "checkpoints/historical", (2, 3, 4)), ("E2", "checkpoints/new", (7, 8, 9))):
    d = json.load(open(ROOT / "logs" / f"protocol_{tag}.json"))
    for s in seeds:
        k = f"REL-XNOR/TF1/s{s}/train"
        if d[k]["resid_mid0@R"]["verdict"] == "SOURCE_ADDITIVE_MEDIATOR":
            r = RP.analyse("REL-XNOR", "TF1", s, ROOT / ckd, "train")["mlp_hidden0@R"]
            out["X1"][k] = {x: r[x] for x in ("verdict", "k", "iia_full_site", "iia", "necessity", "sufficiency", "additive_R2", "surrogates")}
            print("X1", k, r["verdict"], "k", r["k"], "cut", r["iia_full_site"], "S_rel", r["iia"]["S_rel"], "rand", round(r["iia"]["S_rand_mean"], 3),
                  "nec", r["necessity"]["flip_after"], "suff", r["sufficiency"], "addR2", round(r["additive_R2"], 3), flush=True)
# X2
for tag, ckd in (("E1", "checkpoints/historical"), ("E2", "checkpoints/new")):
    d = json.load(open(ROOT / "logs" / f"protocol_{tag}.json"))
    for key, sites in d.items():
        task, arch, s, dist = key.split("/")
        if task.startswith("K3"):
            continue
        for site_name, r in sites.items():
            if r.get("verdict") != "RELATION_COMPUTING_MEDIATOR":
                continue
            hook, pos = site_name.split("@")
            P = ckpt.load(ROOT / ckd / f"{MD.CODE[task]}_{arch}_s{s[1:]}.npz")
            site = MD.Site(P, task, hook, pos)
            A, C = RP.pairs(task, 800, RP.SEEDS["pairs_fit"])
            U, k, _ = MD.s_rel(site, A, C)
            gc = MD.GridCoords(site, U)
            g = gc.g; Aa, Bb = np.meshgrid(g, g, indexing="ij")
            y, _ = G.target(task, Aa.ravel(), Bb.ravel(), np.zeros(Aa.size))
            feats = {"sign(ab)": np.sign(Aa * Bb).ravel(), "ab": (Aa * Bb).ravel(),
                     "sign(ab)sign(a+b)": (np.sign(Aa * Bb) * np.sign(Aa + Bb)).ravel(), "target_y": y}
            best = {}
            for Z in gc.C.values():
                Rz = Z - (Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean((0, 1), keepdims=True))
                Rz = Rz.reshape(-1, Rz.shape[-1])
                for nm, f in feats.items():
                    X = np.column_stack([np.ones_like(f), f])
                    res = Rz - X @ np.linalg.lstsq(X, Rz, rcond=None)[0]
                    best[nm] = float(1 - res.var(0).sum() / Rz.var(0).sum())
            out["X2"][f"{key}/{site_name}"] = best
            print("X2", key, site_name, {k_: round(v, 3) for k_, v in best.items()}, flush=True)
(ROOT / "logs" / "explore_followups.json").write_text(json.dumps(out, indent=1, default=float))
