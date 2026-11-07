"""Adversarial causal checks (reviewer R_causal). Does not modify any package file."""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import ckpt, generators as G, mediator as MD, run_protocol as RP

MODELS = [("historical", 2), ("historical", 4), ("new", 7), ("new", 8), ("new", 9), ("historical", 3)]
TASK = sys.argv[1] if len(sys.argv) > 1 else "REL-XNOR"
HOOK = sys.argv[2] if len(sys.argv) > 2 else "resid_mid1"
ARCH = sys.argv[3] if len(sys.argv) > 3 else "TF2"
POS = "R" if HOOK != "resid_post0" else "ALL"
out = {}

def nn_dist(x, bank):
    d = ((x[:, None, :] - bank[None]) ** 2).sum(-1)
    return np.sqrt(d.min(1)), d.argmin(1)

def custom_pairs(z, da, dc):
    p = np.zeros_like(z)
    A = dict(a=(z + da) / 2, b=(z - da) / 2, p=p); C = dict(a=(z + dc) / 2, b=(z - dc) / 2, p=p)
    for X in (A, C):
        X["y"], _ = G.target(TASK, X["a"], X["b"], X["p"]); X["z"] = X["a"] + X["b"]; X["s"] = (X["a"] * X["b"] > 0).astype(int)
    return A, C

for sub, s in MODELS:
    f = ROOT / "checkpoints" / sub / f"{MD.CODE[TASK]}_{ARCH}_s{s}.npz"
    if not f.exists():
        continue
    P = ckpt.load(f); site = MD.Site(P, TASK, HOOK, POS)
    A, C = RP.pairs(TASK, 800, RP.SEEDS["pairs_fit"]); Ae, Ce = RP.pairs(TASK, 500, RP.SEEDS["pairs_eval"])
    bank = RP.sample(TASK, 1500, RP.SEEDS["bank"], "train")
    big = RP.sample(TASK, 6000, 777, "train")
    U, k, _ = MD.s_rel(site, A, C)
    r = dict(k=k)
    lgb, hb = site.get(bank); _, hbig = site.get(big)
    # (1) content of U: correlations on natural inputs
    c = hb @ U
    feats = {"a": bank["a"], "b": bank["b"], "a-b": bank["a"] - bank["b"], "|a-b|": np.abs(bank["a"] - bank["b"]),
             "sign(ab)": np.sign(bank["a"] * bank["b"]), "model_margin": MD.margin(lgb)}
    r["corr_U0"] = {n: float(abs(np.corrcoef(c[:, 0], v)[0, 1])) for n, v in feats.items()}
    # (1b) readout alignment: effective direct readout direction gf*(WU0-WU1), projected off the mean direction
    w = P["gf"] * (P["WU"][:, 0] - P["WU"][:, 1]); w = w - w.mean(); w /= np.linalg.norm(w)
    if POS == "R":
        Uc = U - U.mean(0, keepdims=True)
        r["cos_U_readout"] = float(np.linalg.svd((Uc / np.linalg.norm(Uc, axis=0)).T @ w[:, None], compute_uv=False).max())
        r["cos_U_readout_raw"] = float(np.abs(U.T @ w).max())
    # (3) manifold: alternative normaliser = typical within-pair distance
    _, hAe = site.get(Ae); _, hCe = site.get(Ce)
    patched = hAe + (hCe - hAe) @ U @ U.T
    dp, _ = nn_dist(patched[:200], hbig)
    dnat, _ = nn_dist(hAe[:200], hbig)
    r["manifold"] = dict(patched_nn_median=float(np.median(dp)), natural_nn_median_bigbank=float(np.median(dnat)),
                         pair_dist_median=float(np.median(np.linalg.norm(hAe - hCe, axis=1))),
                         patched_over_pairdist=float(np.median(dp / np.linalg.norm(hAe - hCe, axis=1)[:200])),
                         U_jump_median=float(np.median(np.abs(((hAe - hCe) @ U)).sum(1))))
    # (3b) on-manifold check: near-boundary matched pairs (inputs differ by 2*eps*|z|), interchange IIA and manifold
    rng = np.random.default_rng(91)
    res3 = {}
    for eps in (0.02, 0.05, 0.1, 0.3):
        z = rng.uniform(0.5, 2.5, 400) * rng.choice([-1, 1], 400); sg = rng.choice([-1, 1], 400)
        Ab, Cb = custom_pairs(z, sg * np.abs(z) * (1 - eps), sg * np.abs(z) * (1 + eps))
        iia, n = MD.iia(site, Ab, Cb, U)
        _, hA2 = site.get(Ab); _, hC2 = site.get(Cb)
        pt = hA2 + (hC2 - hA2) @ U @ U.T
        dpt, idx = nn_dist(pt[:200], hbig); dn, _ = nn_dist(hA2[:200], hbig)
        # decision of patched vs decision of nearest natural state
        lgp = site.run_with(Ab, pt); dpat = MD.decision(TASK, lgp)[:200]
        dnear = MD.decision(TASK, site.get({kk: big[kk][idx] for kk in ("a", "b", "p")})[0])
        res3[str(eps)] = dict(iia=iia, n=n, patched_nn=float(np.median(dpt)), natural_nn=float(np.median(dn)),
                              ratio=float(np.median(dpt) / np.median(dn)), pair_dist=float(np.median(np.linalg.norm(hA2 - hC2, axis=1))),
                              patched_vs_nearest_natural_decision_agree=float((dpat == dnear).mean()),
                              nearest_natural_has_source_s=float((big["s"][idx] == Cb["s"][:200]).mean()))
    r["near_boundary"] = res3
    # (3c) unmatched natural-value interchange: U from random bank input of opposite model decision, any z
    dbank = MD.decision(TASK, lgb)
    rng = np.random.default_rng(5)
    i = rng.integers(0, 1500, 1000); j = rng.integers(0, 1500, 1000)
    m = dbank[i] != dbank[j]; i, j = i[m], j[m]
    base = {kk: bank[kk][i] for kk in ("a", "b", "p")}; hbase = hb[i]; hsrc = hb[j]
    lg = site.run_with(base, hbase + (hsrc - hbase) @ U @ U.T)
    r["unmatched_interchange_iia"] = float((MD.decision(TASK, lg) == dbank[j]).mean())
    # (4) necessity variants
    dA0, dC0 = MD.decision(TASK, site.get(Ae)[0]), MD.decision(TASK, site.get(Ce)[0]); diff = dA0 != dC0
    def ablate(newU_A, newU_C):
        nA = hAe - (hAe @ U) @ U.T + newU_A @ U.T; nC = hCe - (hCe @ U) @ U.T + newU_C @ U.T
        lA, lC = site.run_with(Ae, nA), site.run_with(Ce, nC)
        dA, dC = MD.decision(TASK, lA), MD.decision(TASK, lC)
        return dict(flip_after=float((dA != dC)[diff].mean()),
                    acc_after=float(np.concatenate([dA == Ae["y"], dC == Ce["y"]]).mean()),
                    median_abs_margin_after=float(np.median(np.abs(np.concatenate([MD.margin(lA), MD.margin(lC)])))),
                    median_abs_margin_before=float(np.median(np.abs(np.concatenate([MD.margin(site.get(Ae)[0]), MD.margin(site.get(Ce)[0])])))))
    pm = ((hAe + hCe) / 2) @ U
    nec = {"pair_mean": ablate(pm, pm)}
    bm = np.broadcast_to(hb.mean(0) @ U, pm.shape)
    nec["bank_mean"] = ablate(bm, bm)
    # resample: U-coord of a third natural input with same z (shared by both members)
    z = Ae["z"]; rng = np.random.default_rng(11)
    dd = rng.uniform(-1, 1, len(z)) * np.minimum(6 - np.abs(z), 6) * 0.98
    dd = np.clip(dd, -(6 - np.abs(z)) * 0.99, (6 - np.abs(z)) * 0.99)
    third = dict(a=(z + dd) / 2, b=(z - dd) / 2, p=Ae["p"])
    _, h3 = site.get(third); u3 = h3 @ U
    nec["resample_same_z_shared"] = ablate(u3, u3)
    nec["resample_same_z_shared"]["third_s_rate"] = float((third["a"] * third["b"] > 0).mean())
    # independent resamples per member (should give ~random agreement if U decides)
    dd2 = np.clip(rng.uniform(-1, 1, len(z)) * 6, -(6 - np.abs(z)) * 0.99, (6 - np.abs(z)) * 0.99)
    fourth = dict(a=(z + dd2) / 2, b=(z - dd2) / 2, p=Ae["p"]); _, h4 = site.get(fourth)
    o = ablate(u3, h4 @ U)
    s3 = np.sign(third["a"] * third["b"]); s4 = np.sign(fourth["a"] * fourth["b"])
    o["expected_flip_if_U_decides"] = float((s3 != s4)[diff].mean())
    nec["resample_same_z_independent"] = o
    r["necessity"] = nec
    out[f"{TASK}/{ARCH}/s{s}/{HOOK}"] = r
    print(json.dumps({f"{TASK}/{ARCH}/s{s}/{HOOK}": r}, default=float), flush=True)

(Path(__file__).parent / f"checks_{TASK}_{ARCH}_{HOOK}.json").write_text(json.dumps(out, indent=1, default=float))
