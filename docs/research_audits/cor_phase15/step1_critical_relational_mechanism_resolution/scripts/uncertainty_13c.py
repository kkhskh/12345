"""XIII-C principal uncertainty estimates (post-freeze; reported next to, never replacing, the frozen verdicts).
(1) Episode-bootstrap (B = 200) 95% intervals for lik_RO1 and mix_mass of every R1/R2 confirmation unit and its references, on the confirmation
    stream (88_100_000; same pair sets as unit_metrics), and for the R1 distances d_init and d_data (paired over the same pairs/episodes).
(2) The second-stream sensitivity of the pre-declared borderline procedure: lik_RO1 and mix_mass on stream 88_150_000.
(3) rho_KL bootstrap ranges and route_Q stream sd are read from fair_fits_conf.json and d_rows_conf.json.
Usage: uncertainty_13c.py -> logs/uncertainty_13c.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import xc, ftlib as F, relpairs as RP, metrics as MT
P = json.load(open(os.path.join(HERE, "predictions_13c.json"))); T = {p["id"]: p["thresholds"] for p in P["predictions"]}
units = sorted({u for c in T["R1"]["cells"] + T["R1"]["split_cells"] for u in c} | {u for c in T["R2"]["ro0_units"] + T["R2"]["l2q_units"] for u in c})
def per_item(M, S):
    PL = S["PL"]; m = len(PL["q"]); i = np.arange(m); rD = {}; F.forward(M, jnp.asarray(PL["Xd"]), rec=rD); ro = 36 + PL["q"]; rd = np.asarray(rD["resid1"])
    L0 = np.asarray(F.forward(M, jnp.asarray(PL["Xb"])))[i, PL["q"]]; L = np.asarray(F.forward(M, jnp.asarray(PL["Xb"]), resid_patch={1: lambda x: x.at[i, ro].set(rd[i, ro])}))[i, PL["q"]]
    o = (RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0))); p = RP.cen(PL["PBd"]) - RP.cen(PL["PB"])
    rec = {}; X = S["Xn"][:1500]; q = S["qn"][:1500]; F.forward(M, jnp.asarray(X), rec=rec); A = np.asarray(rec["pattern2"]); k = np.arange(1500); qs = q[:, None] * 6 + np.arange(6)[None]
    mass = np.stack([np.take_along_axis(A[k, h, 36 + q], qs, 1).sum(1) for h in range(4)], 1)       # per-episode mass per head
    return o, p, mass
def stat(o, p, mass, ip, ie):
    oo, pp = o[ip].ravel(), p[ip].ravel(); return float((oo * pp).sum() / (pp * pp).sum()), float(mass[ie].mean(0).max())
out = dict(boot={}, second_stream={})
S1 = MT.sets(88_100_000); S2 = MT.sets(88_150_000); rng = np.random.default_rng(7); B = 200; data = {}
for u in units:
    try: M = xc.load(u)
    except Exception as ex: out["boot"][u] = f"missing ({ex})"; continue
    o, p, mass = per_item(M, S1); n, ne = len(o), len(mass); idx = [(rng.integers(0, n, n), rng.integers(0, ne, ne)) for _ in range(B)]
    data[u] = (o, p, mass)
    bs = np.array([stat(o, p, mass, a, b) for a, b in idx]); pt = stat(o, p, mass, np.arange(n), np.arange(ne))
    out["boot"][u] = dict(lik_RO1=pt[0], mix_mass=pt[1], lik_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), mix_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())
    o2, p2, m2 = per_item(M, S2); out["second_stream"][u] = dict(zip(("lik_RO1", "mix_mass"), stat(o2, p2, m2, np.arange(len(o2)), np.arange(len(m2)))))
# paired bootstrap of R1 distances (same pair/episode indices for cell and references: identical inputs)
out["R1_distances"] = {}
for cell, I, D in T["R1"]["cells"] + [[c, a, b] for c, a, b in T["R1"]["split_cells"]]:
    if not all(x in data for x in (cell, I, D)): continue
    n, ne = len(data[cell][0]), len(data[cell][2]); dI, dD = [], []
    for _ in range(B):
        a, b = rng.integers(0, n, n), rng.integers(0, ne, ne); v = {x: np.array(stat(*data[x], a, b)) for x in (cell, I, D)}
        dI.append(np.linalg.norm(v[cell] - v[I])); dD.append(np.linalg.norm(v[cell] - v[D]))
    dI, dD = np.array(dI), np.array(dD)
    out["R1_distances"][cell] = dict(d_first_ref_ci=np.percentile(dI, [2.5, 97.5]).tolist(), d_second_ref_ci=np.percentile(dD, [2.5, 97.5]).tolist(), frac_boot_first_closer=float((dI < dD).mean()))
for nm, f in (("fair_fits", "fair_fits_conf.json"), ("d_rows", "d_rows_conf.json")):
    p_ = os.path.join(HERE, "..", "logs", f)
    if not os.path.exists(p_): continue
    d = json.load(open(p_))
    if nm == "fair_fits": out["rho_KL_boot"] = {k: v.get("bootstrap", {}).get("PM") for k, v in d.items() if k != "BAYES_SELF"}
    else: out["route_Q_streams"] = {k: dict(mean=v["routes_streams"]["route_Q_mean"], sd=v["routes_streams"]["route_Q_sd"]) for k, v in d.items()}
json.dump(out, open(os.path.join(HERE, "..", "logs", "uncertainty_13c.json"), "w"), indent=1); print(json.dumps(out["R1_distances"], indent=1))
