"""XVIII fail-closed scorer (design v2, PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md §10.3-§10.4).
Usage: score_XVIII.py [DESIGN.json] [MEASURE.jsonl] [OUT.json]  (defaults: logs/XVIII_design.json, logs/measure_c18.jsonl, logs/score_XVIII.json)
Refuses (exit 2, no output) if any designed run lacks a measurement, a run is measured twice, a measurement is on the wrong stream or
parent, or a required field / step is missing.  Never rescales thresholds."""
import json, os, sys, math
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, "..")
DES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(P, "logs", "XVIII_design.json"); MEA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(P, "logs", "measure_c18.jsonl")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(P, "logs", "score_XVIII.json")
REQ = ("nec_E", "nec_M", "KL", "acc", "cls2", "mix_mass", "lik_RO1", "rhoL", "lapseL")
def refuse(msg): print("REFUSED:", msg); sys.exit(2)
d = json.load(open(DES)); recs = {}
if not os.path.exists(MEA): refuse("no measurement file")
for l in open(MEA):
    r = json.loads(l)
    if r["run"] in recs: refuse(f"duplicate measurement {r['run']}")
    recs[r["run"]] = r
runs = {}
for key, spec in d["runs"].items():
    r = recs.get(spec["tag"])
    if r is None: refuse(f"missing measurement for {key} ({spec['tag']})")
    if r["stream"] != d["stream"] or r["parent"] != d["parent"]: refuse(f"wrong stream/parent for {key}")
    for k in ("1000", "1211"):
        if k not in r["final"]: refuse(f"missing step {k} for {key}")
        if any(f not in r["final"][k] for f in REQ): refuse(f"missing field at step {k} for {key}")
    if "100" not in r["traj"]: refuse(f"missing step-100 trajectory for {key}")
    runs[key] = r
ell = lambda x: math.log(max(x, 0.0) + 0.01); OPP = {"EARLY": "MIXTURE", "MIXTURE": "EARLY"}
def phi_nec(a, b, k):
    v = [(ell(a[f"nec_{r}"]) - ell(b[f"nec_{r}"])) / (ell(k[f"nec_{r}"]) - ell(b[f"nec_{r}"])) if abs(ell(k[f"nec_{r}"]) - ell(b[f"nec_{r}"])) > 1e-9 else float("nan") for r in ("E", "M")]
    return 0.5 * (v[0] + v[1])
def band(phi, cls, bcls):
    if phi >= 0.8 and cls == OPP.get(bcls): return "F"
    if phi <= 0.2 and cls == bcls: return "N"
    return "P"
def competent(a, b): return bool(a["KL"] <= 0.045 and a["acc"] >= b["acc"] - 0.03)
EDGEKEY = {"E": "own0_mean", "M": "own2_mean"}
out = dict(pairs={}, arms={}, hypotheses={}, h3_E={}, stability_continuations=[], clamp_checks={})
for b in d["bases"]:
    for D in d["orders"]:
        pk = f"{b}|{D}"; B = runs[f"{pk}|baseline"]; K = runs[f"{pk}|block"]; Bf, Kf = B["final"]["1211"], K["final"]["1211"]
        ev = bool(competent(Bf, Bf) and Bf["cls2"] in OPP and competent(Kf, Bf) and Kf["cls2"] == OPP[Bf["cls2"]])
        pr = dict(evaluable=ev, baseline_cls=Bf["cls2"], block_cls=Kf["cls2"], arms={})
        A_base, A_blk = B["traj"]["100"][EDGEKEY[b]], K["traj"]["100"][EDGEKEY[b]]
        for arm in d["arms"]:
            if arm in ("baseline", "block"): continue
            R = runs[f"{pk}|{arm}"]; x = {}
            for st in ("1000", "1211"):
                a, bb, kk = R["final"][st], B["final"][st], K["final"][st]; ph = phi_nec(a, bb, kk)
                x[st] = dict(phi_nec=ph, cls2=a["cls2"], band=band(ph, a["cls2"], bb["cls2"]), competent=competent(a, bb))
            prim = x["1211"]["band"] if x["1211"]["competent"] else "MISSING"
            phs = (A_base - R["traj"]["100"][EDGEKEY[b]]) / (A_base - A_blk) if abs(A_base - A_blk) > 1e-9 else float("nan")
            med = "F" if phs >= 0.8 else ("N" if phs <= 0.2 else "P")
            rec = dict(primary=prim, unstable=bool(x["1000"]["band"] != x["1211"]["band"]), phi_nec=x["1211"]["phi_nec"], phi_nec_1000=x["1000"]["phi_nec"], cls2=x["1211"]["cls2"],
                       competent=x["1211"]["competent"], KL=R["final"]["1211"]["KL"], phi_share=phs, mediator_band=med)
            if "window_log" in R and "slog" in B:
                wl = [w[2] for w in R["window_log"] if 90 <= w[0] <= 99]; sl = [s_[2] for s_ in B["slog"]["shares"] if 90 <= s_[0] <= 99]
                rec["window_dloss_90_99"] = sum(wl) / len(wl) - sum(sl) / len(sl)
            if arm.startswith("sha"):
                tg = B["slog"]["shares"]; nh = len(tg[0][1]); err = []
                for w in R["window_log"]:
                    i = w[0]; tgt = tg[0][1] if arm == "sha_lo_real" else tg[min(i, len(tg) - 1)][1]; ach = w[8 + nh: 8 + 2 * nh]; err.append(max(abs(p - q) for p, q in zip(ach, tgt)))
                betas = [max(abs(v) for v in w[8:8 + nh]) for w in R["window_log"]]
                rec["clamp_max_err"] = max(err); rec["clamp_frac_beta_gt1"] = sum(bv > 1 for bv in betas) / len(betas); out["clamp_checks"][f"{pk}|{arm}"] = dict(max_err=rec["clamp_max_err"], frac_beta_gt1=rec["clamp_frac_beta_gt1"])
                if rec["clamp_max_err"] > 1e-3: rec["primary"] = "UNINTERPRETABLE"
            if rec["primary"] == "P" or rec["unstable"]: out["stability_continuations"].append(d["runs"][f"{pk}|{arm}"]["tag"])
            pr["arms"][arm] = rec
        out["pairs"][pk] = pr
for arm in d["arms"]:
    if arm in ("baseline", "block"): continue
    bands = [out["pairs"][pk]["arms"][arm]["primary"] for pk in out["pairs"] if out["pairs"][pk]["evaluable"]]
    defin = [x for x in bands if x in ("F", "P", "N")]
    if len(defin) < 5: pooled = "INCONCLUSIVE"
    else:
        top = max(("F", "P", "N"), key=defin.count); pooled = top if defin.count(top) >= 5 else "MIXED"
    per_base = {b: [out["pairs"][f"{b}|{D}"]["arms"][arm]["primary"] if out["pairs"][f"{b}|{D}"]["evaluable"] else "NONEVAL" for D in d["orders"]] for b in d["bases"]}
    out["arms"][arm] = dict(pooled=pooled, per_base=per_base, n_evaluable=len(bands), n_definite=len(defin), counts={k: bands.count(k) for k in set(bands)},
                            phi_nec=[round(out["pairs"][pk]["arms"][arm]["phi_nec"], 3) for pk in out["pairs"]])
PRED = {"H1d": dict(vrg="F", gsg="N", gsv="N", sha_hi_rolled="F", sha_lo_real="N"),
        "Hm": dict(vrg="F", gsg="med", gsv="med", sha_hi_rolled="N", sha_lo_real="F"),
        "H2g_R": dict(vrg="N", gsg="F", gsv="F", sha_hi_rolled="N", sha_lo_real="N"),
        "H2g_RV": dict(vrg="N", gsg="N", gsv="F", sha_hi_rolled="N", sha_lo_real="N"),
        "H4": dict(vrg="F", gsg="N", gsv="N", sha_hi_rolled="N", sha_lo_real="N"),
        "H1d_plus_Hm": dict(vrg="F", gsg="P", gsv="P", sha_hi_rolled="P", sha_lo_real="P")}
for h, pr_ in PRED.items():
    res = {}; contra = False; incon = False
    for arm, pb in pr_.items():
        obs = out["arms"][arm]["pooled"]
        if pb == "med":
            mism = [pk for pk in out["pairs"] if out["pairs"][pk]["evaluable"] and out["pairs"][pk]["arms"][arm]["primary"] in ("F", "P", "N") and out["pairs"][pk]["arms"][arm]["primary"] != out["pairs"][pk]["arms"][arm]["mediator_band"]]
            ok = "contradicted" if len(mism) >= 2 else ("inconclusive" if obs in ("INCONCLUSIVE",) else "consistent"); res[arm] = dict(pred="mediator", obs=obs, mismatched_pairs=mism, status=ok)
        else:
            ok = "inconclusive" if obs in ("INCONCLUSIVE", "MIXED") else ("consistent" if obs == pb else "contradicted"); res[arm] = dict(pred=pb, obs=obs, status=ok)
        contra |= res[arm]["status"] == "contradicted"; incon |= res[arm]["status"] == "inconclusive"
    out["hypotheses"][h] = dict(verdict="CONTRADICTED" if contra else ("INCONCLUSIVE" if incon else "CONSISTENT"), arms=res)
eE = [out["pairs"][f"E|{D}"]["arms"]["vrg"] for D in d["orders"] if out["pairs"][f"E|{D}"]["evaluable"]]
okE = [x for x in eE if x["primary"] == "F" and x.get("window_dloss_90_99", 9) <= 0.061]
out["h3_E"] = dict(rule="vrg F on >= 2 of 3 E pairs with window dloss(90-99) <= 0.061", n_ok=len(okE), dloss=[x.get("window_dloss_90_99") for x in eE], excluded=bool(len(okE) >= 2))
out["P1_graded"] = out["arms"]["gof23"]
json.dump(out, open(OUT, "w"), indent=1, default=float)
print(json.dumps({a: v["pooled"] for a, v in out["arms"].items()}), json.dumps({h: v["verdict"] for h, v in out["hypotheses"].items()}), "H3_E excluded:", out["h3_E"]["excluded"])
