"""XVIII fail-closed scorer (design v2, PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md §10.3-§10.4).
Usage: score_XVIII.py [DESIGN.json] [MEASURE.jsonl] [OUT.json]  (defaults: logs/XVIII_design.json, logs/measure_c18.jsonl, logs/score_XVIII.json)
Refuses (exit 2, no output) if any designed run lacks a measurement, a run is measured twice, a measurement is on the wrong stream or
parent, or a required field / step is missing.  Never rescales thresholds."""
import json, os, sys, math
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, "..")
DES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(P, "logs", "XVIII_design.json"); MEA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(P, "logs", "measure_c18.jsonl")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(P, "logs", "score_XVIII.json")
REQ = ("nec_E", "nec_M", "KL", "acc", "cls2", "cls2_struct", "cls2_causal", "mix_mass", "lik_RO1", "rhoL", "lapseL")
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
def sc(r): return r["cls2_struct"] if r["cls2_struct"] == r["cls2_causal"] else "HYBRID"      # struct + causal only (sensitivity, step-2546 readout)
def band_nb(phi, a, bb): return band(phi, sc(a), sc(bb))
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
                nanphi = ph != ph
                x[st] = dict(phi_nec=ph, cls2=a["cls2"], band="MISSING" if nanphi else band(ph, a["cls2"], bb["cls2"]), band_nb="MISSING" if nanphi else band_nb(ph, a, bb), competent=competent(a, bb))
            prim = x["1211"]["band"] if x["1211"]["competent"] else "MISSING"; prim_nb = x["1211"]["band_nb"] if x["1211"]["competent"] else "MISSING"
            phs = (A_base - R["traj"]["100"][EDGEKEY[b]]) / (A_base - A_blk) if abs(A_base - A_blk) > 1e-9 else float("nan")
            med = "F" if phs >= 0.8 else ("N" if phs <= 0.2 else "P")
            rec = dict(primary=prim, primary_nb=prim_nb, unstable=bool(x["1000"]["band"] != x["1211"]["band"]), phi_nec=x["1211"]["phi_nec"], phi_nec_1000=x["1000"]["phi_nec"], cls2=x["1211"]["cls2"],
                       competent=x["1211"]["competent"], KL=R["final"]["1211"]["KL"], phi_share=phs, mediator_band=med)
            if "window_log" in R and "slog" in B:
                wl = [w[2] for w in R["window_log"] if 90 <= w[0] <= 99]; sl = [s_[2] for s_ in B["slog"]["shares"] if 90 <= s_[0] <= 99]
                rec["window_dloss_90_99"] = sum(wl) / len(wl) - sum(sl) / len(sl)
            if arm.startswith("sha"):
                tg = B["slog"]["shares"]; nh = len(tg[0][1]); err = []
                for w in R["window_log"]:
                    i = w[0]; tgt = tg[0][1] if arm == "sha_lo_real" else tg[min(i, len(tg) - 1)][1]; ach = w[8 + nh: 8 + 2 * nh]; err.append(max(abs(p - q) for p, q in zip(ach, tgt)))
                betas = [max(abs(v) for v in w[8:8 + nh]) for w in R["window_log"]]
                rec["clamp_max_err"] = max(err); rec["clamp_frac_beta_gt1"] = sum(bv > 1 for bv in betas) / len(betas)
                w99 = [w for w in R["window_log"] if w[0] == 99][0]; rec["beta_99"] = w99[8:8 + nh]; rec["fwd_share_99"] = sum(w99[8 + nh:8 + 2 * nh]) / nh
                rec["release_jump"] = R["traj"]["100"][EDGEKEY[b]] - rec["fwd_share_99"]            # latent share at 100 minus clamped forward share at 99 (review B addendum)
                out["clamp_checks"][f"{pk}|{arm}"] = dict(max_err=rec["clamp_max_err"], frac_beta_gt1=rec["clamp_frac_beta_gt1"], beta_99=rec["beta_99"], release_jump=rec["release_jump"])
                if rec["clamp_max_err"] > 1e-3: rec["primary"] = rec["primary_nb"] = "UNINTERPRETABLE"
            if rec["primary"] == "P" or rec["unstable"]:
                for a_ in (arm, "baseline", "block"):
                    tg_ = d["runs"][f"{pk}|{a_}"]["tag"]
                    if tg_ not in out["stability_continuations"]: out["stability_continuations"].append(tg_)
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
        "Hm_window": dict(vrg="med", gsg="med", gsv="med", sha_hi_rolled="N", sha_lo_real="F"),
        "Hm_release": dict(vrg="med", gsg="med", gsv="med", sha_hi_rolled="med", sha_lo_real="med"),
        "H2g_R": dict(vrg="N", gsg="F", gsv="F", sha_hi_rolled="N", sha_lo_real="N"),
        "H2g_RV": dict(vrg="N", gsg="N", gsv="F", sha_hi_rolled="N", sha_lo_real="N"),
        "H4": dict(vrg="F", gsg="N", gsv="N", sha_hi_rolled="N", sha_lo_real="N"),
        "H1d_plus_Hm": dict(vrg="F", gsg="P", gsv="P", sha_hi_rolled="P", sha_lo_real="P"),
        "H1d_plus_H2g": dict(vrg="F", gsg="P", gsv="PF", sha_hi_rolled="F", sha_lo_real="N")}
def judge(key, pairs, n_cons, n_contra):
    res = {}
    for h, pr_ in PRED.items():
        arms_ = {}
        for arm, pb in pr_.items():
            mt = mm = 0; det = []
            for pk in pairs:
                rec = out["pairs"][pk]["arms"][arm]; obs = rec[key]
                if obs not in ("F", "P", "N"): continue
                allowed = rec["mediator_band"] if pb == "med" else pb
                ok = obs in allowed; mt += ok; mm += (not ok); det.append((pk, obs, allowed))
            st = "contradicted" if mm >= n_contra else ("consistent" if mt >= n_cons else "inconclusive")
            arms_[arm] = dict(pred=pb, matches=mt, mismatches=mm, detail=det, status=st)
        stt = [a["status"] for a in arms_.values()]
        res[h] = dict(verdict="CONTRADICTED" if "contradicted" in stt else ("CONSISTENT" if all(x == "consistent" for x in stt) else "INCONCLUSIVE"), arms=arms_)
    return res
evp = [pk for pk in out["pairs"] if out["pairs"][pk]["evaluable"]]
out["hypotheses"] = judge("primary", evp, 5, 2); out["hypotheses_sensitivity_no_behaviour"] = judge("primary_nb", evp, 5, 2)
out["hypotheses_per_base"] = {b: judge("primary", [pk for pk in evp if pk.startswith(b + "|")], 3, 2) for b in d["bases"]}
H3f = os.path.join(os.path.dirname(MEA), "h3_immediate_c18.json")
if not os.path.exists(H3f): refuse("missing h3_immediate_c18.json")
H3 = json.load(open(H3f)); eE = [D_ for D_ in d["orders"] if out["pairs"][f"E|{D_}"]["evaluable"]]
okE = [D_ for D_ in eE if out["pairs"][f"E|{D_}"]["arms"]["vrg"]["primary"] == "F" and H3[f"E|{D_}"]["vrg"] < 0.18]
out["h3_E"] = dict(rule="vrg F on >= 2 of 3 E pairs with immediate dloss at the paired baseline's step-100 parameters < 0.18 (XIV non-flipping E noise)",
                   immediate_dloss=H3, n_ok=len(okE), excluded=bool(len(okE) >= 2))
out["P1_graded"] = out["arms"]["gof23"]
json.dump(out, open(OUT, "w"), indent=1, default=float)
print(json.dumps({a: v["pooled"] for a, v in out["arms"].items()}), json.dumps({h: v["verdict"] for h, v in out["hypotheses"].items()}), "H3_E excluded:", out["h3_E"]["excluded"])
