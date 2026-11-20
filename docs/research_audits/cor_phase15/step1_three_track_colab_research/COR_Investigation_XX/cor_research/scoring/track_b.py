"""Track B frozen scorer (v2 after reviews A B3/B4/M8-M11 and B BLK-1/2, MAJ-1/2/6/7).  python -m cor_research.scoring.track_b PRESET OUT.json [--root R]
Units: confirmation anchors with a trackB/<run>.json battery record.  A unit is EVALUABLE at a level ('matched' | 'final') iff the level
exists, the model is competent and causally pure there (labels recomputed from the recorded necessity numbers).  Parents are the
independent units: every statistic is one number per parent, inference by t-intervals on parent means (exact sign-flip p reported);
classifications with < 6 parents are labelled 'indicative'.  Tests and the classification are protocol.TRACK_B (frozen).
Fail-closed: battery job not done / output hash differs, checkpoint hash != registry snapshot hash, or code hash != frozen -> PROVENANCE_FAIL."""
import os, sys, json, itertools, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import numpy as np
from scipy import stats
from cor_research import env as E, protocol as PR
from cor_research.scoring.track_a import cls, comp, sign_flip_p
P = PR.full(); RULES = ("local", "noelig", "nocount", "maprole"); DIR = ("local", "noelig")   # directed rules (tie-aware; review B r2 N1)


def load_units(preset, root, frozen_code):
    from cor_research.jobs import plan as PL, runner as RN
    D = PL.design(preset, root); R = RN.load_all(root); U = []; problems = []
    for key, tag in D["tags"].items():
        if not key.endswith((":fullB", ":fullI")): continue
        rel = os.path.join("trackB", tag + ".json"); f = os.path.join(root, rel); parent, mid, arm = key.split(":")
        if not os.path.exists(f): U.append(dict(key=key, parent=parent, base=mid[0], D=int(mid[1:]), arm=arm, status="MISSING")); continue
        r = json.load(open(f)); bj = R.get("fbbat:" + key); cj = R.get("cont:" + key, {})
        if bj is None or bj.get("status") != "done" or bj.get("output_hashes", {}).get(rel) != E.sha256(f): problems.append(f"{rel}: battery job not registered done with this output hash")
        if frozen_code is not None and r.get("code_sha256") != frozen_code: problems.append(f"{rel}: Track B code hash differs from the frozen hash")
        for lev, a in r["analyses"].items():
            if a.get("missing"): continue
            h = cj.get("snapshot_hashes", {}).get(f"step{a['step']}.npz")
            if h is None or h != a["necessity"]["ckpt_sha256"] or h != a["functional"]["ckpt_sha256"]: problems.append(f"{rel}: {lev} checkpoint hash differs from the registry snapshot hash")
        U.append(dict(key=key, parent=parent, base=mid[0], D=int(mid[1:]), arm=arm, tag=tag, rec=r, status="PRESENT"))
    return D, U, problems


def level(u, lev, root):
    a = u.get("rec", {}).get("analyses", {}).get(lev)
    if a is None or a.get("missing"): return dict(evaluable=False, reason="censored_or_missing")
    n = a["necessity"]
    if not comp(n) or cls(n) == "HYBRID": return dict(evaluable=False, reason="noncompetent" if not comp(n) else "hybrid")
    f = a["functional"]; zp = os.path.join(root, "trackB", u["tag"] + f"_{lev}.npz")
    if a.get("decisions_sha256") is None or E.sha256(zp) != a["decisions_sha256"]: raise RuntimeError(f"{zp}: decision file hash differs from its battery record")   # review C r2 N5
    z = np.load(zp)
    return dict(evaluable=True, cls=cls(n), step=a["step"], KL=f["natural"]["KL"], w={k: f["natural"]["rules"][k]["w"] for k in RULES},
                w_par_null={k: f["natural"]["nulls"]["parametric"]["rules"][k]["w"] for k in RULES}, oqi2=f["interchange"]["OQI_2"]["IIA_norm"], ic=f["interchange"],
                dec=z["dec"], rule={k: z[f"rule_{k}"] for k in ("bayes",) + RULES}, base=u["base"], arm=u["arm"])


def tci(vals, conf=0.95):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) < 2: return dict(n=len(v), mean=(float(v.mean()) if len(v) else None), ci=None, sign_flip_p=None)
    m, se = v.mean(), v.std(ddof=1) / math.sqrt(len(v)); t = stats.t.ppf(0.5 + conf / 2, len(v) - 1)
    tp = float(2 * stats.t.sf(abs(m / se), len(v) - 1)) if se > 0 else (0.0 if m != 0 else 1.0)
    return dict(n=len(v), mean=float(m), ci=[float(m - t * se), float(m + t * se)], t_p=tp, sign_flip_p=sign_flip_p(list(v)), parent_values=[float(x) for x in v])


def holm(ps, alpha):
    order = sorted((p if p is not None else 1.0, k) for k, p in ps.items()); out = {}; ok = True
    for i, (p, k) in enumerate(order):
        ok = ok and p <= alpha / (len(order) - i); out[k] = ok
    return out


def score_level(U, lev, root, th, technical):
    L = {u["key"]: (u, level(u, lev, root)) for u in U if u["status"] == "PRESENT"}; ev = {k: (u, l) for k, (u, l) in L.items() if l["evaluable"]}
    byp = {}
    for k, (u, l) in ev.items(): byp.setdefault(u["parent"], []).append((u, l))
    both = sorted(p for p, it in byp.items() if {l["cls"] for _, l in it} >= {"EARLY", "MIXTURE"})
    res = dict(n_units=len(L), n_evaluable=len(ev), parents_with_both_classes=both,
               censoring_by_class={c: sum(1 for _, (u, l) in L.items() if not l["evaluable"] and ((u["base"] == "E") == (u["arm"] == "fullB")) == (c == "EARLY")) for c in ("EARLY", "MIXTURE")},
               matched_step_by_class={c: [l["step"] for _, l in ev.values() if l["cls"] == c] for c in ("EARLY", "MIXTURE")})
    # HB_c primary: same unit (parent, base), D1 != D2 in BOTH terms
    def dstat(l, k):                                                                  # precision-adjusted directed agreement (review B r2 N2)
        S = (l["rule"][k] >= 0) & (l["rule"][k] != l["rule"]["bayes"]) & (l["rule"]["bayes"] >= 0)
        if not S.any(): return float("nan")
        d = l["dec"][S]; return float((d == l["rule"][k][S]).mean() - ((d != l["rule"][k][S]) & (d != l["rule"]["bayes"][S])).mean())
    hbc, hbc2, dirs, hist_conf = {}, {}, {k: {} for k in DIR}, {}
    for p in both:
        bt, wt, b2, w2 = [], [], [], []
        for (u1, l1), (u2, l2) in itertools.combinations(byp[p], 2):
            dis = float((l1["dec"] != l2["dec"]).mean())
            if u1["base"] == u2["base"] and u1["D"] != u2["D"]: (bt if l1["cls"] != l2["cls"] else wt).append(dis)
            if u1["base"] != u2["base"] and u1["D"] == u2["D"]: (b2 if l1["cls"] != l2["cls"] else w2).append(dis)
        prim = np.mean(bt) - np.mean(wt) if bt and wt else None; sec = np.mean(b2) - np.mean(w2) if b2 and w2 else None
        if prim is not None and sec is not None: hbc[p] = 0.5 * (prim + sec); hist_conf[p] = False      # the same-D cross-unit term is history-matched: the average cancels history (review A r2 N-B2)
        elif prim is not None: hbc[p] = prim; hist_conf[p] = True
        if sec is not None: hbc2[p] = sec
        for k in DIR:
            ag = {c: [dstat(l, k) for _, l in byp[p] if l["cls"] == c] for c in ("EARLY", "MIXTURE")}
            if ag["EARLY"] and ag["MIXTURE"]: dirs[k][p] = np.nanmean(ag["EARLY"]) - np.nanmean(ag["MIXTURE"])
    t = {}; t["HB_c"] = dict(primary=tci(list(hbc.values())), primary_upper_one_sided=tci(list(hbc.values()), 0.90), secondary_same_D_2x2=tci(list(hbc2.values())),
                             history_confounded_parents=[p for p, v in hist_conf.items() if v])
    pc = t["HB_c"]["primary"]; t["HB_c"]["FUNCTIONAL_DIFF"] = bool(pc["ci"] and pc["ci"][0] > th["delta_dis"])
    up = t["HB_c"]["primary_upper_one_sided"]; t["HB_c"]["EQUIVALENT"] = bool(up["ci"] and up["ci"][1] < th["delta_dis"])
    dd = {k: tci(list(v.values())) for k, v in dirs.items()}; hp = holm({k: (v.get("t_p") if v["n"] >= 2 else None) for k, v in dd.items()}, th["alpha"])
    d90 = {k: tci(list(v.values()), 0.90) for k, v in dirs.items()}
    nat_ind = {}
    for k in DIR:
        nat, ind = [], []
        for p in both:
            get = lambda base, arm: [l for u, l in byp[p] if u["base"] == base and u["arm"] == arm]
            eB, mB, mI, eI = ([dstat(l, k) for l in get(b_, a_)] for b_, a_ in (("E", "fullB"), ("M", "fullB"), ("M", "fullI"), ("E", "fullI")))
            if eB and mB: nat.append(np.nanmean(eB) - np.nanmean(mB))
            if mI and eI: ind.append(np.nanmean(mI) - np.nanmean(eI))
        nat_ind[k] = dict(native=tci(nat), induced=tci(ind), same_sign=(bool(np.sign(np.mean(nat)) == np.sign(np.mean(ind))) if nat and ind else None))
    t["HB_c_dir"] = dict(per_rule=dd, holm_reject=hp, ci90=d90, native_vs_induced=nat_ind, rules=list(DIR),
                         DIRECTED_DIFF=any(hp[k] and abs(dd[k]["mean"] or 0) > th["delta_dir"] and nat_ind[k]["same_sign"] is True for k in DIR),
                         EQUIVALENT=all(d90[k]["ci"] and -th["delta_dir"] < d90[k]["ci"][0] and d90[k]["ci"][1] < th["delta_dir"] for k in DIR))
    def pdiff(fn):
        out = {}
        for p in both:
            e = [fn(l) for _, l in byp[p] if l["cls"] == "EARLY"]; m = [fn(l) for _, l in byp[p] if l["cls"] == "MIXTURE"]
            out[p] = float(np.mean(e) - np.mean(m))
        return out
    hd = pdiff(lambda l: l["oqi2"]); hdt = tci(list(hd.values())); sgn = [np.sign(v) for v in hd.values()]
    t["HB_d"] = dict(stat=hdt, LOCUS_DIFF=bool(hdt["ci"] and abs(hdt["mean"]) > th["delta_oqi"] and (hdt["ci"][0] > 0 or hdt["ci"][1] < 0) and len(sgn) - max(sgn.count(1.0), sgn.count(-1.0)) <= 1),
                     profiles={c: {f"{kind}_{l_}": float(np.mean([l["ic"][f"{kind}_{l_}"]["IIA_norm"] for _, l in ev.values() if l["cls"] == c])) for kind in ("OQI", "OQI_unmatched", "RS") for l_ in range(4)}
                               for c in ("EARLY", "MIXTURE") if any(l["cls"] == c for _, l in ev.values())},
                     null_base_patch_max_logit_diff=max([l["ic"]["null_base_patch_max_logit_diff"] for _, l in ev.values()] or [None]), note="descriptive locus profile; not a test of what is computed")
    he = tci(list(pdiff(lambda l: l["KL"]).values())); t["HB_e"] = dict(stat=he, PRECISION_DIFF=bool(he["ci"] and abs(he["mean"]) > th["delta_kl"] and (he["ci"][0] > 0 or he["ci"][1] < 0)))
    t["HB_a_descriptive"] = {k: tci(list(pdiff(lambda l, k=k: l["w"][k]).values())) for k in RULES}
    t["HB_b_descriptive"] = dict(fraction_models_above_parametric_null=float(np.mean([any(l["w"][k] - l["w_par_null"][k] > 0.02 for k in RULES) for _, l in ev.values()])) if ev else None)
    cr = res["censoring_by_class"]; n_c = {c: sum(1 for _, (u, l) in L.items() if ((u["base"] == "E") == (u["arm"] == "fullB")) == (c == "EARLY")) for c in ("EARLY", "MIXTURE")}
    cens_gap = abs(cr["EARLY"] / max(n_c["EARLY"], 1) - cr["MIXTURE"] / max(n_c["MIXTURE"], 1))
    fd = t["HB_c"]["FUNCTIONAL_DIFF"]; eq = t["HB_c"]["EQUIVALENT"] and t["HB_c_dir"]["EQUIVALENT"]
    if len(both) < th["min_parents_both"]: fs = "INSUFFICIENT_UNITS"
    elif lev == "matched" and cens_gap > 0.20: fs = "NON_IDENTIFIABLE (class-dependent censoring)"
    elif fd and hbc and all(hist_conf.values()): fs = "FUNCTIONAL_DIFFERENCE (class and intervention history not separable)"
    elif fd and t["HB_c_dir"]["DIRECTED_DIFF"]: fs = "B1-FUNCTIONAL"
    elif fd: fs = "FUNCTIONAL_DIFFERENCE_RULE_UNIDENTIFIED"
    elif eq: fs = "EQUIVALENT"
    else: fs = "NON_IDENTIFIABLE (underpowered)"
    c_ = fs
    res.update(tests=t, censoring_gap=cens_gap, functional_status=fs, n_parents_both=len(both))
    return res


def score(preset, root=None, frozen_code="__frozen__", inputs=None):
    root = root or E.RUN_ROOT; C = PR.effective(preset); th = C["TRACK_B"]["thresholds"]
    if frozen_code == "__frozen__":
        fz = E.read_json(os.path.join(E.PKG, "protocols", "FREEZE.json")) or {}; frozen_code = fz.get("trackb_code_sha256")
        if frozen_code is None: return dict(verdict="PROVENANCE_FAIL", problems=["protocols/FREEZE.json has no trackb_code_sha256"])
    D, U, problems = inputs or load_units(preset, root, frozen_code)
    if problems: return dict(verdict="PROVENANCE_FAIL", problems=problems)
    tech = C["TECHNICAL_ONLY"]; out = dict(preset=preset, technical_only=tech, units={u["key"]: u["status"] for u in U}, levels={lev: score_level(U, lev, root, th, tech) for lev in ("matched", "final")})
    m, f = out["levels"]["matched"], out["levels"]["final"]                              # review A r2 N-B1: function at matched precision, precision at the final step
    if m["functional_status"] != f["functional_status"]: head = f"NON_IDENTIFIABLE (functional status differs: matched '{m['functional_status']}', final '{f['functional_status']}')"
    elif m["functional_status"] == "EQUIVALENT":
        loc = m["tests"]["HB_d"]["LOCUS_DIFF"]; prec = f["tests"]["HB_e"]["PRECISION_DIFF"]
        head = ("B2-LOCUS" + ("+B3" if prec else "")) if loc else ("B3" if prec else "EQUIVALENT_NO_DIFFERENCE_DETECTED")
    else: head = m["functional_status"]
    if m.get("n_parents_both", 0) < th["indicative_below_parents"] and not head.startswith("INSUFFICIENT"): head += " [indicative: < 6 parents]"
    for lev in ("matched", "final"): out["levels"][lev]["classification"] = "TECHNICAL_ONLY" if tech else out["levels"][lev]["functional_status"]
    out["verdict"] = "TECHNICAL_ONLY" if tech else head; out["verdict_if_not_technical"] = head; out["frozen_prediction"] = C["TRACK_B"]["frozen_prediction"]; return out


def _np(o):
    if isinstance(o, dict): return {str(k): _np(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_np(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
    return o


if __name__ == "__main__":
    preset, dst = sys.argv[1], sys.argv[2]; root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else None
    r = _np(score(preset, root)); E.atomic_json(r, dst); print(json.dumps(dict(verdict=r.get("verdict"), problems=r.get("problems")), indent=1)); sys.exit(2 if r.get("verdict") == "PROVENANCE_FAIL" else 0)
