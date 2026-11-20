"""Track A frozen scorer (v2 after review A B1, M1, M3, M5-M7, M13, m1-m7 and review C M7).
python -m cor_research.scoring.track_a PRESET OUT.json [--root R] [--step S]
Fail-closed: any provenance failure (measurement-code hash != frozen, checkpoint hash != the registry's hash for that run, measurement job
not registered done or its output hash differs, parent checkpoint hash differs, wrong stream) -> PROVENANCE_FAIL (exit 2), no verdicts.
Every expected (pair, arm) appears with a status: MISSING / NONCOMPETENT / ANCHORS_NOT_EVALUABLE / CONTROL_INVALID / INCOMPATIBLE /
SUPPORTED_PAIR / CONTRA_PAIR / MIDDLE.  Labels are recomputed from the recorded numbers.  Evaluability depends only on the anchors (and the
frozen validity rules), never on a hypothesis band (XVIII lesson; planted test 'band_change_keeps_evaluability').
Rules: cor_research/protocol.py (BAND, DELTA, EPS, PARENT_RULE, CROSS_RULE, VALIDITY); operating characteristics: tests/test_verdict_oc.py."""
import os, sys, json, math, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E, protocol as PR
from cor_research.analysis import hier as HS
P = PR.full()
OPP = {"EARLY": "MIXTURE", "MIXTURE": "EARLY"}; NATIVE = {"E": "EARLY", "M": "MIXTURE"}
LIVE = ("SUPPORTED_PAIR", "CONTRA_PAIR", "MIDDLE")


def cls(r, P=P):
    c = P["CLASS"]
    if r["nec_E"] >= c["E_hi"] and r["nec_M"] <= c["E_lo_M"]: return "EARLY"
    if r["nec_M"] >= c["M_hi"] and r["nec_E"] <= c["M_lo_E"]: return "MIXTURE"
    return "HYBRID"


def comp(r, P=P): return bool(r["KL"] <= P["COMP"]["KL"] and r["acc"] >= P["COMP"]["acc"])


def phi_parts(a, b, i, eps=0.01):
    l = lambda x: math.log(max(x, 0.0) + eps); v = {}
    for r in ("E", "M"):
        den = l(i[f"nec_{r}"]) - l(b[f"nec_{r}"]); v[r] = float("nan") if abs(den) < 1e-9 else (l(a[f"nec_{r}"]) - l(b[f"nec_{r}"])) / den
    return v


def phi(a, b, i, eps=0.01): v = phi_parts(a, b, i, eps); return 0.5 * (v["E"] + v["M"])


def load_inputs(preset, root, frozen_measure_code):
    from cor_research.jobs import plan as PL, runner as RN
    C = PR.effective(preset); D = PL.design(preset, root); R = RN.load_all(root); problems = []; parent_sha = {}
    def get(jid, track, tag, step, stream, train_jid=None):
        rel = PL.mpath(track, tag, step, stream); f = os.path.join(root, rel)
        if not os.path.exists(f): return None
        r = json.load(open(f)); mj = R.get(jid)
        if mj is None or mj.get("status") != "done" or mj.get("output_hashes", {}).get(rel) != E.sha256(f): problems.append(f"{rel}: measurement job {jid} not registered done with this output hash"); return r
        if r.get("stream") != stream: problems.append(f"{rel}: stream {r.get('stream')} != frozen {stream}")
        if frozen_measure_code is not None and r.get("code_sha256") != frozen_measure_code: problems.append(f"{rel}: measurement code hash differs from the frozen hash")
        p = r["parent"]
        if p not in parent_sha: parent_sha[p] = E.sha256(E.patch_parents().parent_path(p))
        if r.get("parent_sha256") != parent_sha[p]: problems.append(f"{rel}: parent checkpoint hash differs")
        if train_jid:
            tj = R.get(train_jid, {}); ck = f"ft/{tag}/step{step}.npz"; h = dict(tj.get("output_hashes", {})); h.update({f"ft/{tag}/{k}": v for k, v in tj.get("snapshot_hashes", {}).items()})
            if tj.get("status") != "done": problems.append(f"{rel}: training job {train_jid} not done")
            elif h.get(ck) != r["ckpt_sha256"]: problems.append(f"{rel}: checkpoint hash differs from the registry hash")
        return r
    SP, SW = C["STREAMS"]["measure_primary"], C["STREAMS"]["measure_window"]; ST = C["STEPS"]; M, W, TP = {}, {}, {}
    for key, tag in D["tags"].items():
        for step in ST["early_checks"] + [ST["horizon_hist"], ST["primary"]]: M[(key, step)] = get(f"meas:{key}@{step}", "trackA", tag, step, SP, f"cont:{key}")
    for key, tag in D["windows"].items(): W[key] = get(f"mwin:{key}", "window", tag, 100, SW, f"win:{key}")
    for key, tag in D["tps"].items(): TP[key] = get(f"mtp:{key}", "composed", tag, 100, SW, None)
    return D, R, dict(M=M, W=W, TP=TP), problems


def pair_outcome(h, a_rec, B, I, base, cfg, eps, flags=()):
    bandI, bandB, dl = cfg["BAND"]["I_like"], cfg["BAND"]["B_like"], cfg["DELTA"]
    if B is None or I is None: return "MISSING", None
    if not (comp(B, cfg) and cls(B, cfg) == NATIVE[base]): return "ANCHORS_NOT_EVALUABLE", None
    if h["expect"] == "switch":
        if not comp(I, cfg): return "NONCOMPETENT", None
        c = cls(I, cfg); return ("SUPPORTED_PAIR" if c == OPP[NATIVE[base]] else "CONTRA_PAIR" if c == NATIVE[base] else "MIDDLE"), c
    if not (comp(I, cfg) and cls(I, cfg) == OPP[NATIVE[base]]): return "ANCHORS_NOT_EVALUABLE", None
    if "CONTROL_INVALID" in flags: return "CONTROL_INVALID", None
    if a_rec is None: return "MISSING", None
    if not comp(a_rec, cfg): return "NONCOMPETENT", None
    v = phi(a_rec, B, I, eps); c = cls(a_rec, cfg); cI, cB = cls(I, cfg), cls(B, cfg)
    if not math.isfinite(v): return "ANCHORS_NOT_EVALUABLE", None
    if h["expect"] == "I_like":
        st = "SUPPORTED_PAIR" if (v >= bandI[0] and c == cI) else "CONTRA_PAIR" if (v < bandI[0] - dl or c == cB) else "MIDDLE"
    elif h["expect"] == "B_like":
        st = "SUPPORTED_PAIR" if (v <= bandB[1] and c == cB) else "CONTRA_PAIR" if (v > bandB[1] + dl or c == cI) else "MIDDLE"
        pass
    elif h["expect"] == "partial":
        lo, hi = cfg["BAND"]["partial"]; st = "SUPPORTED_PAIR" if (lo <= v <= hi and c == "HYBRID") else "CONTRA_PAIR" if (c in (cB, cI) and (v < lo - dl or v > hi + dl)) else "MIDDLE"
    elif h["expect"] == "not_I_like":
        st = "SUPPORTED_PAIR" if (v < bandI[0] or c != cI) else "CONTRA_PAIR" if (v >= bandI[0] + dl and c == cI) else "MIDDLE"
    else: raise ValueError(h["expect"])
    if st == "SUPPORTED_PAIR" and "INCOMPATIBLE" in flags: st = "INCOMPATIBLE"                  # a broken composed state is not evidence (any transplant)
    return st, v


def parent_verdict(outs, h, cfg):
    ev = [o for o in outs if o in LIVE]; nc = outs.count("NONCOMPETENT"); pre_ev = [o for o in outs if o not in ("MISSING", "ANCHORS_NOT_EVALUABLE", "NOT_IN_PRESET", "CONDITION_NOT_MET")]
    if pre_ev and nc > len(pre_ev) / 2: return "PARENT_NONCOMPETENT"
    if len(ev) < cfg["MIN_PAIRS_PER_PARENT"]: return "PARENT_INCONCLUSIVE"
    s, c = ev.count("SUPPORTED_PAIR"), ev.count("CONTRA_PAIR"); R = cfg["PARENT_RULE"]
    if h["expect"] == "persist": return "PARENT_SUPPORTS" if s >= 0.8 * len(ev) else "PARENT_CONTRADICTS" if c > len(ev) * R["contra_frac"] else "PARENT_PARTIAL"
    if s >= math.ceil(R["support_frac"] * len(ev)) and c <= max(R["max_contra"], math.floor(R["max_contra_frac"] * len(ev))): return "PARENT_SUPPORTS"
    if c >= math.ceil(len(ev) * R["contra_frac"]) and c > s: return "PARENT_CONTRADICTS"
    return "PARENT_PARTIAL"


def cross_parent(pv, cfg):
    ev = {p: v for p, v in pv.items() if v in ("PARENT_SUPPORTS", "PARENT_CONTRADICTS", "PARENT_PARTIAL")}; n = len(ev); R = cfg["CROSS_RULE"]
    if n < cfg["MIN_PARENTS"]: return "INCONCLUSIVE"
    s = sum(v == "PARENT_SUPPORTS" for v in ev.values()); c = sum(v == "PARENT_CONTRADICTS" for v in ev.values())
    if s >= max(n - R["gen_exceptions"], math.ceil(R["gen_frac"] * n)) and c == 0: return "GENERALIZES"
    if c >= max(R["fail_min"], math.ceil(R["fail_frac"] * n)): return "FAILS_TO_GENERALIZE"
    return "NOT_ESTABLISHED"


def sign_flip_p(means):
    v = [m for m in means if m is not None and math.isfinite(m)]
    if len(v) < 2: return None
    obs = abs(sum(v)); cnt = 0; tot = 0
    for s in itertools.product([1, -1], repeat=len(v)): tot += 1; cnt += abs(sum(a * b for a, b in zip(s, v))) >= obs - 1e-12
    return cnt / tot


def flags_for(key, arm, base, cfg, I_, step_meas, thr_=None):
    """CONTROL_INVALID (lbl strength) and INCOMPATIBLE (composed-state shock) flags of one pair-arm"""
    f = set(); W = I_["W"]; unit = key.rsplit(":", 1)[0]
    if arm == "lbl":
        b, i, l = W.get(f"{unit}:B"), W.get(f"{unit}:I"), W.get(f"{unit}:lbl")
        if None in (b, i, l): f.add("CONTROL_INVALID")
        elif (l["KL"] - b["KL"]) < cfg["VALIDITY"]["lbl_excess_ratio_min"] * (i["KL"] - b["KL"]): f.add("CONTROL_INVALID")
    ad = cfg["ARMS"].get(arm) or cfg["HIGH_EXTRA_ARMS"].get(arm, {}); thr = cfg["VALIDITY"]["incompatible_kl_excess"] if thr_ is None else thr_
    if ad.get("kind") in ("transplant", "perturb") and arm not in ("fullB", "fullI"):     # review A r2 N5: every composed state, incl. step 100
        tp, wins = I_["TP"].get(key), [W.get(f"{unit}:{x}") for x in ad.get("src", ())]
        if tp is not None and all(w_ is not None for w_ in wins) and tp["KL"] - max(w_["KL"] for w_ in wins) > thr: f.add("INCOMPATIBLE")
        for st in cfg["STEPS"]["early_checks"]:
            a, bb, ii = step_meas(key, st), step_meas(unit + ":fullB", st), step_meas(unit + ":fullI", st)
            if None in (a, bb, ii): continue
            if a["KL"] - max(bb["KL"], ii["KL"]) > thr: f.add("INCOMPATIBLE")
    return f


def score(preset, root=None, step=None, cfg=None, frozen_measure_code="__frozen__", inputs=None, eps=None, parents=None):
    root = root or E.RUN_ROOT; cfg = cfg or PR.effective(preset); step = step or cfg["STEPS"]["primary"]; eps = eps if eps is not None else cfg["EPS"]
    if frozen_measure_code == "__frozen__":
        fz = E.read_json(os.path.join(E.PKG, "protocols", "FREEZE.json")) or {}; frozen_measure_code = fz.get("measure_code_sha256")
        if frozen_measure_code is None: return dict(verdict="PROVENANCE_FAIL", problems=["protocols/FREEZE.json has no measure_code_sha256"])
    D, R, I_, problems = inputs or load_inputs(preset, root, frozen_measure_code)
    if problems: return dict(verdict="PROVENANCE_FAIL", problems=problems)
    M = I_["M"]; rec = lambda key, st: M.get((key, st)); units = [u for u in D["units"] if parents is None or u["parent"] in parents]
    out = dict(preset=preset, step=step, eps=eps, technical_only=cfg.get("TECHNICAL_ONLY", preset in ("smoke", "pilot")), units=units, eligibility=D.get("eligibility"),
               parents_subset=parents, pairs={}, hypotheses={}, stats={}, labels="mediation verdicts are CONDITIONAL on the switch in the same pair (review A M1)")
    for h in cfg["HYPOTHESES_A"]:
        for base in h["bases"]:
            bunits = [u for u in units if u["base"] == base]; per_parent = {}; vals = {}; valsE = {}; valsM = {}
            for u in bunits:
                p = u["parent"]
                for Dd in D["D"]:
                    k = lambda a: f"{p}:{base}{Dd}:{a}"; B, I = rec(k("fullB"), step), rec(k("fullI"), step)
                    if h["expect"] == "persist":
                        B1, I1 = rec(k("fullB"), cfg["STEPS"]["horizon_hist"]), rec(k("fullI"), cfg["STEPS"]["horizon_hist"])
                        if None in (B, I, B1, I1): st, v = "MISSING", None
                        elif not (comp(B1, cfg) and cls(B1, cfg) == NATIVE[base] and comp(I1, cfg) and cls(I1, cfg) == OPP[NATIVE[base]]): st, v = "ANCHORS_NOT_EVALUABLE", None
                        elif not (comp(B, cfg) and comp(I, cfg)): st, v = "NONCOMPETENT", None
                        else:
                            same = [cls(B, cfg) == cls(B1, cfg), cls(I, cfg) == cls(I1, cfg)]; flip = [cls(B, cfg) == OPP[cls(B1, cfg)], cls(I, cfg) == OPP[cls(I1, cfg)]]
                            st, v = ("SUPPORTED_PAIR" if all(same) else "CONTRA_PAIR" if any(flip) else "MIDDLE"), [cls(B, cfg), cls(I, cfg)]
                    elif h["arm"] not in D["arms_by_base"].get(base, D["arms"]): st, v = "NOT_IN_PRESET", None
                    else:
                        fl = flags_for(k(h["arm"]), h["arm"], base, cfg, I_, rec); st, v = pair_outcome(h, rec(k(h["arm"]), step), B, I, base, cfg, eps, fl)
                        if h.get("condition") == "vsgo_I_like" and st in LIVE + ("NONCOMPETENT", "INCOMPATIBLE"):   # review A r2 N2: condition on the vsgo effect
                            vs, _ = pair_outcome(dict(expect="I_like"), rec(k("vsgo"), step), B, I, base, cfg, eps)
                            if vs != "SUPPORTED_PAIR": st, v = "CONDITION_NOT_MET", None
                        a_ = rec(k(h["arm"]), step)
                        if isinstance(v, float) and a_ is not None:
                            pp = phi_parts(a_, B, I, eps); valsE.setdefault(p, []).append(pp["E"]); valsM.setdefault(p, []).append(pp["M"])
                    per_parent.setdefault(p, []).append(st); out["pairs"].setdefault(h["id"] + ":" + base, []).append(dict(parent=p, D=Dd, status=st, value=v))
                    if isinstance(v, float): vals.setdefault(p, []).append(v)
            pv = {p: parent_verdict(o, h, cfg) for p, o in per_parent.items()}; verdict = cross_parent(pv, cfg) if bunits else "INCONCLUSIVE"
            counts = {s_: sum(o.count(s_) for o in per_parent.values()) for s_ in ("MISSING", "NONCOMPETENT", "ANCHORS_NOT_EVALUABLE", "CONTROL_INVALID", "INCOMPATIBLE", "CONDITION_NOT_MET", "NOT_IN_PRESET") + LIVE}
            hm = HS.hier_mean(vals) if vals else None
            out["hypotheses"][h["id"] + ":" + base] = dict(text=h["text"], arm=h["arm"], expect=h["expect"], role=h.get("role"), parent_verdicts=pv, verdict=verdict,
                n_parents_evaluable=sum(v_ in ("PARENT_SUPPORTS", "PARENT_CONTRADICTS", "PARENT_PARTIAL") for v_ in pv.values()), pair_status_counts=counts,
                denominator=dict(preregistered=len(D.get("parents", [])), units_this_base=len(bunits), parents_with_pairs=len(per_parent),
                                 parents_with_3_evaluable=sum(1 for o in per_parent.values() if sum(x in LIVE for x in o) >= 3)),
                phi=hm, phi_E=HS.hier_mean(valsE) if valsE else None, phi_M=HS.hier_mean(valsM) if valsM else None, variance_components_between_unit=HS.var_comp(vals) if vals else None,
                sign_flip_p_parent_means=sign_flip_p(list((hm or {}).get("parent_means", {}).values())) if hm else None)
    for k, h in out["hypotheses"].items():                                            # conditional label (review A M1)
        if h["role"] in ("mediation", "discriminating", "replication", "control", "replication_anchor", "low_confidence", "weak_negative") and h["verdict"] == "GENERALIZES":
            if out["hypotheses"].get("HA1:" + k.split(":")[1], {}).get("verdict") != "GENERALIZES": h["verdict"] = "GENERALIZES_CONDITIONAL"
        if out["technical_only"]: h["verdict_if_not_technical"] = h["verdict"]; h["verdict"] = "TECHNICAL_ONLY"
    for base in "EM":
        sw, cp, unc = {}, {}, {}
        for u in [u for u in units if u["base"] == base]:
            for Dd in D["D"]:
                B, I = rec(f"{u['parent']}:{base}{Dd}:fullB", step), rec(f"{u['parent']}:{base}{Dd}:fullI", step)
                native = B is not None and I is not None and comp(B, cfg) and cls(B, cfg) == NATIVE[base]
                if native: sw.setdefault(u["parent"], []).append(int(comp(I, cfg) and cls(I, cfg) == OPP[NATIVE[base]]))
                for a in D["arms_by_base"].get(base, D["arms"]):
                    r = rec(f"{u['parent']}:{base}{Dd}:{a}", step)
                    if r is not None:
                        cp.setdefault(a, {}).setdefault(u["parent"], []).append(int(comp(r, cfg)))
                        if native and comp(r, cfg): unc.setdefault(a, []).append(dict(parent=u["parent"], D=Dd, cls=cls(r, cfg), fullI_cls=cls(I, cfg), concordant=cls(r, cfg) == cls(I, cfg)))
        out["stats"][base] = dict(switch_probability=HS.prop(sw) if sw else None, competence_rate={a: HS.prop(v) for a, v in cp.items()}, unconditional_sensitivity_XA5=unc)
    return out


def score_all(preset, root=None, step=None, frozen_measure_code="__frozen__"):
    """primary score + fresh-parents-only verdicts + EPS sensitivity + the HA8E co-primary step (one input load, one provenance check)"""
    root = root or E.RUN_ROOT; cfg = PR.effective(preset)
    if frozen_measure_code == "__frozen__":
        fz = E.read_json(os.path.join(E.PKG, "protocols", "FREEZE.json")) or {}; frozen_measure_code = fz.get("measure_code_sha256")
        if frozen_measure_code is None: return dict(verdict="PROVENANCE_FAIL", problems=["protocols/FREEZE.json has no measure_code_sha256"])
    inp = load_inputs(preset, root, frozen_measure_code)
    if inp[3]: return dict(verdict="PROVENANCE_FAIL", problems=inp[3])
    main = score(preset, root, step, cfg, None, inp)
    fresh = [u["parent"] for u in inp[0]["units"] if u["parent"] not in cfg["PREVIOUSLY_OBSERVED"]]
    fr = score(preset, root, step, cfg, None, inp, parents=set(fresh)); main["fresh_only"] = {k: dict(verdict=h["verdict"], n_parents_evaluable=h["n_parents_evaluable"]) for k, h in fr["hypotheses"].items()}
    for k, h in main["hypotheses"].items():
        if h["verdict"] in ("GENERALIZES", "GENERALIZES_CONDITIONAL"):
            fv = main["fresh_only"][k]["verdict"]
            if fv == "FAILS_TO_GENERALIZE": h["verdict"] = "NOT_ESTABLISHED"; h["note"] = "fresh-parent-only verdict FAILS (review A M3)"
            elif main["fresh_only"][k]["n_parents_evaluable"] < cfg["MIN_PARENTS"]: h["note"] = "supported only with previously observed parents included (< 4 fresh parents evaluable)"
    inc = {}
    for thr in cfg["VALIDITY"]["incompatible_sensitivity"]:
        _ff = flags_for; globals()["flags_for"] = lambda *a, **k: _ff(*a, thr_=thr)
        try: inc[str(thr)] = {k: h["verdict"] for k, h in score(preset, root, step, cfg, None, inp)["hypotheses"].items()}
        finally: globals()["flags_for"] = _ff
    main["incompatible_threshold_sensitivity"] = inc
    main["eps_sensitivity"] = {str(e): {k: h["verdict"] for k, h in score(preset, root, step, cfg, None, inp, eps=e)["hypotheses"].items()} for e in cfg["EPS_SENS"]}
    h8 = next(h for h in cfg["HYPOTHESES_A"] if h["id"] == "HA8E"); s2 = score(preset, root, h8["co_primary_step"] if not cfg["TECHNICAL_ONLY"] else cfg["STEPS"]["horizon_hist"], cfg, None, inp)
    main["co_primary"] = {"HA8E:E@" + str(s2["step"]): s2["hypotheses"].get("HA8E:E", {}).get("verdict")}
    return main


if __name__ == "__main__":
    preset, dst = sys.argv[1], sys.argv[2]; root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else None
    step = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else None
    r = score_all(preset, root, step); E.atomic_json(r, dst); print(json.dumps({k: v["verdict"] for k, v in r.get("hypotheses", {}).items()} or r, indent=1)[:3000])
    sys.exit(2 if r.get("verdict") == "PROVENANCE_FAIL" else 0)
