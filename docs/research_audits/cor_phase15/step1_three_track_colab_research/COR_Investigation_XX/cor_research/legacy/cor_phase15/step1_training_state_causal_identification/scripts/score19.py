"""XIX frozen scorer (v2, after pre-experiment reviews A/B/C).  Usage: score19.py DESIGN.json MEASURE.jsonl OUT.json [--sens NAME] [--no-files]
Fail-closed: any provenance failure -> verdict PROVENANCE_FAIL for the whole score, exit code 2, no hypothesis bands.

DESIGN keys: stream, parent, score_code_sha256, refs_sha256, dep_sha256 {file: sha}, dep_sha256_measure, harness_sha256, min_pairs,
pairs [{name, base, L, W, D, Isuf, win {arm: suffix}, wenv {arm: {FX_*: value}}}], families {FAM: {anchors: {B, I}}},
arms {ARM: {family, run, transplant | branch_from}}, matching [{arm, kind, lo, hi}] (optional), hypotheses [...].
  transplant = {tp, params_from_I, opt, perturb, alpha, src_dirs {B, I}}; branch_from = training-arm key of pair.win / pair.wenv.
  arms may carry bases "E" | "M" | "EM" (default); a perturbation arm's seed for a pair is perturb + D (+ 500 on base M).

Per pair x arm (labels are RECOMPUTED here from the recorded numbers; record labels are ignored):
  competent  KL <= COMP_KL and acc >= COMP_ACC           (primary 0.06 / 0.80; variants comp_strict 0.045 / 0.80, comp_loose 0.08 / 0.78)
  causal     EARLY if nec_E >= 0.30 and nec_M <= 0.10; MIXTURE if nec_M >= 0.30 and nec_E <= 0.15; else HYBRID (threshold variants below)
  functional residual-profile concordance (RPC) class of f: EARLY f >= 0.05, MIXTURE f <= -0.05, else AMBIGUOUS (NOT a test of Claim 2)
  phi        log-necessity phi against the pair's family anchors (EPS); variant 'lin' uses linear necessity
  phif       (f_arm - f_Banchor) / (f_Ianchor - f_Banchor)
  category   NONCOMPETENT | HYBRID[RPC class] | BOTH | CAUSAL_ONLY | FUNCTIONAL_ONLY | NEITHER | INCONCLUSIVE(...)
Pair evaluability (per family): both anchors competent, causally pure, of opposite classes (variant evalB: only the B anchor).
Functional evaluability (category hypotheses): additionally each anchor's RPC class equals its causal class.
Hypotheses are per base only ("base": "E" | "M"; anything else fails closed).
  kind "band"     v = expr per evaluable pair over phi / phif / f / func; need_class "I" | "B" | null adds the class condition (every arm in
                  expr must have the I-anchor's / B-anchor's causal class).  A pair is dropped and listed if: not evaluable, an arm in expr
                  non-competent, v NaN, or the matched-control gate failed / is missing.  INCONCLUSIVE if < min_pairs remain; SUPPORTED if
                  every v in [lo, hi] with the class condition; CONTRADICTED if a strict majority have v outside [lo-DELTA, hi+DELTA] or fail
                  the class condition; else PARTIAL.
  kind "category" the arm's category per functionally evaluable pair whose category is in "eligible" (default support + contradict; P7 uses
                  only pairs with a causal switch): SUPPORTED if all in "support"; CONTRADICTED if a strict majority in "contradict";
                  INCONCLUSIVE if < min_pairs; else PARTIAL.  Any hypothesis: NONCOMPETENT if more than half of its base's pairs drop for a
                  non-competent arm.
Provenance (files mode): code, dependency and reference hashes; exactly one record per run with the frozen stream, parent, step, hashes;
checkpoint hash; run_config.json of every continuation (argv parent L W D, no intervention, FX_NOFALLBACK=1, branch source inside this
package and equal to the intended state, branch-source hashes, harness hash); transplant provenance (groups, opt, perturb seed, alpha,
composed and source hashes); native B / I step-100 states trained with the intended settings; fullB bit-identical to native B (parameters,
moments, rng) and fullI parameters to native I; training-arm window states trained with the intended window environment.  Duplicate
composed states within a pair are reported (one unit).
Frozen sensitivity variants (--sens): delta0, delta20, phicut70, phicut80, eps003, eps03, lin, evalB, thr_pm05, thr_strict, comp_strict, comp_loose."""
import os, sys, json, math, hashlib, argparse
H = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(H, "..")); sys.path.insert(0, os.path.join(H, "..", "src"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
CFG = dict(DELTA=0.10, PHI_HI=0.75, PHI_LO=0.25, EPS=0.01, EVAL="both", THR=0.0, THR_STRICT=False, COMP_KL=0.06, COMP_ACC=0.80, LIN=False)
SENS = dict(delta0=dict(DELTA=0.0), delta20=dict(DELTA=0.20), phicut70=dict(PHI_HI=0.70, PHI_LO=0.30), phicut80=dict(PHI_HI=0.80, PHI_LO=0.20),
            eps003=dict(EPS=0.003), eps03=dict(EPS=0.03), lin=dict(LIN=True), evalB=dict(EVAL="B"), thr_pm05=dict(THR=0.05), thr_strict=dict(THR_STRICT=True),
            comp_strict=dict(COMP_KL=0.045), comp_loose=dict(COMP_KL=0.08, COMP_ACC=0.78))

def phi(a, b, i, cfg):
    l = (lambda x: x) if cfg["LIN"] else (lambda x: math.log(max(x, 0.0) + cfg["EPS"])); v = []
    for r in ("E", "M"):
        den = l(i[f"nec_{r}"]) - l(b[f"nec_{r}"])
        if abs(den) < 1e-9: return float("nan")
        v.append((l(a[f"nec_{r}"]) - l(b[f"nec_{r}"])) / den)
    return 0.5 * sum(v)

def causal(n, cfg):
    shifts = [(0.05, 0.05)] if cfg["THR_STRICT"] else ([(cfg["THR"], cfg["THR"]), (-cfg["THR"], -cfg["THR"])] if cfg["THR"] else [(0, 0)])
    out = set()
    for up, dn in shifts:
        out.add("EARLY" if (n["nec_E"] >= 0.30 + up and n["nec_M"] <= 0.10 - dn) else ("MIXTURE" if (n["nec_M"] >= 0.30 + up and n["nec_E"] <= 0.15 - dn) else "HYBRID"))
    return out.pop() if len(out) == 1 else "HYBRID"
competent = lambda n, cfg: bool(n["KL"] <= cfg["COMP_KL"] and n["acc"] >= cfg["COMP_ACC"])
fclass = lambda f: "EARLY" if f >= 0.05 else ("MIXTURE" if f <= -0.05 else "AMBIGUOUS")
fmt = lambda tmpl, D, p: tmpl.format(parent=D["parent"], pair=p["name"], **{k: v for k, v in p.items() if k not in ("name", "win", "wenv")})
WKEYS = ("FX_INTERV", "FX_EDGES", "FX_B", "FX_FREEZE", "FX_PNZ_SEED", "FX_PNZ_SCALE", "FX_LBL_P", "FX_LBL_SEED")

def provenance(D, recs, ckroot, files=True):
    errs = []
    for k in ("stream", "parent", "score_code_sha256", "refs_sha256", "min_pairs", "pairs", "families", "arms", "hypotheses"):
        if k not in D: errs.append(f"design lacks {k}")
    if errs: return errs, {}, {}
    for h in D["hypotheses"]:
        if h.get("base") not in ("E", "M"): errs.append(f"hypothesis {h.get('id')}: base must be E or M (no pooling)")
        if h.get("kind", "band") not in ("band", "category"): errs.append(f"hypothesis {h.get('id')}: unknown kind")
        import re
        for an in set(re.findall(r"\['([A-Za-z0-9_+]+)'\]", h.get("expr", ""))) | ({h["arm"]} if h.get("arm") else set()):
            if an not in D["arms"] or h.get("base") not in D["arms"][an].get("bases", "EM"): errs.append(f"hypothesis {h.get('id')}: arm {an} not defined on base {h.get('base')}")
    if files:
        if "harness_sha256" not in D or "dep_sha256" not in D or "dep_sha256_measure" not in D: errs.append("design lacks harness/dependency hashes")
        if sha(os.path.join(PKG, "src", "xix_score.py")) != D["score_code_sha256"]: errs.append("src/xix_score.py differs from the frozen hash")
        for f, hsh in D.get("dep_sha256", {}).items():
            if not os.path.exists(os.path.join(PKG, f)) or sha(os.path.join(PKG, f)) != hsh: errs.append(f"dependency {f} differs from the frozen hash")
    if errs: return errs, {}, {}
    need = {(p["name"], an): fmt(A["run"], D, p) for p in D["pairs"] for an, A in D["arms"].items() if p["base"] in A.get("bases", "EM")}
    byrun = {}
    for r in recs: byrun.setdefault(r["run"], []).append(r)
    composed = {}
    for (pn, an), run in need.items():
        rs = byrun.get(run, []); p = [q for q in D["pairs"] if q["name"] == pn][0]; A = D["arms"][an]
        if len(rs) != 1: errs.append(f"{run}: {len(rs)} measurement records (need exactly 1)"); continue
        r = rs[0]
        for k, v in (("stream", D["stream"]), ("parent", D["parent"]), ("score_code_sha256", D["score_code_sha256"]), ("refs_sha256", D["refs_sha256"]), ("step", 1211)):
            if r.get(k) != v: errs.append(f"{run}: {k}={r.get(k)} != {v}")
        for k in ("nec_E", "nec_M", "KL", "acc", "f"):
            if not isinstance(r.get(k), (int, float)) or r[k] != r[k]: errs.append(f"{run}: missing or NaN {k}")
        if not files: continue
        if r.get("dep_sha256") != D["dep_sha256_measure"]: errs.append(f"{run}: measurement dependency hashes differ from the frozen set")
        rd = os.path.join(ckroot, run); ck = os.path.join(rd, "step1211.npz")
        if not os.path.exists(ck): errs.append(f"{run}: checkpoint missing"); continue
        if sha(ck) != r.get("ckpt_sha256"): errs.append(f"{run}: checkpoint hash mismatch")
        try: rc = json.load(open(os.path.join(rd, "run_config.json")))
        except Exception as e: errs.append(f"{run}: run_config unreadable ({e})"); continue
        if rc.get("argv", [None])[1:5] != [D["parent"], str(p["L"]), str(p["W"]), str(p["D"])]: errs.append(f"{run}: argv {rc.get('argv')} does not match the pair")
        if rc.get("env", {}).get("FX_NOFALLBACK") != "1": errs.append(f"{run}: FX_NOFALLBACK not set")
        if rc.get("env", {}).get("FX_INTERV"): errs.append(f"{run}: continuation has an intervention set")
        if rc.get("code", {}).get("xix_train.py") != D["harness_sha256"]: errs.append(f"{run}: harness hash differs")
        bd = os.path.abspath(rc.get("branch_dir") or "/")
        if not bd.startswith(PKG): errs.append(f"{run}: branch source outside this package ({bd})")
        if A.get("transplant"):
            T = A["transplant"]; tp = fmt(T["tp"], D, p); tpd = os.path.abspath(os.path.join(ckroot, tp))
            if os.path.normpath(bd) != os.path.normpath(tpd): errs.append(f"{run}: branched from {bd}, not {tp}")
            try: pv = json.load(open(os.path.join(tpd, "provenance.json")))
            except Exception as e: errs.append(f"{run}: transplant provenance unreadable ({e})"); continue
            got_groups = pv.get("perturbed_groups") if T.get("perturb") is not None else pv.get("params_from_I")
            if sorted(got_groups or []) != sorted(T["params_from_I"]): errs.append(f"{run}: provenance groups {got_groups} != {T['params_from_I']}")
            pseed = None if T.get("perturb") is None else T["perturb"] + p["D"] + (500 if p["base"] == "M" else 0)   # per-pair seed rule (pre-review B4)
            for k, want in (("opt", T["opt"]), ("perturb_seed", pseed), ("alpha", T.get("alpha"))):
                if pv.get(k) != want: errs.append(f"{run}: provenance {k}={pv.get(k)} != {want}")
            for f in ("step100.npz", "opt_step100.pkl"):
                h_ = sha(os.path.join(tpd, f)) if os.path.exists(os.path.join(tpd, f)) else None
                if h_ is None or h_ != pv.get("outputs", {}).get(f): errs.append(f"{run}: composed {f} hash mismatch")
                if (rc.get("branch_src") or {}).get(f) != pv.get("outputs", {}).get(f): errs.append(f"{run}: the run started from a {f} other than the recorded composition")
            for lab in ("B", "I"):
                src = fmt(T["src_dirs"][lab], D, p)
                for f in ("step100.npz", "opt_step100.pkl"):
                    fp = os.path.join(ckroot, src, f)
                    if not os.path.exists(fp) or pv.get("inputs", {}).get(f"{lab}/{f}") != sha(fp): errs.append(f"{run}: composed from a {lab}/{f} that differs from {src}")
            composed.setdefault(pn, {})[an] = pv.get("outputs", {}).get("step100.npz")
        elif A.get("branch_from"):
            arm = A["branch_from"]; s100 = f"{D['parent']}_fx_L{p['L']}_W{p['W']}_D{p['D']}{p['win'][arm]}_stop100_s100c19"; sd = os.path.abspath(os.path.join(ckroot, s100))
            if os.path.normpath(bd) != os.path.normpath(sd): errs.append(f"{run}: branched from {bd}, not {s100}")
            for f in ("step100.npz", "opt_step100.pkl"):
                if not os.path.exists(os.path.join(sd, f)) or (rc.get("branch_src") or {}).get(f) != sha(os.path.join(sd, f)): errs.append(f"{run}: step-100 source {f} differs from the hash recorded at run start")
            try: wc = json.load(open(os.path.join(sd, "run_config.json")))
            except Exception as e: errs.append(f"{run}: window-state run_config unreadable ({e})"); continue
            want = p["wenv"][arm]; got = {k: v for k, v in wc.get("env", {}).items() if k in WKEYS}
            if got != want: errs.append(f"{run}: window state was trained with {got}, design says {want}")
            if want.get("FX_INTERV", "").startswith("pnz"):
                pre_ = f"checkpoints/ft/{D['parent']}_fx_L{p['L']}_W{p['W']}_D{p['D']}"; ref = f"{pre_}_stop100_s100c19:{pre_}{p['Isuf']}_stop100_s100c19"
                if wc.get("env", {}).get("FX_PNZ_REF") != ref: errs.append(f"{run}: noise scales taken from {wc.get('env', {}).get('FX_PNZ_REF')}, not the pair's own B/I states")
            if wc.get("argv", [None])[1:5] != [D["parent"], str(p["L"]), str(p["W"]), str(p["D"])]: errs.append(f"{run}: window-state argv does not match the pair")
            if wc.get("code", {}).get("xix_train.py") != D["harness_sha256"]: errs.append(f"{run}: window-state harness hash differs")
    if files:
        import numpy as np, pickle
        for p in D["pairs"]:
            pre = f"{D['parent']}_fx_L{p['L']}_W{p['W']}_D{p['D']}"
            for nmd, want in ((pre + "_stop100_s100c19", {}), (pre + p["Isuf"] + "_stop100_s100c19", p["wenv"]["I"])):
                try: wc = json.load(open(os.path.join(ckroot, nmd, "run_config.json")))
                except Exception as e: errs.append(f"{nmd}: run_config unreadable ({e})"); continue
                got = {k: v for k, v in wc.get("env", {}).items() if k in WKEYS}
                if got != want: errs.append(f"{nmd}: trained with {got}, design says {want}")
                if wc.get("code", {}).get("xix_train.py") != D["harness_sha256"]: errs.append(f"{nmd}: harness hash differs")
            try:
                zB, zI = np.load(os.path.join(ckroot, pre + "_stop100_s100c19", "step100.npz")), np.load(os.path.join(ckroot, pre + p["Isuf"] + "_stop100_s100c19", "step100.npz"))
                tB, tI = np.load(os.path.join(ckroot, f"tp_{p['name']}_fullB", "step100.npz")), np.load(os.path.join(ckroot, f"tp_{p['name']}_fullI", "step100.npz"))
                if not all(np.array_equal(zB[k], tB[k]) for k in zB.files): errs.append(f"{p['name']}: composed fullB is not bit-identical to the native B state")
                if not all(np.array_equal(zI[k], tI[k]) for k in zI.files): errs.append(f"{p['name']}: composed fullI parameters are not bit-identical to the native I state")
                oB = pickle.load(open(os.path.join(ckroot, pre + "_stop100_s100c19", "opt_step100.pkl"), "rb")); ofB = pickle.load(open(os.path.join(ckroot, f"tp_{p['name']}_fullB", "opt_step100.pkl"), "rb"))
                if not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(oB["opt"], ofB["opt"])) or oB["rng"] != ofB["rng"]: errs.append(f"{p['name']}: composed fullB optimiser/rng state differs from native B")
            except Exception as e: errs.append(f"{p['name']}: composition identity check failed to run ({e})")
    dup = {pn: [sorted(a for a, h_ in c.items() if h_ == h) for h in set(c.values()) if list(c.values()).count(h) > 1] for pn, c in composed.items()}
    return errs, {k: byrun[v][0] for k, v in need.items() if len(byrun.get(v, [])) == 1}, {k: v for k, v in dup.items() if v}

def matching(D, ckroot):
    """Predeclared matching gates for the non-targeted controls (step-100 states only; fail closed on any missing state).  D["matching"] is a
    list of {arm, kind, lo, hi}:  kind "dist": total parameter distance of the control's window state from B relative to I's, in [lo, hi];
    kind "kl100": step-100 loss disturbance (KL(Bayes||model) of the control state minus B's, relative to I's minus B's, on the frozen
    stream's natural set), in [lo, hi].  Result: {arm: {pair: {ratio, ok}}}."""
    import numpy as np
    out = {}
    for G in (D.get("matching") or []):
        o = out.setdefault(G["arm"], {})
        for p in D["pairs"]:
            try:
                pre = f"{D['parent']}_fx_L{p['L']}_W{p['W']}_D{p['D']}"; dB, dI, dC = pre + "_stop100_s100c19", pre + p["Isuf"] + "_stop100_s100c19", pre + p["win"][G["arm"]] + "_stop100_s100c19"
                if G["kind"] == "dist":
                    ld = lambda d: np.load(os.path.join(ckroot, d, "step100.npz")); zB, zI, zC = ld(dB), ld(dI), ld(dC)
                    dist = lambda z: float(np.sqrt(sum(((z[k] - zB[k]) ** 2).sum() for k in zB.files))); r = dist(zC) / dist(zI)
                elif G["kind"] == "kl100":
                    sys.path.insert(0, os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src")); import xviiim as XM, xix_score as XS
                    kl = {d: XS.necessity(XM.load_dir(os.path.join(ckroot, d), D["parent"], 100), D["stream"])["KL"] for d in (dB, dI, dC)}
                    eI, eC = kl[dI] - kl[dB], kl[dC] - kl[dB]; mn = G.get("min_block_excess", 0.005)
                    if eC <= 0: o[p["name"]] = dict(ratio=None, excess_block=eI, excess_ctl=eC, ok=False, why="control has no loss disturbance"); continue
                    if eI < mn: o[p["name"]] = dict(ratio=None, excess_block=eI, excess_ctl=eC, ok=bool(eC >= mn), na=False, why=f"block excess < {mn}: loss matching not applicable; control usable if its own excess >= {mn}"); continue
                    r = eC / eI; o[p["name"]] = dict(ratio=r, excess_block=eI, excess_ctl=eC, ok=bool(G["lo"] <= r <= G["hi"])); continue
                else: raise ValueError(G["kind"])
                o[p["name"]] = dict(ratio=r, ok=bool(G["lo"] <= r <= G["hi"]))
            except Exception as e: o[p["name"]] = dict(ratio=None, ok=False, error=str(e))
    return out

def score(D, M, cfg, match=None):
    out = dict(cfg=cfg, pairs={}, hypotheses={}, matching=match or {}); gated = {g["arm"] for g in (D.get("matching") or [])}
    for p in D["pairs"]:
        P = out["pairs"][p["name"]] = dict(base=p["base"], families={})
        for fam, F in D["families"].items():
            b, i = M[(p["name"], F["anchors"]["B"])], M[(p["name"], F["anchors"]["I"])]
            cb, ci = causal(b, cfg), causal(i, cfg); kb, ki = competent(b, cfg), competent(i, cfg)
            okB = kb and cb != "HYBRID"; okI = ki and ci != "HYBRID" and ci != cb
            ev = okB if cfg["EVAL"] == "B" else (okB and okI)
            fev = ev and fclass(b["f"]) == cb and fclass(i["f"]) == ci
            arms = {}
            for an, A in D["arms"].items():
                if A["family"] != fam or p["base"] not in A.get("bases", "EM"): continue
                a = M[(p["name"], an)]; ph = phi(a, b, i, cfg); ca = causal(a, cfg); ka = competent(a, cfg); fa = fclass(a["f"])
                if not ka: cat = "NONCOMPETENT"
                elif ph != ph: cat = "INCONCLUSIVE(phi_undefined)"
                else:
                    cs = "SWITCH" if (ph >= cfg["PHI_HI"] and ca == ci) else ("NONE" if (ph <= cfg["PHI_LO"] and ca == cb) else "PARTIAL")
                    fs = "SWITCH" if fa == ci else ("NONE" if fa == cb else "AMBIG")
                    cat = f"HYBRID[{fa}]" if cs == "PARTIAL" else ("INCONCLUSIVE(causal_%s)" % cs if fs == "AMBIG" else {("SWITCH", "SWITCH"): "BOTH", ("SWITCH", "NONE"): "CAUSAL_ONLY", ("NONE", "SWITCH"): "FUNCTIONAL_ONLY", ("NONE", "NONE"): "NEITHER"}[(cs, fs)])
                arms[an] = dict(phi=ph, causal=ca, functional=fa, f=a["f"], competent=ka, category=cat, nec_E=a["nec_E"], nec_M=a["nec_M"], KL=a["KL"], acc=a["acc"])
            P["families"][fam] = dict(anchors=F["anchors"], evaluable=ev, functional_evaluable=fev, anchor_B=cb, anchor_I=ci, anchors_competent=[kb, ki], arms=arms)
    for h in D["hypotheses"]:
        vals, dropped, kind = {}, [], h.get("kind", "band")
        for pn, P in out["pairs"].items():
            if P["base"] != h["base"]: continue
            Fm = P["families"][h["family"]]
            used = [an for an in Fm["arms"] if f"'{an}'" in h.get("expr", "") or f'"{an}"' in h.get("expr", "") or an == h.get("arm")]
            if not Fm["evaluable"]: dropped.append((pn, "pair not evaluable")); continue
            if kind == "category" and not Fm["functional_evaluable"]: dropped.append((pn, "anchors not functionally concordant")); continue
            if any(not Fm["arms"][an]["competent"] for an in used): dropped.append((pn, "arm non-competent")); continue
            gm_ = [out["matching"].get(an, {}).get(pn, {}) for an in used if an in gated]
            if any("error" in g_ for g_ in gm_): dropped.append((pn, "control gate error (fail closed)")); continue
            if any(g_.get("na") for g_ in gm_) and not h.get("allow_gate_na"): dropped.append((pn, "control gate not applicable")); continue
            if any(not g_.get("ok", False) and not g_.get("na") for g_ in gm_): dropped.append((pn, "control not matched")); continue
            if kind == "category":
                ca_ = Fm["arms"][h["arm"]]["category"]
                if ca_ not in h.get("eligible", h["support"] + h["contradict"]): dropped.append((pn, f"not eligible ({ca_})")); continue
                vals[pn] = dict(category=ca_); continue
            aB, aI = D["families"][h["family"]]["anchors"]["B"], D["families"][h["family"]]["anchors"]["I"]; fB, fI = Fm["arms"][aB]["f"], Fm["arms"][aI]["f"]
            env = dict(phif={an: ((Fm["arms"][an]["f"] - fB) / (fI - fB) if abs(fI - fB) > 1e-9 else float("nan")) for an in Fm["arms"]},
                       phi={an: Fm["arms"][an]["phi"] for an in Fm["arms"]}, f={an: Fm["arms"][an]["f"] for an in Fm["arms"]},
                       func={an: Fm["arms"][an]["functional"] for an in Fm["arms"]}, I=Fm["anchor_I"], B=Fm["anchor_B"])
            v = float(eval(h["expr"], {"__builtins__": {}, "min": min, "max": max, "abs": abs}, env))
            if v != v: dropped.append((pn, "value undefined (NaN)")); continue
            nc = h.get("need_class"); cok = True
            if nc: cok = all(Fm["arms"][an]["causal"] == (Fm["anchor_I"] if nc == "I" else Fm["anchor_B"]) for an in used)
            vals[pn] = dict(v=v, class_ok=cok)
        n = len(vals); nbase = sum(1 for P in out["pairs"].values() if P["base"] == h["base"])
        if sum(1 for _, why in dropped if why == "arm non-competent") > nbase / 2: band = "NONCOMPETENT"     # pre-review C R2-6
        elif n < D["min_pairs"]: band = "INCONCLUSIVE"
        elif kind == "category":
            if all(x["category"] in h["support"] for x in vals.values()): band = "SUPPORTED"
            elif sum(x["category"] in h["contradict"] for x in vals.values()) > n / 2: band = "CONTRADICTED"
            else: band = "PARTIAL"
        else:
            lo, hi, d = h["lo"], h["hi"], cfg["DELTA"]
            if all(lo <= x["v"] <= hi and x["class_ok"] for x in vals.values()): band = "SUPPORTED"
            elif sum((x["v"] < lo - d) or (x["v"] > hi + d) or not x["class_ok"] for x in vals.values()) > n / 2: band = "CONTRADICTED"
            else: band = "PARTIAL"
        out["hypotheses"][h["id"]] = dict(band=band, n_evaluable=n, values=vals, dropped=dropped, **{k: h[k] for k in ("lo", "hi", "expr", "need_class", "arm", "support", "contradict", "kind", "family") if k in h})
    out["evaluable_pairs"] = {fam: {b_: [sum(1 for P in out["pairs"].values() if P["base"] == b_ and P["families"][fam]["evaluable"]), sum(1 for P in out["pairs"].values() if P["base"] == b_)] for b_ in ("E", "M")} for fam in D["families"]}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("design"); ap.add_argument("measure"); ap.add_argument("out"); ap.add_argument("--sens", default=None); ap.add_argument("--no-files", action="store_true"); ap.add_argument("--ckroot", default=os.path.join(H, "..", "checkpoints", "ft"))
    a = ap.parse_args(); D = json.load(open(a.design)); recs = [json.loads(l) for l in open(a.measure) if l.strip()]
    cfg = dict(CFG, **(SENS[a.sens] if a.sens else {}))
    errs, M, dup = provenance(D, recs, a.ckroot, files=not a.no_files)
    if errs:
        json.dump(dict(verdict="PROVENANCE_FAIL", errors=errs), open(a.out, "w"), indent=1); print("PROVENANCE_FAIL", len(errs)); [print(" ", e) for e in errs[:20]]; sys.exit(2)
    R = score(D, M, cfg, {} if a.no_files else matching(D, a.ckroot)); R["verdict"] = "SCORED_NO_FILES (not a frozen verdict)" if a.no_files else "SCORED"; R["duplicate_composed_states"] = dup
    R["design_sha256"] = sha(a.design); R["measure_sha256"] = sha(a.measure); R["sens"] = a.sens
    json.dump(R, open(a.out, "w"), indent=1)
    for k, v in R["hypotheses"].items(): print(k.ljust(30), v["band"].ljust(13), "n=%d" % v["n_evaluable"], {p: (round(x["v"], 3) if "v" in x else x["category"]) for p, x in v["values"].items()})
