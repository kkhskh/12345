"""Job planner (v2).  The plan is a pure function of (frozen protocol, preset, VERIFIED upstream outputs): parents -> pretraining ->
parent validity -> screening on two data orders -> eligibility (written once, never revised) -> windows -> compositions ->
continuations -> measurements (step 100 states, 150 / 300 compatibility, 1211, 4000) -> Track B (selection -> parent matching ->
batteries) -> stability.  Re-planning after jobs finish only ADDS jobs whose inputs now exist; it never changes a job.
Every job is a dict: id, track, phase, priority, argv, env, outputs (relative to RUN_ROOT), deps, and registry metadata."""
import os, json, sys, math
from cor_research import env as E, protocol as PR

PY = sys.executable
HARN = E.HARNESS; COMPOSE = os.path.join(os.path.dirname(HARN), "compose_xx.py")      # absolute; jobs run with cwd = RUN_ROOT
P = PR.full()                     # module default (historical schedule); plan()/design() use PR.effective(preset)


def T_of(base): return "own0" if base == "E" else "own2"


def win_part(w, base, D):
    """the harness's intervention tag fragment for the window modes used in XX (verified against the harness in tests/test_tags.py)"""
    if w == "B": return ""
    if w == "I": return f"_gof{T_of(base)}b1e9T100"
    if w in ("vsgo", "rsg"): return f"_{w}{T_of(base)}T100"
    if w == "lbl": return f"_lblp0.5s{lbl_seed(base, D)}T100"
    raise KeyError(w)


def lbl_seed(base, D): return 10_000 + D + (500 if base == "M" else 0)
def p_seed(base, D): return 9_000 + D + (500 if base == "M" else 0)
def sfx(C): return "" if C["STEPS"]["primary"] == 4000 else f"_s{C['STEPS']['primary']}"
def scr_tag(p, s, D, C=P): return f"{p}_fx_L{s}_W{s}_D{D}_stop{C['SCREEN']['stop']}_scr20"
def win_tag(p, s, D, w, base, C=P): return f"{p}_fx_L{s}_W{s}_D{D}{sfx(C)}{win_part(w, base, D)}_stop100_s100x20"
def cont_tag(p, s, D, base, arm, C=P): return f"{p}_fx_L{s}_W{s}_D{D}{sfx(C)}_br100_{base}{D}_{arm}"
def stab_tag(p, s, D, base, arm, lr, C=P): return f"{p}_fx_L{s}_W{s}_D{D}_s{C['STEPS']['stability_end']}_br{C['STEPS']['primary']}_stab{lr}_{base}{D}_{arm}"
def tp_tag(p, base, D, arm): return f"tp_{p}_{base}{D}_{arm}"
def mpath(track, tag, step, stream): return f"measure/{track}/{tag}@{step}_{stream}.json"
def harness_env(**kw): return dict(FX_NOFALLBACK="1", **{k: str(v) for k, v in kw.items()})
def job(id, track, phase, priority, argv, env, outputs, deps=(), **meta): return dict(id=id, track=track, phase=phase, priority=priority, argv=argv, env=env, outputs=list(outputs), deps=list(deps), **meta)
def arm_def(C, a): return C["ARMS"].get(a) or C["HIGH_EXTRA_ARMS"][a]


def measure_job(jid, track, tag, parent, step, stream, deps, pr, **meta):
    mp = mpath(track, tag, step, stream)
    return job(jid, "A", "measure", pr, [PY, "-m", "cor_research.scoring.measure", f"ft/{tag}", parent, str(step), str(stream), mp], {}, [mp], deps, parent=parent, stream=stream, **meta)


def parent_record(p):
    NV = E.patch_parents(); path = NV.parent_path(p); js = os.path.join(os.path.dirname(path), "pre.json")
    return (json.load(open(js)) if os.path.exists(js) else None), path


def parent_valid(p):
    rec, _ = parent_record(p)
    if rec is None: return None
    return bool(rec["eval"]["aux_acc"] >= P["PARENT_VALIDITY"]["min_aux_acc"])


def classify(r):
    c = P["CLASS"]
    if r["nec_E"] >= c["E_hi"] and r["nec_M"] <= c["E_lo_M"]: return "EARLY"
    if r["nec_M"] >= c["M_hi"] and r["nec_E"] <= c["M_lo_E"]: return "MIXTURE"
    return "HYBRID"


def competent(r): return bool(r["KL"] <= P["COMP"]["KL"] and r["acc"] >= P["COMP"]["acc"])


def _verified_screen(root, R, C, p, s, D, frozen_code):
    """a screening record is used only if its measurement job and its training job are registered done and the hashes / stream match"""
    tg = scr_tag(p, s, D, C); rel = mpath("screen", tg, C["SCREEN"]["stop"], C["SCREEN"]["stream"]); f = os.path.join(root, rel)
    mj, tj = R.get(f"mscr:{p}:s{s}:D{D}", {}), R.get(f"scr:{p}:s{s}:D{D}", {})
    if mj.get("status") != "done" or tj.get("status") != "done" or not os.path.exists(f): return None
    r = json.load(open(f)); ck = f"ft/{tg}/step{C['SCREEN']['stop']}.npz"
    if r["stream"] != C["SCREEN"]["stream"] or tj["output_hashes"].get(ck) != r["ckpt_sha256"] or mj["output_hashes"].get(rel) != E.sha256(f):
        raise RuntimeError(f"screening record {f} fails provenance (integrity failure)")
    if frozen_code is not None and r["code_sha256"] != frozen_code: raise RuntimeError(f"screening record {f}: measurement code differs from the frozen code")
    return r


def eligibility(preset, root=None):
    """per parent, once its screening is complete and verified -> units.  Written to RUN_ROOT/eligibility.json once per parent; a later
    recomputation that differs from the written record is an integrity failure (raises).  Failed pretraining is recorded."""
    from cor_research.jobs import runner as RN
    root = root or E.RUN_ROOT; C = PR.effective(preset); pre = C["PRESETS"][preset]; path = os.path.join(root, "eligibility.json")
    old = E.read_json(path, {}) or {}; new = dict(old); R = RN.load_all(root)
    fz = E.read_json(os.path.join(E.PKG, "protocols", "FREEZE.json")) or {}; frozen_code = fz.get("measure_code_sha256") if not C["TECHNICAL_ONLY"] else None
    for p in pre["parents"]:
        if pre.get("forced_units"): continue
        v = parent_valid(p)
        if v is None: continue                       # pretraining not done (failed pretraining is reported from the registry, never frozen here: review C r2 N3)
        elif v is False: rec = dict(status="INVALID_PRETRAINING", aux_acc=parent_record(p)[0]["eval"]["aux_acc"], units={})
        else:
            recs = {}
            for s in pre["screen_seeds"]:
                for D in (C["SCREEN"]["D"], C["SCREEN"]["D2"]):
                    r = _verified_screen(root, R, C, p, s, D, frozen_code)
                    if r is None:
                        sj = R.get(f"scr:{p}:s{s}:D{D}", {}); mj = R.get(f"mscr:{p}:s{s}:D{D}", {})
                        if any(j.get("status") == "failed" and len(j.get("attempts", [])) >= 3 for j in (sj, mj)):   # terminal failure (3 attempts): this SEED
                            recs[f"{s}:D{D}"] = dict(failed=True, cls=None, competent=False); continue        # is non-qualifying (rule frozen in SCREEN.unit_rule)
                        recs = None; break
                    recs[f"{s}:D{D}"] = dict(KL=r["KL"], acc=r["acc"], nec_E=r["nec_E"], nec_M=r["nec_M"], cls=classify(r), competent=competent(r))
                if recs is None: break
            if recs is None: continue
            rec = None
            units = {}
            for base, cls in (("E", "EARLY"), ("M", "MIXTURE")):
                ok = [s for s in pre["screen_seeds"] if all(recs[f"{s}:D{D}"]["competent"] and recs[f"{s}:D{D}"]["cls"] == cls for D in (C["SCREEN"]["D"], C["SCREEN"]["D2"]))]
                units[base] = dict(eligible=bool(ok), seed=(min(ok) if ok else None), qualifying_seeds=ok)
            if rec is None: rec = dict(status="SCREENED", previously_observed=p in C["PREVIOUSLY_OBSERVED"], screening=recs, units=units)
        if p in old and json.dumps(old[p], sort_keys=True) != json.dumps(rec, sort_keys=True):
            raise RuntimeError(f"eligibility for {p} changed after it was recorded: integrity failure")
        new[p] = rec
    if new != old: E.atomic_json(new, path)
    return new


def units_of(preset, root, C, EL):
    pre = C["PRESETS"][preset]
    if pre.get("forced_units"): u = [tuple(x) for x in pre["forced_units"]]
    else: u = [(p, b, EL[p]["units"][b]["seed"]) for p in pre["parents"] if p in EL and EL[p]["status"] == "SCREENED" for b in "EM" if EL[p]["units"][b]["eligible"]]
    return u[: pre["max_units"]] if pre.get("max_units") else u


def arms_for(C, preset, base): return [a for a in C["PRESETS"][preset]["arms"] if base in arm_def(C, a)["bases"]]


def plan(preset, root=None, tracks="ABC"):
    root = root or E.RUN_ROOT; C = PR.effective(preset); pre = C["PRESETS"][preset]; J = []; ST = C["STEPS"]; XS = [] if ST["primary"] == 4000 else [str(ST["primary"])]
    SP, SW = C["STREAMS"]["measure_primary"], C["STREAMS"]["measure_window"]
    if "C" in tracks: J.append(job("math:trackC", "C", "math", 0, [PY, "-m", "cor_research.mathematics.run_track_c", "track_c/track_c_math.json"], {}, ["track_c/track_c_math.json"]))
    if not ("A" in tracks or "B" in tracks): return J
    for p in pre["parents"]:
        if p.startswith("n"):
            J.append(job(f"pre:{p}", "A", "pretrain", 1, [PY, "-m", "cor_research.training.pretrain", p[2:], str(C["PRETRAIN"]["steps"])], {},
                         [os.path.relpath(os.path.join(E.PARENT_ROOT, f"z{p[1:]}", "pre.json"), root)], parent=p, target_steps=C["PRETRAIN"]["steps"], start_ckpt="random init"))
        if pre.get("forced_units") or not parent_valid(p): continue
        for s in pre["screen_seeds"]:
            for D in (C["SCREEN"]["D"], C["SCREEN"]["D2"]):
                tg = scr_tag(p, s, D, C); dep = [f"pre:{p}"] if p.startswith("n") else []
                J.append(job(f"scr:{p}:s{s}:D{D}", "A", "screen", 2, [PY, HARN, p, str(s), str(s), str(D)], harness_env(FX_STOP=C["SCREEN"]["stop"], FX_TAGSUF="_scr20"),
                             [f"ft/{tg}/step{C['SCREEN']['stop']}.npz"], dep, parent=p, seed=s, data_order=D, arm="screen_baseline", start_ckpt=f"parent {p}", target_steps=C["SCREEN"]["stop"]))
                J.append(measure_job(f"mscr:{p}:s{s}:D{D}", "screen", tg, p, C["SCREEN"]["stop"], C["SCREEN"]["stream"], [f"scr:{p}:s{s}:D{D}"], 2, seed=s, data_order=D, arm="screen_baseline"))
    EL = eligibility(preset, root)
    for (p, base, s) in units_of(preset, root, C, EL):
        T = T_of(base); arms = arms_for(C, preset, base)
        for D in C["D_LIST"]:
            need = {"B", "I"}
            for a in arms:
                ad = arm_def(C, a); need |= set(ad["src"]) if ad["kind"] != "window" else {ad["window"]}
            for w in sorted(need):
                env = {k: v.format(T=T, lblseed=lbl_seed(base, D)) for k, v in C["WINDOW"][w].items()}; tg = win_tag(p, s, D, w, base, C)
                J.append(job(f"win:{p}:{base}{D}:{w}", "A", "window", 3, [PY, HARN, p, str(s), str(s), str(D)] + XS, harness_env(FX_STOP=100, FX_SAVEOPT=1, FX_SAVEOPT_AT=100, FX_WLOSS=1, FX_TAGSUF="_s100x20", **env),
                             [f"ft/{tg}/step100.npz", f"ft/{tg}/opt_step100.pkl"], parent=p, base=base, seed=s, data_order=D, arm=f"window_{w}", start_ckpt=f"parent {p}", target_steps=100))
                J.append(measure_job(f"mwin:{p}:{base}{D}:{w}", "window", tg, p, 100, SW, [f"win:{p}:{base}{D}:{w}"], 3, base=base, seed=s, data_order=D, arm=f"window_{w}"))
            for arm in arms:
                a = arm_def(C, arm); anchor = arm in ("fullB", "fullI"); pr = 4 if anchor else 5; ct = cont_tag(p, s, D, base, arm, C)
                if a["kind"] in ("transplant", "perturb"):
                    sb, si = (win_tag(p, s, D, x, base, C) for x in a["src"]); tp = tp_tag(p, base, D, arm)
                    groups = a["groups_by_base"][base] if "groups_by_base" in a else a["groups"]
                    argv = [PY, COMPOSE, f"ft/{sb}", f"ft/{si}", tp, "--params-from-I", groups, "--opt", a["opt"]]
                    if a["kind"] == "perturb": argv += ["--perturb", str(p_seed(base, D))]
                    if a.get("alpha") is not None: argv += ["--alpha", str(a["alpha"])]
                    J.append(job(f"tp:{p}:{base}{D}:{arm}", "A", "compose", pr, argv, {}, [f"ft/{tp}/step100.npz", f"ft/{tp}/opt_step100.pkl", f"ft/{tp}/provenance.json"],
                                 [f"win:{p}:{base}{D}:{x}" for x in a["src"]], parent=p, base=base, seed=s, data_order=D, arm=arm))
                    J.append(measure_job(f"mtp:{p}:{base}{D}:{arm}", "composed", tp, p, 100, SW, [f"tp:{p}:{base}{D}:{arm}"], pr, base=base, seed=s, data_order=D, arm=arm))
                    src, deps = tp, [f"tp:{p}:{base}{D}:{arm}"]
                else: src, deps = win_tag(p, s, D, a["window"], base, C), [f"win:{p}:{base}{D}:{a['window']}"]
                stab = pre["stability"] == "all_arms" or (pre["stability"] and anchor)
                env = harness_env(FX_BRANCH=f"{src}:100", FX_TAGSUF=f"_{base}{D}_{arm}", **(dict(FX_SAVEOPT=1, FX_SAVEOPT_AT=ST["primary"]) if stab else {}))
                outs = [f"ft/{ct}/final.json", f"ft/{ct}/step{ST['horizon_hist']}.npz", f"ft/{ct}/step{ST['primary']}.npz"] + ([f"ft/{ct}/opt_step{ST['primary']}.pkl"] if stab else [])
                keep = [f"step{x}.npz" for x in ST["early_checks"] + [ST["horizon_hist"], ST["primary"]]]
                J.append(job(f"cont:{p}:{base}{D}:{arm}", "A", "continue", pr, [PY, HARN, p, str(s), str(s), str(D)] + XS, env, outs, deps, parent=p, base=base, seed=s, data_order=D, arm=arm,
                             start_ckpt=f"ft/{src}/step100.npz", target_steps=ST["primary"], prune=(None if (anchor or stab) else dict(dir=f"ft/{ct}", keep=keep))))
                for step in ST["early_checks"] + [ST["horizon_hist"], ST["primary"]]:
                    J.append(measure_job(f"meas:{p}:{base}{D}:{arm}@{step}", "trackA", ct, p, step, SP, [f"cont:{p}:{base}{D}:{arm}"], pr, base=base, seed=s, data_order=D, arm=arm))
                if anchor and pre.get("trackb") and "B" in tracks:
                    J.append(job(f"fbsel:{p}:{base}{D}:{arm}", "B", "functional", 6, [PY, "-m", "cor_research.scoring.trackb_run", "select", f"ft/{ct}", p, f"trackB/sel/{ct}.json"], dict(COR_PRESET=preset),
                                 [f"trackB/sel/{ct}.json"], [f"cont:{p}:{base}{D}:{arm}"], parent=p, base=base, seed=s, data_order=D, arm=arm))
                if stab:
                    for lr, tagl in ((ST["stability_lr"], "lo"), (ST["stability_lr_hi"], "hi")):
                        if tagl == "hi" and not anchor: continue                         # full-LR positive control: anchors only
                        sg = stab_tag(p, s, D, base, arm, lr, C)
                        J.append(job(f"stab{tagl}:{p}:{base}{D}:{arm}", "A", "stability", 7, [PY, HARN, p, str(s), str(s), str(D), str(ST["stability_end"])],
                                     harness_env(FX_BRANCH=f"{ct}:{ST['primary']}", FX_STAB_LR=lr, FX_EXTRA_SNAPS=",".join(map(str, C["STABILITY_SNAPS"])), FX_TAGSUF=f"_{base}{D}_{arm}"),
                                     [f"ft/{sg}/final.json", f"ft/{sg}/step{ST['stability_end']}.npz"], [f"cont:{p}:{base}{D}:{arm}"], parent=p, base=base, seed=s, data_order=D, arm=arm,
                                     start_ckpt=f"ft/{ct}/step{ST['primary']}.npz", target_steps=ST["stability_end"]))
                        for step in C["STABILITY_SNAPS"] + [ST["stability_end"]]:
                            J.append(measure_job(f"mstab{tagl}:{p}:{base}{D}:{arm}@{step}", "stability", sg, p, step, SP, [f"stab{tagl}:{p}:{base}{D}:{arm}"], 7, base=base, seed=s, data_order=D, arm=arm))
    if pre.get("trackb") and "B" in tracks:                                              # Track B parent-level matching, then batteries
        byp = {}
        for j in J:
            if j["id"].startswith("fbsel:"): byp.setdefault(j["parent"], []).append(j)
        for p, sel in byp.items():
            out = f"trackB/match/{p}.json"
            J.append(job(f"fbmatch:{p}", "B", "functional", 6, [PY, "-m", "cor_research.scoring.trackb_run", "match", p, out] + [s_["outputs"][0] for s_ in sel], dict(COR_PRESET=preset),
                         [out], [s_["id"] for s_ in sel], parent=p))
            for s_ in sel:
                ct = s_["outputs"][0].split("/")[-1][:-5]; o = f"trackB/{ct}.json"
                J.append(job("fbbat:" + s_["id"][6:], "B", "functional", 6, [PY, "-m", "cor_research.scoring.trackb_run", "battery", f"ft/{ct}", p, out, o], dict(COR_PRESET=preset), [o],
                             [f"fbmatch:{p}"], parent=p, base=s_["base"], seed=s_["seed"], data_order=s_["data_order"], arm=s_["arm"]))
    return J


def design(preset, root=None):
    """the realised confirmation design (units and pairs) for the scorers; derived only from eligibility.json and the protocol"""
    root = root or E.RUN_ROOT; C = PR.effective(preset); pre = C["PRESETS"][preset]; EL = E.read_json(os.path.join(root, "eligibility.json"), {}) or {}
    units = [dict(parent=p, base=b, seed=s) for p, b, s in units_of(preset, root, C, EL)]
    return dict(preset=preset, parents=pre["parents"], eligibility=EL, units=units, D=C["D_LIST"], arms=pre["arms"], previously_observed=C["PREVIOUSLY_OBSERVED"],
                arms_by_base={b: arms_for(C, preset, b) for b in "EM"},
                tags={f"{u['parent']}:{u['base']}{D}:{a}": cont_tag(u["parent"], u["seed"], D, u["base"], a, C) for u in units for D in C["D_LIST"] for a in arms_for(C, preset, u["base"])},
                windows={f"{u['parent']}:{u['base']}{D}:{w}": win_tag(u["parent"], u["seed"], D, w, u["base"], C) for u in units for D in C["D_LIST"] for w in ("B", "I", "lbl", "vsgo", "rsg")},
                tps={f"{u['parent']}:{u['base']}{D}:{a}": tp_tag(u["parent"], u["base"], D, a) for u in units for D in C["D_LIST"] for a in arms_for(C, preset, u["base"]) if arm_def(C, a)["kind"] != "window"})
