"""Planted-case validation of score19.py v2 (mandate §9 F2/F4/F5; pre-review C items 1, 2, 4-9).
Part 1 (no-files): records built from real measured models (logs/scorer_validation19.json, stream 97 960 000) plus synthetic records;
  checks categories, per-base evaluability, need_class, category hypotheses, NaN handling, label recomputation, competence boundaries and
  every frozen sensitivity variant (evaluability recomputed in each).
Part 2 (files): a complete miniature confirmation world on disk (fake step-100 / step-1211 states, run_config.json, provenance.json, records)
  that scores cleanly, then one defect at a time, each of which must make the scorer fail closed (exit 2).
Writes logs/scorer_planted19.json; exits 1 on any failure."""
import os, sys, json, copy, subprocess, tempfile, hashlib, pickle, shutil
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import score19
import numpy as np
V = json.load(open(os.path.join(H, "..", "logs", "scorer_validation19.json")))["97960000"]
res = dict(checks=[], ok=True); T = tempfile.mkdtemp()
def check(name, cond, info=""):
    res["checks"].append(dict(check=name, passed=bool(cond), info=str(info))); res["ok"] &= bool(cond); print("PASS" if cond else "FAIL", name, info)
def run(design, recs, tag, sens=None, files=False, ck=None):
    dp, mp, op = (os.path.join(T, tag + x) for x in (".design.json", ".m.jsonl", ".out.json"))
    json.dump(design, open(dp, "w")); open(mp, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    cmd = [sys.executable, os.path.join(H, "score19.py"), dp, mp, op] + ([] if files else ["--no-files"]) + (["--sens", sens] if sens else []) + (["--ckroot", ck] if ck else [])
    rc = subprocess.run(cmd, capture_output=True, text=True).returncode
    return rc, (json.load(open(op)) if os.path.exists(op) else dict(verdict="CRASH"))
# ---------------- Part 1
def rec(run_, model, **over):
    m = V[model]; r = dict(run=run_, parent="uP160", stream=1, step=1211, ckpt_sha256="x", score_code_sha256="c", refs_sha256="r", **{k: m[k] for k in ("nec_E", "nec_M", "KL", "acc", "f")}); r.update(over); return r
MOD = dict(B="uP160_fx_L21_W21_D801_stop1211_B_base", I="uP160_fx_L21_W21_D801_ro0T100_stop1211_B_ro0100", sw="uP160_fx_L21_W21_D802_ro0T100_stop1211_B_ro0100",
           none="uP160_fx_L21_W21_D801_mskoq0T100_stop1211_XIV", hyb="uP160_fx_L23_W23_D811_gsgown2T100_stop1211_c18", nc="uP162_fx_L25_W25_D811_stop1211_scr",
           amb="uP160_fx_L21_W21_D813_gofown0b1e9T100_stop1211_c18", conly="uP160_fx_L21_W21_D802_ro0T100_stop1211_B_ro0100", fonly="uP160_fx_L21_W21_D801_mskoq0T100_stop1211_XIV",
           pureE_mid="uP160_fx_L21_W21_D801_mskoq0T100_stop1211_XIV")
SYN = dict(conly=dict(f=0.2), fonly=dict(f=-0.2), pureE_mid=dict(nec_E=0.45, nec_M=0.05))   # pureE_mid: still causally EARLY but phi in (0.25, 0.5)
EXPECT = dict(B="NEITHER", I="BOTH", sw="BOTH", none="NEITHER", hyb="HYBRID[AMBIGUOUS]", nc="NONCOMPETENT", amb="INCONCLUSIVE(causal_SWITCH)", conly="CAUSAL_ONLY", fonly="FUNCTIONAL_ONLY")
pairs = [dict(name=f"P{k}", base="E", L=21, W=21, D=k) for k in (1, 2, 3)] + [dict(name=f"Q{k}", base="M", L=23, W=23, D=k) for k in (1, 2, 3)]
def hb(i, base, expr, lo, hi, nc=None): return dict(id=i, family="F", base=base, expr=expr, lo=lo, hi=hi, **({"need_class": nc} if nc else {}))
D = dict(stream=1, parent="uP160", score_code_sha256="c", refs_sha256="r", min_pairs=2, pairs=pairs, families=dict(F=dict(anchors=dict(B="B", I="I"))),
         arms={a: dict(family="F", run="{pair}_" + a, transplant=None) for a in MOD},
         hypotheses=[hb("H_switch", "E", "phi['sw']", 0.75, 1.5, "I"), hb("H_switch_wrongpred", "E", "phi['sw']", -0.5, 0.25, "B"), hb("H_none", "E", "phi['none']", -0.25, 0.25, "B"),
                     hb("H_noncompetent", "E", "phi['nc']", -1, 1), hb("H_mid_class", "E", "phi['pureE_mid']", 0.25, 0.75, "I"), hb("H_M_side", "M", "phi['sw']", 0.75, 1.5, "I"),
                     dict(id="H_cat_conly", family="F", base="E", kind="category", arm="conly", support=["BOTH"], contradict=["CAUSAL_ONLY"], eligible=["BOTH", "CAUSAL_ONLY", "INCONCLUSIVE(causal_SWITCH)"]),
                     dict(id="H_cat_sw", family="F", base="E", kind="category", arm="sw", support=["BOTH"], contradict=["CAUSAL_ONLY"], eligible=["BOTH", "CAUSAL_ONLY", "INCONCLUSIVE(causal_SWITCH)"]),
                     dict(id="H_cat_none", family="F", base="E", kind="category", arm="none", support=["BOTH"], contradict=["CAUSAL_ONLY"], eligible=["BOTH", "CAUSAL_ONLY", "INCONCLUSIVE(causal_SWITCH)"]),
                     dict(id="H_dq", family="F", base="E", expr='phi["nc"]', lo=-1, hi=1),
                     hb("H_phif", "E", "phif['sw']", 0.5, 1.5)])
R = []
for p in pairs:
    for a, m in MOD.items():
        mm = m
        if p["name"] == "P2" and a == "I": mm = MOD["hyb"]          # P2: hybrid I anchor -> pair not evaluable
        if p["name"].startswith("Q") and a == "B": mm = MOD["nc"]    # every M pair: non-competent B anchor -> base M non-evaluable
        R.append(rec(f"{p['name']}_{a}", mm, **SYN.get(a, {})))
rc, O = run(D, R, "base"); check("scores when provenance is complete (no-files label)", rc == 0 and O["verdict"].startswith("SCORED_NO_FILES"), (rc, O["verdict"]))
for pn in ("P1", "P3"):
    A = O["pairs"][pn]["families"]["F"]["arms"]
    for a, e in EXPECT.items(): check(f"{pn} category {a}", A[a]["category"] == e, (A[a]["category"], round(A[a]["phi"], 3)))
check("P2 not evaluable (hybrid I anchor)", O["pairs"]["P2"]["families"]["F"]["evaluable"] is False)
for h, e in dict(H_switch="SUPPORTED", H_switch_wrongpred="CONTRADICTED", H_none="SUPPORTED", H_noncompetent="NONCOMPETENT", H_mid_class="CONTRADICTED",
                 H_M_side="INCONCLUSIVE", H_cat_conly="CONTRADICTED", H_cat_sw="SUPPORTED", H_cat_none="INCONCLUSIVE", H_dq="NONCOMPETENT", H_phif="SUPPORTED").items():
    check(f"band {h}", O["hypotheses"][h]["band"] == e, (O["hypotheses"][h]["band"], O["hypotheses"][h]["n_evaluable"]))
check("evaluable-pair counts reported per base", O["evaluable_pairs"]["F"] == {"E": [2, 3], "M": [0, 3]}, O["evaluable_pairs"])
# label recomputation: a record that claims to be competent but is not
R2 = [dict(r, competent=True, causal="EARLY", functional="EARLY") if r["run"] == "P1_sw" else r for r in R]
R2 = [dict(r, KL=0.30, acc=0.50) if r["run"] == "P1_sw" else r for r in R2]
rc, O2 = run(D, R2, "labels"); check("record labels ignored; competence recomputed", O2["pairs"]["P1"]["families"]["F"]["arms"]["sw"]["category"] == "NONCOMPETENT", O2["pairs"]["P1"]["families"]["F"]["arms"]["sw"]["category"])
# competence boundary: anchors at KL 0.05 are competent in the primary, not in comp_strict
R3 = [dict(r, KL=0.05) if r["run"] in ("P1_B", "P3_B") else r for r in R]
rc, O3 = run(D, R3, "comp"); rc2, O4 = run(D, R3, "comp_s", sens="comp_strict")
check("competence boundary 0.05: primary evaluable, comp_strict not", O3["evaluable_pairs"]["F"]["E"][0] == 2 and O4["evaluable_pairs"]["F"]["E"][0] == 0, (O3["evaluable_pairs"], O4["evaluable_pairs"]))
# NaN phi: identical anchors on a pair -> dropped, not PARTIAL
R5 = [dict(r, nec_E=V[MOD["B"]]["nec_E"], nec_M=V[MOD["B"]]["nec_M"]) if r["run"] == "P1_I" else r for r in R]
rc, O5 = run(dict(D, hypotheses=[hb("H_nan", "E", "phi['sw']", 0.75, 1.5)]), R5, "nan", sens="evalB")
check("NaN phi dropped with reason (evalB)", ("P1", "value undefined (NaN)") in [tuple(x) for x in O5["hypotheses"]["H_nan"]["dropped"]], O5["hypotheses"]["H_nan"]["dropped"])
# pooling refused
rc, O6 = run(dict(D, hypotheses=[hb("H_pool", "both", "phi['sw']", 0.75, 1.5)]), R, "pool"); check("pooled 'both' hypothesis fails closed", rc == 2, rc)
for sv in score19.SENS:
    rc, O7 = run(D, R, "s_" + sv, sens=sv); ev2 = O7["pairs"]["P2"]["families"]["F"]["evaluable"]
    check(f"sensitivity {sv} runs, evaluability recomputed (P2 evaluable={ev2})", rc == 0 and ev2 == (sv == "evalB"), {k: v["band"] for k, v in O7["hypotheses"].items()})
# matching gate (function level): ratio 1.0 passes, 3.0 fails, a missing state fails closed and drops the control
ck = os.path.join(T, "ckm"); Dm = copy.deepcopy(D); Dm["matching"] = [dict(arm="ctl", kind="dist", lo=0.67, hi=1.5)]
for p in Dm["pairs"]:
    p["Isuf"] = "_I"; p["win"] = dict(ctl="_C"); pre = f"uP160_fx_L{p['L']}_W{p['W']}_D{p['D']}"
    for suf, x in (("", 0.0), ("_I", 1.0), ("_C", 1.0 if p["name"] != "P3" else 3.0)):
        if p["name"] == "Q1" and suf == "_C": continue
        os.makedirs(os.path.join(ck, pre + suf + "_stop100_s100c19"), exist_ok=True); np.savez(os.path.join(ck, pre + suf + "_stop100_s100c19", "step100.npz"), np.full(4, x))
Mt = score19.matching(Dm, ck); check("matching gate: ratio, threshold, missing state fails", Mt["ctl"]["P1"]["ok"] and not Mt["ctl"]["P3"]["ok"] and not Mt["ctl"]["Q1"]["ok"], {k: v.get("ratio") for k, v in Mt["ctl"].items()})
Dm["arms"]["ctl"] = dict(family="F", run="{pair}_ctl", transplant=None); Dm["hypotheses"] = [hb("H_ctl", "E", "phi['ctl']", -0.25, 0.25)]
Rm = R + [rec(f"{p['name']}_ctl", MOD["none"]) for p in pairs]; Om = score19.score(Dm, {(r["run"].split("_")[0], r["run"].split("_", 1)[1]): r for r in Rm}, dict(score19.CFG), Mt)
check("unmatched control pair dropped", ("P3", "control not matched") in [tuple(x) for x in Om["hypotheses"]["H_ctl"]["dropped"]] and Om["hypotheses"]["H_ctl"]["band"] == "INCONCLUSIVE", Om["hypotheses"]["H_ctl"]["dropped"])
Om2 = score19.score(Dm, {(r["run"].split("_")[0], r["run"].split("_", 1)[1]): r for r in Rm}, dict(score19.CFG), {})
check("missing matching result fails closed", all(x[1] in ("control not matched", "pair not evaluable") for x in Om2["hypotheses"]["H_ctl"]["dropped"]) and Om2["hypotheses"]["H_ctl"]["n_evaluable"] == 0, Om2["hypotheses"]["H_ctl"]["dropped"])
# ---------------- Part 2: miniature on-disk world
PKG = os.path.abspath(os.path.join(H, "..")); ckw = os.path.join(PKG, "checkpoints", "planted19_world"); shutil.rmtree(ckw, ignore_errors=True); os.makedirs(ckw)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); HS = "harnesshash"; DEPM = {"x.py": "1"}
def wpair(nm, base, L, k):
    T_ = "own0" if base == "E" else "own2"
    return dict(name=nm, base=base, L=L, W=L, D=k, Isuf="_I", win=dict(vsg="_V", pnz="_N"), wenv=dict(I={"FX_INTERV": "gof:100", "FX_EDGES": T_, "FX_B": "1e9"}, vsg={"FX_INTERV": "vsg:100", "FX_EDGES": T_}, pnz={"FX_INTERV": "pnz:100", "FX_PNZ_SEED": str(9100 + k)}))
wp = [wpair("E1", "E", 21, 1), wpair("E2", "E", 21, 2), wpair("M1", "M", 23, 1), wpair("M2", "M", 23, 2)]
def rcfg(d, env, argv, bdir=None, bsrc=None):
    json.dump(dict(argv=["x"] + argv, env=env, code={"xix_train.py": HS}, branch_dir=bdir, branch_src=bsrc), open(os.path.join(d, "run_config.json"), "w"))
WR = []
for p in wp:
    L_ = p["L"]; pre = f"uP160_fx_L{L_}_W{L_}_D{p['D']}"; argv = ["uP160", str(L_), str(L_), str(p["D"])]; rng = {"s": p["D"]}
    pref = f"checkpoints/ft/{pre}_stop100_s100c19:checkpoints/ft/{pre}_I_stop100_s100c19"
    for suf, env, x in (("", {}, 0.0), ("_I", p["wenv"]["I"], 1.0), ("_V", p["wenv"]["vsg"], 0.5), ("_N", dict(p["wenv"]["pnz"], FX_PNZ_REF=pref), 0.2)):
        d = os.path.join(ckw, pre + suf + "_stop100_s100c19"); os.makedirs(d); np.savez(os.path.join(d, "step100.npz"), np.full(3, x), np.full(2, x))
        pickle.dump(dict(step=100, rng=rng, hist=[], opt=[np.zeros(1), np.full(3, x), np.full(2, x), np.full(3, x), np.full(2, x)]), open(os.path.join(d, "opt_step100.pkl"), "wb")); rcfg(d, env, argv)
    for an, src_I in (("fullB", False), ("fullI", True), ("rn", None)):
        tpd = os.path.join(ckw, f"tp_{p['name']}_{an}"); os.makedirs(tpd); s0 = os.path.join(ckw, pre + ("_I" if src_I else "") + "_stop100_s100c19")
        shutil.copy(os.path.join(s0, "step100.npz"), tpd); shutil.copy(os.path.join(ckw, pre + "_stop100_s100c19", "opt_step100.pkl") if not src_I else os.path.join(s0, "opt_step100.pkl"), tpd)
        if src_I is None: np.savez(os.path.join(tpd, "step100.npz"), np.full(3, 0.3), np.full(2, 0.0))
        seed_ = (19000 + p["D"] + (500 if p["base"] == "M" else 0)) if src_I is None else None
        pv = dict(params_from_I=["L0_QK"] if src_I else [], perturbed_groups=["L0_QK"] if src_I is None else [], perturb_seed=seed_, alpha=None, opt="B" if src_I is None else "follow",
                  inputs={f"{l}/{f}": sha(os.path.join(ckw, pre + s + "_stop100_s100c19", f)) for l, s in (("B", ""), ("I", "_I")) for f in ("step100.npz", "opt_step100.pkl")},
                  outputs={f: sha(os.path.join(tpd, f)) for f in ("step100.npz", "opt_step100.pkl")}); json.dump(pv, open(os.path.join(tpd, "provenance.json"), "w"))
    for an, srcd, model in (("fullB", f"tp_{p['name']}_fullB", MOD["B"]), ("fullI", f"tp_{p['name']}_fullI", MOD["I"]), ("vsg", pre + "_V_stop100_s100c19", MOD["hyb"]),
                            ("rn", f"tp_{p['name']}_rn", MOD["none"]), ("pz", pre + "_N_stop100_s100c19", MOD["none"])):
        run_ = f"{pre}_stop1211_br100_{p['name']}_{an}"; d = os.path.join(ckw, run_); os.makedirs(d); np.savez(os.path.join(d, "step1211.npz"), np.full(3, hash(run_) % 97))
        sd = os.path.join(ckw, srcd); rcfg(d, {"FX_NOFALLBACK": "1", "FX_BRANCH": srcd + ":100", "FX_STOP": "1211"}, argv, sd, {f: sha(os.path.join(sd, f)) for f in ("step100.npz", "opt_step100.pkl")})
        WR.append(rec(run_, model, stream=7, ckpt_sha256=sha(os.path.join(d, "step1211.npz")), dep_sha256=DEPM))
def tpa(an, grp): return dict(family="T", run="{parent}_fx_L{L}_W{W}_D{D}_stop1211_br100_{pair}_" + an, transplant=dict(tp="tp_{pair}_" + an, params_from_I=grp, opt="follow", perturb=None, alpha=None,
                              src_dirs=dict(B="{parent}_fx_L{L}_W{W}_D{D}_stop100_s100c19", I="{parent}_fx_L{L}_W{W}_D{D}{Isuf}_stop100_s100c19")))
xs = sha(os.path.join(PKG, "src", "xix_score.py"))
WD = dict(stream=7, parent="uP160", score_code_sha256=xs, refs_sha256="r", dep_sha256={"src/xix_score.py": xs}, dep_sha256_measure=DEPM, harness_sha256=HS, min_pairs=2, pairs=wp,
          families=dict(T=dict(anchors=dict(B="fullB", I="fullI"))), arms=dict(fullB=tpa("fullB", []), fullI=tpa("fullI", ["L0_QK"]), vsg=dict(family="T", run="{parent}_fx_L{L}_W{W}_D{D}_stop1211_br100_{pair}_vsg", transplant=None, branch_from="vsg"),
          rn=dict(tpa("rn", ["L0_QK"]), transplant=dict(tpa("rn", ["L0_QK"])["transplant"], opt="B", perturb=19000)), pz=dict(family="T", run="{parent}_fx_L{L}_W{W}_D{D}_stop1211_br100_{pair}_pz", transplant=None, branch_from="pnz")),
          hypotheses=[dict(id="H_vsg", family="T", base="E", expr="phi['vsg']", lo=0.0, hi=1.0), dict(id="H_vsg_M", family="T", base="M", expr="phi['vsg']", lo=0.0, hi=1.0)])
WR = [dict(r, score_code_sha256=xs) for r in WR]
rc, OW = run(WD, WR, "world", files=True, ck=ckw); check("file-mode world scores cleanly", rc == 0 and OW["verdict"] == "SCORED", (rc, OW.get("errors", [])[:3]))
def corrupt(name, fn, expect):
    """apply one defect IN PLACE (paths in run_config must stay valid), score, restore; the scorer must fail closed FOR THAT REASON"""
    bk = os.path.join(T, "bk_world"); shutil.rmtree(bk, ignore_errors=True); shutil.copytree(ckw, bk); d2, r2 = copy.deepcopy(WD), copy.deepcopy(WR); fn(ckw, d2, r2)
    rc, Ox = run(d2, r2, "wd_" + name.replace(" ", "_"), files=True, ck=ckw); errs = Ox.get("errors") or []
    check(f"file-mode fail-closed: {name}", rc == 2 and Ox["verdict"] == "PROVENANCE_FAIL" and any(expect in e for e in errs), (rc, len(errs), [e[:80] for e in errs if expect in e][:1] or errs[:1]))
    shutil.rmtree(ckw); shutil.copytree(bk, ckw)
def edit_json(path, **kv):
    j = json.load(open(path)); j.update(kv); json.dump(j, open(path, "w"))
corrupt("checkpoint replaced", lambda w, d, r: np.savez(os.path.join(w, "uP160_fx_L21_W21_D1_stop1211_br100_E1_vsg", "step1211.npz"), np.zeros(3)), "checkpoint hash mismatch")
corrupt("run_config missing", lambda w, d, r: os.remove(os.path.join(w, "uP160_fx_L21_W21_D1_stop1211_br100_E1_fullI", "run_config.json")), "run_config unreadable")
corrupt("wrong seed in argv", lambda w, d, r: edit_json(os.path.join(w, "uP160_fx_L21_W21_D1_stop1211_br100_E1_fullB", "run_config.json"), argv=["x", "uP160", "21", "21", "9"]), "does not match the pair")
corrupt("fallback allowed", lambda w, d, r: edit_json(os.path.join(w, "uP160_fx_L21_W21_D2_stop1211_br100_E2_fullB", "run_config.json"), env={"FX_BRANCH": "tp_E2_fullB:100"}), "FX_NOFALLBACK not set")
corrupt("harness hash differs", lambda w, d, r: edit_json(os.path.join(w, "uP160_fx_L21_W21_D1_stop1211_br100_E1_vsg", "run_config.json"), code={"xix_train.py": "other"}), "harness hash differs")
corrupt("branch source outside package", lambda w, d, r: edit_json(os.path.join(w, "uP160_fx_L21_W21_D1_stop1211_br100_E1_fullB", "run_config.json"), branch_dir="/tmp/elsewhere"), "outside this package")
corrupt("transplant groups differ", lambda w, d, r: edit_json(os.path.join(w, "tp_E1_fullI", "provenance.json"), params_from_I=["L1_QK"]), "provenance groups")
corrupt("transplant alpha differs", lambda w, d, r: edit_json(os.path.join(w, "tp_E1_fullI", "provenance.json"), alpha=0.5), "provenance alpha")
corrupt("composed state changed after the run", lambda w, d, r: np.savez(os.path.join(w, "tp_E2_fullI", "step100.npz"), np.full(3, 9.0), np.full(2, 9.0)), "composed step100.npz hash mismatch")
corrupt("window state trained with other settings", lambda w, d, r: edit_json(os.path.join(w, "uP160_fx_L21_W21_D1_V_stop100_s100c19", "run_config.json"), env={"FX_INTERV": "gsg:100", "FX_EDGES": "own0"}), "window state was trained with")
corrupt("native I trained with other settings", lambda w, d, r: edit_json(os.path.join(w, "uP160_fx_L21_W21_D2_I_stop100_s100c19", "run_config.json"), env={"FX_INTERV": "gof:100", "FX_EDGES": "own0", "FX_B": "2.3"}), "trained with")
corrupt("fullB not identical to native B", lambda w, d, r: (np.savez(os.path.join(w, "tp_E1_fullB", "step100.npz"), np.full(3, 0.1), np.full(2, 0.0)), edit_json(os.path.join(w, "tp_E1_fullB", "provenance.json"), outputs={"step100.npz": sha(os.path.join(w, "tp_E1_fullB", "step100.npz")), "opt_step100.pkl": sha(os.path.join(w, "tp_E1_fullB", "opt_step100.pkl"))})), "not bit-identical")
corrupt("measurement dependency hash differs", lambda w, d, r: r.__setitem__(0, dict(r[0], dep_sha256={"x.py": "2"})), "measurement dependency hashes")
corrupt("duplicate record", lambda w, d, r: r.append(r[0]), "measurement records (need exactly 1)")
corrupt("missing record", lambda w, d, r: r.pop(), "0 measurement records")
corrupt("wrong stream", lambda w, d, r: r.__setitem__(1, dict(r[1], stream=8)), "stream=")
corrupt("reference hash differs", lambda w, d, r: r.__setitem__(2, dict(r[2], refs_sha256="z")), "refs_sha256=")
corrupt("design lacks harness hash", lambda w, d, r: d.pop("harness_sha256"), "lacks harness")
corrupt("scorer library changed", lambda w, d, r: d.__setitem__("score_code_sha256", "z"), "differs from the frozen hash")
corrupt("wrong perturbation seed (M pair)", lambda w, d, r: edit_json(os.path.join(w, "tp_M2_rn", "provenance.json"), perturb_seed=19002), "provenance perturb_seed")
corrupt("pnz scales from another pair", lambda w, d, r: edit_json(os.path.join(w, "uP160_fx_L23_W23_D1_N_stop100_s100c19", "run_config.json"), env=dict(json.load(open(os.path.join(w, "uP160_fx_L23_W23_D1_N_stop100_s100c19", "run_config.json")))["env"], FX_PNZ_REF="checkpoints/ft/x:checkpoints/ft/y")), "noise scales taken from")
corrupt("M window state trained on the E target", lambda w, d, r: edit_json(os.path.join(w, "uP160_fx_L23_W23_D2_V_stop100_s100c19", "run_config.json"), env={"FX_INTERV": "vsg:100", "FX_EDGES": "own0"}), "window state was trained with")
shutil.rmtree(ckw)
# ---------------- Part 3: kl100 gate on real development step-100 states (symlinked under confirmation names)
try:
    ck3 = tempfile.mkdtemp(); C = os.path.join(PKG, "checkpoints", "ft"); P8 = "uP160_fx_L21_W21_D801"
    for src, dst in ((P8 + "_stop100_s100dev", P8 + "_stop100_s100c19"), (P8 + "_gofown0b1e9T100_stop100_s100dev", P8 + "_I_stop100_s100c19"), (P8 + "_lblp0.5s1T100_stop100_s100dev", P8 + "_L_stop100_s100c19"),
                     (P8 + "_stop100_s100dev", P8 + "_Z_stop100_s100c19")):
        os.symlink(os.path.join(C, src), os.path.join(ck3, dst))
    for src, dst in ((P8 + "_stop100_s100dev", "uP160_fx_L21_W21_D802_stop100_s100c19"), (P8 + "_stop100_s100dev", "uP160_fx_L21_W21_D802_I_stop100_s100c19"), (P8 + "_lblp0.5s1T100_stop100_s100dev", "uP160_fx_L21_W21_D802_L_stop100_s100c19")):
        os.symlink(os.path.join(C, src), os.path.join(ck3, dst))
    Dk = dict(parent="uP160", stream=86300000, pairs=[dict(name="E801", L=21, W=21, D=801, Isuf="_I", win=dict(lbl="_L", z="_Z")), dict(name="E802", L=21, W=21, D=802, Isuf="_I", win=dict(lbl="_L"))],
              matching=[dict(arm="lbl", kind="kl100", lo=0.5, hi=1e9, min_block_excess=0.005)])
    Mk = score19.matching(Dk, ck3); Dz = dict(Dk, pairs=Dk["pairs"][:1], matching=[dict(arm="z", kind="kl100", lo=0.5, hi=1e9)]); Mz = score19.matching(Dz, ck3)
    check("kl100 gate: dev lbl passes (ratio 0.88)", Mk["lbl"]["E801"]["ok"] and abs(Mk["lbl"]["E801"]["ratio"] - 0.879) < 0.01, Mk["lbl"]["E801"])
    check("kl100 gate: block excess 0 -> loss matching not applicable, control with positive excess usable", Mk["lbl"]["E802"]["ok"] and Mk["lbl"]["E802"]["ratio"] is None, Mk["lbl"]["E802"])
    check("kl100 gate: control without loss disturbance fails", not Mz["z"]["E801"]["ok"], Mz["z"]["E801"])
except Exception as e: check("kl100 gate tests ran", False, e)
json.dump(res, open(os.path.join(H, "..", "logs", "scorer_planted19.json"), "w"), indent=1); print("ALL PASS" if res["ok"] else "FAILURES", len(res["checks"]), "checks"); sys.exit(0 if res["ok"] else 1)
