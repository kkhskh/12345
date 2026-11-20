"""Planted-result tests (mandate §12; v2 after reviews A/B/C): every scorer gets confirmation, falsification, inconclusive, missing-input
refusal, wrong-hash refusal and incompetent-model cases, plus: band change keeps evaluability (XVIII), lbl CONTROL_INVALID, reverse-transplant
INCOMPATIBLE, Track B shared-D null, Track B underpowered -> NON_IDENTIFIABLE (not B2), eligibility immutability, runner (failed job kept,
definition change refused, unregistered outputs blocked), integrity check 14 on a corrupted checkpoint.  Synthetic inputs only.
python tests/test_scorers_planted.py -> table + $COR_TEST_OUT/planted_results.json (default: a temporary directory); exit 1 on any failure."""
import os, sys, json, tempfile, copy
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG); os.environ.setdefault("COR_DEVICE", "cpu")
import numpy as np
from cor_research import env as E, protocol as PR
E.configure("cpu")
from cor_research.scoring import track_a as TA, track_b as TB
from cor_research.mathematics import run_track_c as RC
RES = {}
def check(name, cond, detail=""): RES[name] = dict(pass_=bool(cond), detail=str(detail)[:300]); return cond
EARLY = dict(KL=0.03, acc=0.85, nec_E=1.0, nec_M=0.005); MIX = dict(KL=0.012, acc=0.85, nec_E=0.001, nec_M=0.8)
def mix(a, b, t, cls_from=None):
    l = lambda x: np.log(x + 0.01); f = lambda xa, xb: float(np.exp(l(xa) + t * (l(xb) - l(xa))) - 0.01); return dict(KL=0.03, acc=0.85, nec_E=f(a["nec_E"], b["nec_E"]), nec_M=f(a["nec_M"], b["nec_M"]))
C = PR.effective("standard")
def design_A(nparents, nD=4, arms=("fullB", "fullI", "paramI_optB", "lbl", "IfromB_L0")):
    units = [dict(parent=f"nP{9000 + i}", base="E", seed=20) for i in range(nparents)]; Ds = list(range(2001, 2001 + nD))
    return dict(units=units, D=Ds, arms=list(arms), arms_by_base=dict(E=list(arms), M=list(arms)), parents=[u["parent"] for u in units], eligibility={},
                tags={f"{u['parent']}:E{d}:{a}": f"t_{u['parent']}_{d}_{a}" for u in units for d in Ds for a in arms})
def inputs_A(D, arm_fn, lbl_ok=True, early=None):
    M, W = {}, {}
    for u in D["units"]:
        for d in D["D"]:
            unit = f"{u['parent']}:E{d}"
            for st in (150, 300, 1211, 4000):
                M[(f"{unit}:fullB", st)] = dict(EARLY); M[(f"{unit}:fullI", st)] = dict(MIX)
                r = arm_fn(u["parent"], d)
                for a in ("paramI_optB", "lbl", "IfromB_L0"):
                    if r is not None: M[(f"{unit}:{a}", st)] = dict(r) if a != "IfromB_L0" else dict(EARLY)
            if early: M[(f"{unit}:IfromB_L0", 150)] = dict(EARLY, KL=0.5)
            W[f"{unit}:B"] = dict(KL=0.30); W[f"{unit}:I"] = dict(KL=0.50); W[f"{unit}:lbl"] = dict(KL=0.45 if lbl_ok else 0.32)
    return (D, {}, dict(M=M, W=W, TP={}), [])
h = lambda r, k="HA2:E": r["hypotheses"][k]
sc = lambda inp, cfg=C: TA.score("standard", "/x", cfg=cfg, frozen_measure_code=None, inputs=inp)
r = sc(inputs_A(design_A(4), lambda p, d: mix(EARLY, MIX, 0.95))); check("A_confirm", h(r)["verdict"] == "GENERALIZES", h(r)["verdict"])
check("A_conditional_label_needs_HA1", h(r)["verdict"] == "GENERALIZES" and h(r, "HA1:E")["verdict"] == "GENERALIZES")
r = sc(inputs_A(design_A(4), lambda p, d: mix(EARLY, MIX, 0.05))); check("A_falsify", h(r)["verdict"] == "FAILS_TO_GENERALIZE", h(r)["verdict"])
r = sc(inputs_A(design_A(3), lambda p, d: mix(EARLY, MIX, 0.95))); check("A_inconclusive_too_few_parents", h(r)["verdict"] == "INCONCLUSIVE", h(r)["verdict"])
r = sc(inputs_A(design_A(4), lambda p, d: None if d > 2001 else mix(EARLY, MIX, 0.95)))
check("A_missing_inputs_counted_not_dropped", h(r)["verdict"] == "INCONCLUSIVE" and h(r)["pair_status_counts"]["MISSING"] == 12, h(r)["pair_status_counts"])
r = sc(inputs_A(design_A(4), lambda p, d: dict(mix(EARLY, MIX, 0.95), KL=0.20)))
check("A_incompetent_classified", h(r)["pair_status_counts"]["NONCOMPETENT"] == 16 and all(v == "PARENT_NONCOMPETENT" for v in h(r)["parent_verdicts"].values()), h(r)["parent_verdicts"])
C2 = copy.deepcopy(C); C2["BAND"]["I_like"] = (0.90, 9.99); inp = inputs_A(design_A(4), lambda p, d: mix(EARLY, MIX, 0.80))
r1, r2 = sc(inp), sc(inp, C2); ev = lambda r: {k: x["pair_status_counts"]["ANCHORS_NOT_EVALUABLE"] + x["pair_status_counts"]["MISSING"] for k, x in r["hypotheses"].items()}
check("A_band_change_keeps_evaluability", ev(r1) == ev(r2) and h(r1)["verdict"] != h(r2)["verdict"], f"{h(r1)['verdict']} -> {h(r2)['verdict']}")
r = sc(inputs_A(design_A(4), lambda p, d: dict(EARLY), lbl_ok=False)); check("A_lbl_control_invalid", h(r, "HA6b:E")["pair_status_counts"]["CONTROL_INVALID"] == 16, h(r, "HA6b:E")["pair_status_counts"])
r = sc(inputs_A(design_A(4), lambda p, d: dict(EARLY), early=True)); check("A_reverse_transplant_incompatible", h(r, "HA4b:E")["pair_status_counts"]["INCOMPATIBLE"] == 16, h(r, "HA4b:E")["pair_status_counts"])
tmp = tempfile.mkdtemp(); os.environ["COR_RUN_ROOT"] = tmp
from cor_research.jobs import plan as PL, runner as RN
P_ = PR.effective("smoke"); tag = PL.cont_tag("uP160", 21, 2001, "E", "fullB", P_); os.makedirs(os.path.join(tmp, "ft", tag)); ck = os.path.join(tmp, "ft", tag, "step200.npz"); np.savez(ck, a=np.zeros(3))
SP = P_["STREAMS"]["measure_primary"]; rel = PL.mpath("trackA", tag, 200, SP); par = E.sha256(E.patch_parents().parent_path("uP160"))
def put(ck_hash, code): rec = dict(run=tag, parent="uP160", step=200, stream=SP, ckpt_sha256=ck_hash, parent_sha256=par, code_sha256=code, **EARLY); E.atomic_json(rec, os.path.join(tmp, rel)); return E.sha256(os.path.join(tmp, rel))
os.makedirs(RN.REG(tmp), exist_ok=True)
def reg(mh): E.atomic_json(dict(id="cont:uP160:E2001:fullB", status="done", output_hashes={f"ft/{tag}/step200.npz": E.sha256(ck)}), RN.rec_path(tmp, "cont:uP160:E2001:fullB")); E.atomic_json(dict(id="meas:uP160:E2001:fullB@200", status="done", output_hashes={rel: mh}), RN.rec_path(tmp, "meas:uP160:E2001:fullB@200"))
reg(put("0" * 64, {"x": "y"})); r = TA.score_all("smoke", tmp, frozen_measure_code={"x": "y"}); check("A_wrong_checkpoint_hash_refused", r.get("verdict") == "PROVENANCE_FAIL" and any("checkpoint hash" in p for p in r["problems"]), r.get("problems"))
reg(put(E.sha256(ck), {"x": "y"})); r = TA.score_all("smoke", tmp, frozen_measure_code={"x": "DIFFERENT"}); check("A_wrong_code_hash_refused", r.get("verdict") == "PROVENANCE_FAIL" and any("code hash" in p for p in r["problems"]), r.get("problems"))
r = TA.score_all("smoke", tmp, frozen_measure_code={"x": "y"}); check("A_consistent_inputs_accepted", r.get("verdict") != "PROVENANCE_FAIL", r.get("problems"))
# ---------------- Track B (synthetic decision vectors through the real scorer)
rootB = tempfile.mkdtemp(); os.makedirs(os.path.join(rootB, "trackB"))
def unitsB(nparents, f_dir=0.0, f_rand=0.0, oE=0.3, oM=0.3, klE=0.03, klM=0.03, noise=0.01, par_sd=0.0, shared_D=0.0, compE=True, nD=4, seed=0):
    rng = np.random.default_rng(seed); n = 4000; U = []; rules = {}
    base = rng.integers(0, 3, n); rules["bayes"] = base
    for k in TB.RULES: rr = base.copy(); m = rng.random(n) < 0.3; rr[m] = (rr[m] + 1) % 3; rules[k] = rr
    S = rules["local"] != base
    for i in range(nparents):
        p = f"nP{9100 + i}"; fd = f_dir + par_sd * rng.standard_normal(); Dn = {d: rng.random(n) < shared_D for d in range(2001, 2001 + nD)}
        mdir = S & (rng.random(n) < max(fd, 0)); mrand = rng.random(n) < f_rand          # a FIXED (systematic) class difference per parent
        for d in range(2001, 2001 + nD):
            for arm, cls_ in (("fullB", "EARLY"), ("fullI", "MIXTURE")):
                dec = base.copy(); fl = (rng.random(n) < noise) | Dn[d]; dec[fl] = (dec[fl] + 2) % 3
                if cls_ == "MIXTURE":
                    dec[mdir] = rules["local"][mdir]; dec[mrand] = (base[mrand] + 2) % 3   # systematic, but toward no candidate rule (rules differ from Bayes only by +1)
                tg = f"tb_{p}_{d}_{arm}"; nec = dict(EARLY if cls_ == "EARLY" else MIX); nec["KL"] = 0.2 if (cls_ == "EARLY" and not compE) else nec["KL"]; nec["ckpt_sha256"] = "h"
                np.savez(os.path.join(rootB, "trackB", tg + "_matched.npz"), dec=dec.astype(np.int8), **{f"rule_{k}": v for k, v in rules.items()})
                np.savez(os.path.join(rootB, "trackB", tg + "_final.npz"), dec=dec.astype(np.int8), **{f"rule_{k}": v for k, v in rules.items()})
                ic = {f"{k}_{l}": dict(IIA_norm=((oE if cls_ == "EARLY" else oM) if (k, l) == ("OQI", 2) else 0.5)) for k in ("OQI", "OQI_unmatched", "RS") for l in range(4)}; ic["null_base_patch_max_logit_diff"] = 0.0
                fn = dict(ckpt_sha256="h", natural=dict(KL=(klE if cls_ == "EARLY" else klM) + 0.0005 * rng.standard_normal(), rules={k: dict(w=0.05) for k in TB.RULES}, nulls=dict(parametric=dict(rules={k: dict(w=0.04) for k in TB.RULES}))), interchange=ic)
                an = {lev: dict(step=1000, necessity=nec, functional=fn, decisions_sha256=E.sha256(os.path.join(rootB, "trackB", f"{tg}_{lev}.npz"))) for lev in ("matched", "final")}
                U.append(dict(key=f"{p}:E{d}:{arm}", parent=p, base="E", D=d, arm=arm, tag=tg, status="PRESENT", rec=dict(analyses=an)))
    return ({}, U, [])
def tb(**kw):
    r = TB.score("standard", rootB, frozen_code=None, inputs=unitsB(**kw)); lv = r["levels"]["matched"]; lv["classification"] = r["verdict"]; return lv
lv = tb(nparents=6, f_dir=0.3); check("B_E_only_functional_difference_is_history_confounded", lv["classification"].startswith("FUNCTIONAL_DIFFERENCE (class and intervention history"), lv["classification"])
lv = tb(nparents=6, f_rand=0.10); check("B_functional_difference_flagged_E_only", lv["classification"].startswith("FUNCTIONAL_DIFFERENCE"), lv["classification"])
lv = tb(nparents=6, oE=0.5, oM=0.2); check("B_B2_locus", lv["classification"].startswith("B2-LOCUS") and "+B3" not in lv["classification"], lv["classification"])
lv = tb(nparents=6, klE=0.040, klM=0.020); check("B_B3_precision", lv["classification"].startswith("B3"), lv["classification"])
lv = tb(nparents=4, par_sd=0.15, oE=0.5, oM=0.2); check("B_underpowered_is_NON_IDENTIFIABLE_not_B2", lv["classification"].startswith("NON_IDENTIFIABLE (underpowered)"), lv["classification"])
lv = tb(nparents=6, shared_D=0.05); check("B_shared_D_null_no_difference", (not lv["tests"]["HB_c"]["FUNCTIONAL_DIFF"]) and abs(lv["tests"]["HB_c"]["primary"]["mean"]) < 0.01, lv["tests"]["HB_c"]["primary"])
lv = tb(nparents=3, f_dir=0.3); check("B_insufficient_units", lv["classification"].startswith("INSUFFICIENT_UNITS"), lv["classification"])
lv = tb(nparents=6, f_dir=0.3, compE=False); check("B_incompetent_units_not_evaluable", lv["classification"].startswith("INSUFFICIENT_UNITS") and lv["n_evaluable"] == 24, (lv["classification"], lv["n_evaluable"]))
D_, U_, _ = unitsB(6, f_dir=0.3); U_[0] = dict(key=U_[0]["key"], parent=U_[0]["parent"], base="E", D=2001, arm="fullB", status="MISSING")
rr = TB.score("standard", rootB, frozen_code=None, inputs=(D_, U_, [])); check("B_missing_input_listed", rr["units"][U_[0]["key"]] == "MISSING")
tmpB = tempfile.mkdtemp(); os.environ["COR_RUN_ROOT"] = tmpB; tagb = PL.cont_tag("uP160", 21, 2001, "E", "fullB", P_); os.makedirs(os.path.join(tmpB, "trackB"))
E.atomic_json(dict(run=tagb, analyses=dict(final=dict(step=200, necessity=dict(EARLY, ckpt_sha256="bad"), functional=dict(ckpt_sha256="bad"))), code_sha256={"x": "y"}), os.path.join(tmpB, "trackB", tagb + ".json"))
os.makedirs(RN.REG(tmpB)); E.atomic_json(dict(id="fbbat:uP160:E2001:fullB", status="done", output_hashes={f"trackB/{tagb}.json": E.sha256(os.path.join(tmpB, "trackB", tagb + ".json"))}), RN.rec_path(tmpB, "fbbat:uP160:E2001:fullB"))
E.atomic_json(dict(id="cont:uP160:E2001:fullB", status="done", snapshot_hashes={"step200.npz": "good"}), RN.rec_path(tmpB, "cont:uP160:E2001:fullB"))
rr = TB.score("smoke", tmpB, frozen_code={"x": "y"}); check("B_wrong_hash_refused_via_loader", rr.get("verdict") == "PROVENANCE_FAIL" and any("checkpoint hash" in p for p in rr["problems"]), rr.get("problems"))
# ---------------- Track C
base = json.load(open(os.path.join(PKG, "tests", "fixtures", "track_c_math_reference.json")))
c = RC.classify(base); check("C_reference_classification", c["repaired_OD_gate"] == "C2" and c["neural_prediction"].startswith("C5") and c["original_OD"]["source_level_instantiation"] == "C4", c)
b2 = copy.deepcopy(base)
for x in b2["repaired_gate"].values(): x["M2_deleted_h_destination_equals_bayes_without_factor"]["h1"] = 0.3
check("C_planted_distinctive_prediction_detected", not RC.classify(b2)["neural_prediction"].startswith("C5"))
b3 = copy.deepcopy(base)
for x in b3["repaired_gate"].values(): x["LP_max_margin_by_order"]["1"]["representable"] = True
check("C_planted_original_unresolved", RC.classify(b3)["original_OD"]["source_level_instantiation"] == "UNRESOLVED")
# ---------------- eligibility immutability, runner rules, integrity check 14
tmpE = tempfile.mkdtemp(); os.environ["COR_RUN_ROOT"] = tmpE; E.atomic_json({"uP161": dict(status="SCREENED", units={})}, os.path.join(tmpE, "eligibility.json"))
try: PL.eligibility("standard", tmpE); ok_e = True
except RuntimeError: ok_e = False
check("eligibility_unchanged_when_inputs_absent", ok_e)
E.atomic_json({"uP162": dict(status="SCREENED", units={})}, os.path.join(tmpE, "eligibility.json"))
try: PL.eligibility("standard", tmpE); ok_e = False
except RuntimeError: ok_e = True
check("eligibility_change_raises", ok_e)
tmpR = tempfile.mkdtemp(); Rn = RN.Runner(tmpR, 1, log=lambda *a: None); j = PL.job("t:fail", "A", "window", 1, [sys.executable, "-c", "import sys; sys.exit(5)"], {}, ["x/y.npz"])
Rn.run(lambda: [j], require_freeze=False); Rn.run(lambda: [j], require_freeze=False); rec = RN.load_all(tmpR)["t:fail"]
check("runner_failed_job_kept_not_replaced", rec["status"] == "failed" and len(rec["attempts"]) == 1)
try: Rn.register([dict(j, env={"X": "1"})]); ok_d = False
except RuntimeError: ok_d = True
check("runner_definition_change_refused", ok_d)
os.makedirs(os.path.join(tmpR, "ft", "stray")); open(os.path.join(tmpR, "ft", "stray", "resume.pkl"), "w").write("x")
Rn.register([PL.job("t:stray", "A", "continue", 1, [sys.executable, "-c", "0"], {}, ["ft/stray/final.json"])]); check("runner_unregistered_outputs_blocked", RN.load_all(tmpR)["t:stray"]["status"] == "blocked")
from cor_research.verification import integrity as IN
tmpI = tempfile.mkdtemp(); os.environ["COR_RUN_ROOT"] = tmpI; os.makedirs(os.path.join(tmpI, "ft", "r")); f = os.path.join(tmpI, "ft", "r", "step1.npz"); np.savez(f, a=np.zeros(2))
os.makedirs(RN.REG(tmpI)); E.atomic_json(dict(id="j", status="done", phase="continue", parent="nP1", seed=1, data_order=2001, arm="fullB", attempts=[], output_hashes={"ft/r/step1.npz": E.sha256(f)}), RN.rec_path(tmpI, "j")); np.savez(f, a=np.ones(2))
check("integrity_14_detects_corrupted_checkpoint", IN.run("standard", tmpI)["checks"]["14_checkpoint_integrity"]["status"] == "FAIL")
fails = [k for k, v in RES.items() if not v["pass_"]]
for k, v in RES.items(): print(f"{'PASS' if v['pass_'] else 'FAIL'}  {k}  {'' if v['pass_'] else v['detail']}")
json.dump(RES, open(os.path.join(os.environ.get("COR_TEST_OUT") or tempfile.mkdtemp(), "planted_results.json"), "w"), indent=1, default=str)   # never into the package
sys.exit(1 if fails else 0)
