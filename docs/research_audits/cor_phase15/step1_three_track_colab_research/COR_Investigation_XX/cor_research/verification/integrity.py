"""The 20 automated scientific-integrity checks (mandate §12), v2 after review C (M9: no vacuous passes; NOT_YET when nothing exists to
check).  python -m cor_research.verification.integrity PRESET [OUT.json]
Each check returns PASS / FAIL / NOT_YET with details.  Run before confirmation (package checks 01-03, 05-07, 11, 13) and after (all).
The report generator prints every FAIL at the top of every RESULTS file."""
import os, sys, json, glob, pickle, importlib.metadata as IM, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E, protocol as PR
import numpy as np


def R_(ok, **d): return dict(status="PASS" if ok else "FAIL", **d)
def NY(**d): return dict(status="NOT_YET", **d)


def run(preset, root=None):
    from cor_research.jobs import runner as RN, plan as PL
    root = root or E.RUN_ROOT; C = PR.effective(preset); out = {}; REG = RN.load_all(root); pkg = E.PKG
    man = E.read_json(os.path.join(pkg, "manifests", "PACKAGE_MANIFEST.json"))
    if man:
        bad = [f for f, h in man["files"].items() if not os.path.exists(os.path.join(pkg, f)) or E.sha256(os.path.join(pkg, f)) != h]
        nb = [f for f in bad if f.startswith("notebooks/")]; bad = [f for f in bad if not f.startswith("notebooks/")]
        out["01_code_version"] = R_(not bad, source_commit=man.get("source_commit"), changed_or_missing=bad[:20], notebooks_edited=nb)
    else: out["01_code_version"] = R_(False, reason="manifests/PACKAGE_MANIFEST.json missing (package not built by scripts/build_package.py)")
    pins = {}
    for line in open(os.path.join(pkg, "requirements.txt")):
        line = line.split("#")[0].strip()
        if "==" in line: n, v = line.split("=="); pins[n.split("[")[0]] = v
    got = {}
    for n in pins:
        try: got[n] = IM.version(n)
        except Exception: got[n] = None
    out["02_dependencies"] = R_(all(got[n] == v for n, v in pins.items()) and sys.version_info >= (3, 12), pinned=pins, installed=got, python=sys.version.split()[0])
    ck = E.read_json(os.path.join(pkg, "manifests", "CHECKPOINTS.json"), {}) or {}; bad = [f for f, h in ck.items() if not os.path.exists(os.path.join(pkg, f)) or E.sha256(os.path.join(pkg, f)) != h]
    fresh = []
    for p in C["PRESETS"][preset]["parents"]:
        if p.startswith("n"):
            d = os.path.join(E.PARENT_ROOT, "z" + p[1:]); L = E.read_json(os.path.join(d, "lineage.json"))
            if L: fresh.append((p, L["checkpoint"] == E.sha256(os.path.join(d, "pre.npz"))))
    out["03_checkpoint_provenance"] = (R_(not bad and all(ok for _, ok in fresh), vendored_bad=bad, n_vendored=len(ck), fresh_parents=fresh) if ck else R_(False, reason="manifests/CHECKPOINTS.json missing or empty"))
    conf = [r for r in REG.values() if r["phase"] in ("window", "continue")]
    if conf:
        Ds = {r.get("data_order") for r in conf}; dev_D = set(range(801, 830)) | {C["SCREEN"]["D"], C["SCREEN"]["D2"]}; dup = {}
        for r in REG.values():
            if r["phase"] == "continue": dup.setdefault((r.get("parent"), r.get("seed"), r.get("data_order"), r.get("arm"), r.get("base")), []).append(r["id"])
        streams = list(C["STREAMS"].values()) + [C["SCREEN"]["stream"]] + [C["TRACK_B"]["sets"][k] for k in ("nat_stream", "disc_stream", "pair_stream")] + [C["TRACK_B"]["matching"]["selection_stream"]]
        pilot_mix = bool({r.get("parent") for r in conf} & set(C["PILOT_PARENTS"])) and preset not in ("pilot",)
        out["04_seed_independence"] = R_(not (Ds & dev_D) and all(len(v) == 1 for v in dup.values()) and len(set(streams)) == len(streams) and not pilot_mix,
                                         confirmation_D=sorted(x for x in Ds if x), overlap_with_dev_or_screen=sorted(Ds & dev_D), pilot_material_in_run=pilot_mix)
    else: out["04_seed_independence"] = NY(reason="no confirmation jobs registered")
    ok, bad = RN.verify_freeze(pkg); out["05_frozen_protocol_hashes"] = R_(ok, mismatched=bad[:20])
    ga = E.read_json(os.path.join(root, "preflight", "graph_audit.json"))
    if ga:
        out["06_intervention_tensor_locations"] = R_(all(v for k, v in ga["checks"].items() if "zero_grad" in k or "_T" in k), zero_gradient_tensors=ga.get("zero_gradient_tensors"))
        out["07_forward_backward_pathway_effects"] = R_(ga["all_pass"], checks=ga["checks"])
    else: out["06_intervention_tensor_locations"] = out["07_forward_backward_pathway_effects"] = NY(reason="run the preflight graph audit (notebook 00, step 6)")
    tps = sorted(glob.glob(os.path.join(root, "ft", "tp_*", "provenance.json"))); probs = []; optp = []
    for f in tps:
        pv = json.load(open(f)); d = os.path.dirname(f); arm = os.path.basename(d).split("_", 3)[-1]
        for k, h in pv["outputs"].items():
            if E.sha256(os.path.join(d, k)) != h: probs.append(f"{f}: output {k} changed")
        for k, h in pv["inputs"].items():
            lab, fn = k.split("/", 1); src = os.path.join(root, "ft", pv[lab], fn)
            if not os.path.exists(src) or E.sha256(src) != h: probs.append(f"{f}: input {k} differs from the window state")
        if arm.endswith("fullB") and pv["param_dist_to_B"] != 0: probs.append(f"{f}: fullB differs from B")
        if arm.endswith("fullI") and pv["param_dist_to_I"] != 0: probs.append(f"{f}: fullI differs from I")
        if arm.endswith("fullB"):
            a, b = pickle.load(open(os.path.join(root, "ft", pv["B"], "opt_step100.pkl"), "rb")), pickle.load(open(os.path.join(d, "opt_step100.pkl"), "rb"))
            optp.append(bool(a["rng"] == b["rng"] and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a["opt"], b["opt"]))))
    out["08_transplant_composition"] = R_(not probs, n=len(tps), problems=probs[:20]) if tps else NY()
    out["09_optimizer_state_preservation"] = R_(all(optp), n_fullB_checked=len(optp)) if optp else NY()
    ctl = []
    for f in [f for f in tps if f.split(os.sep)[-2].endswith("randnorm")]:
        pv = json.load(open(f)); ctl.append(bool(pv["perturb_seed"] is not None and pv["perturbed_groups"]))
    lbl = [(r["id"], E.read_json(os.path.join(root, os.path.dirname(r["outputs"][0]), "run_config.json"))) for r in REG.values() if r["phase"] == "window" and r.get("arm") == "window_lbl" and r["status"] == "done"]
    ctl += [bool(c and c["env"].get("FX_LBL_P") == "0.5") for _, c in lbl]
    out["10_evidence_preserving_controls"] = R_(all(ctl), n=len(ctl)) if ctl else NY()
    tc = E.read_json(os.path.join(root, "track_c", "track_c_math.json"))
    out["11_functional_decision_references"] = R_(tc["t4_reference"]["closed_vs_brute_max_abs"] < 1e-12 and all(c["R1_max_err_signed_identity"] < 1e-9 for c in tc["repaired_gate"].values())
                                                  and all(max(c["M2_deleted_h_destination_equals_bayes_without_factor"].values()) < 1e-9 for c in tc["repaired_gate"].values()), t4=tc["t4_reference"]) if tc else NY()
    nd = [r["id"] for r in REG.values() if r["status"] != "done"]; unreg = []
    try: unreg = [j["id"] for j in PL.plan(preset, root) if j["id"] not in REG]
    except Exception as e: unreg = [f"planner error: {e}"]
    out["12_run_completion"] = (R_(not nd and not unreg, n_jobs=len(REG), not_done=nd[:50], n_not_done=len(nd), planned_but_unregistered=unreg[:20]) if REG else NY(reason="no jobs registered"))
    fzp = os.path.join(pkg, "protocols", "frozen_hypotheses", "track_a_hypotheses.json"); fzj = E.read_json(fzp) or {}
    out["13_competence_thresholds"] = R_(fzj.get("competence") == C["COMP"] and fzj.get("causal_class") == C["CLASS"], frozen_file=dict(competence=fzj.get("competence"), causal_class=fzj.get("causal_class")), in_use=dict(comp=C["COMP"], cls=C["CLASS"]))
    hashed = [(r["id"], o, h) for r in REG.values() if r["status"] == "done" for o, h in r.get("output_hashes", {}).items()]
    badh = [f"{i}:{o}" for i, o, h in hashed if not os.path.exists(os.path.join(root, o)) or E.sha256(os.path.join(root, o)) != h]
    out["14_checkpoint_integrity"] = R_(not badh, n_checked=len(hashed), bad=badh[:20]) if hashed else NY()
    res = [(os.path.basename(d), E.read_json(os.path.join(d, "resume_log.json"))) for d in glob.glob(os.path.join(root, "ft", "*")) if os.path.exists(os.path.join(d, "resume_log.json"))]
    rt = E.read_json(os.path.join(root, "preflight", "RESUME_TEST.json"))
    out["15_resumed_training_consistency"] = (R_(bool(rt.get("resume_bitwise_identical")), runtime_resume_test=rt, resumed_runs=res[:50]) if rt else NY(reason="runtime resume test not run (preflight)", resumed_runs=res[:50]))
    A = E.read_json(os.path.join(root, "scores", f"track_a_{preset}.json"))
    if A and "hypotheses" in A:
        from cor_research.scoring import track_a as TA
        again = TA.score_all(preset, root); out["16_frozen_score_reproduction"] = R_(json.dumps(again, sort_keys=True, default=str) == json.dumps(A, sort_keys=True, default=str))
        parents = {u["parent"] for u in A.get("units", [])}; hs = {}
        for p in parents: hs.setdefault(E.sha256(E.patch_parents().parent_path(p)), []).append(p)
        out["17_statistical_unit_independence"] = R_(all(len(v) == 1 for v in hs.values()) and all((h.get("phi") or {}).get("n_parents", 0) <= len(parents) for h in A["hypotheses"].values()), parents=sorted(parents))
        D = PL.design(preset, root); exp = {b: len([u for u in D["units"] if u["base"] == b]) * len(D["D"]) for b in "EM"}
        miss = {k: (sum(h["pair_status_counts"].values()), exp[k.split(":")[1]]) for k, h in A["hypotheses"].items() if sum(h["pair_status_counts"].values()) != exp[k.split(":")[1]]}
        cont = [r for r in REG.values() if r["phase"] == "continue"]; listed = {(p["parent"], p["D"]) for v in A["pairs"].values() for p in v}
        dropped = [r["id"] for r in cont if (r.get("parent"), r.get("data_order")) not in listed]
        out["18_negative_and_censored_inclusion"] = R_(not miss and not dropped, expected_pairs_per_base=exp, mismatches=miss, registered_pairs_absent_from_scores=dropped[:20])
    else: out["16_frozen_score_reproduction"] = out["17_statistical_unit_independence"] = out["18_negative_and_censored_inclusion"] = NY(reason="Track A not scored yet")
    fv = os.path.join(root, "RESULTS", "FROZEN_PREDICTION_VERDICTS.md")
    if A and os.path.exists(fv) and "hypotheses" in A:
        txt = open(fv).read(); out["19_final_report_consistency"] = R_(all(f"| {k} | A | {h['verdict']} |" in txt for k, h in A["hypotheses"].items()))
    else: out["19_final_report_consistency"] = NY()
    mf = E.read_json(os.path.join(root, "RESULTS", "COMPLETE_EVIDENCE_MANIFEST.json"))
    if mf:
        bad = [f for f, h in mf["evidence"].items() if not os.path.exists(os.path.join(root, f)) or E.sha256(os.path.join(root, f)) != h]
        out["20_archive_and_output_hashes"] = R_(not bad, n=len(mf["evidence"]), bad=bad[:20])
    else: out["20_archive_and_output_hashes"] = NY()
    return dict(preset=preset, checks=out, n_pass=sum(v["status"] == "PASS" for v in out.values()), n_fail=sum(v["status"] == "FAIL" for v in out.values()), n_not_yet=sum(v["status"] == "NOT_YET" for v in out.values()))


if __name__ == "__main__":
    r = run(sys.argv[1]); dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(E.RUN_ROOT, "INTEGRITY_CHECKS.json"); E.atomic_json(r, dst)
    for k, v in r["checks"].items(): print(f"{k:42s} {v['status']}")
    print(f"PASS {r['n_pass']}  FAIL {r['n_fail']}  NOT_YET {r['n_not_yet']}")
