"""RESULTS/ generator (mandate §11).  python -m cor_research.reporting.report PRESET [OUT_DIR]
Every statement in the generated files is computed from the run root (registry, eligibility, measurement records, scorer outputs,
Track C output, review files); no scientific conclusion is written in advance.  Labels used:
  [PROVEN] mathematical result with proof   [VERIFIED-NUM] exact numerical check   [DEV] development observation
  [FROZEN-CONFIRMED] / [FROZEN-FALSIFIED] frozen prediction outcome   [EXPLORATORY]   [INCONCLUSIVE]   [TECHNICAL-ONLY] smoke/pilot
If any registered job is not 'done', every file states INCOMPLETE and the synthesis refuses to claim completion."""
import os, sys, json, time, glob, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E, protocol as PR

FILES = ["MAIN_RESEARCH_SYNTHESIS.md", "TRACK_A_CAUSAL_GENERALIZATION.md", "TRACK_B_FUNCTIONAL_IDENTIFICATION.md", "TRACK_C_OD_MATHEMATICAL_STATUS.md",
         "FROZEN_PREDICTION_VERDICTS.md", "STATISTICAL_ANALYSIS.md", "NEGATIVE_AND_INCONCLUSIVE_RESULTS.md", "METHODS_AND_REPRODUCIBILITY.md",
         "COMPUTE_AND_EXECUTION_LOG.md", "REVIEW_AND_ADJUDICATION.md", "COMPLETE_EVIDENCE_MANIFEST.json"]
LAB = {"GENERALIZES": "[FROZEN-CONFIRMED]", "GENERALIZES_CONDITIONAL": "[FROZEN-CONFIRMED, conditional on an unestablished switch]", "NOT_ESTABLISHED": "[INCONCLUSIVE: not established]", "FAILS_TO_GENERALIZE": "[FROZEN-FALSIFIED]", "PARTIAL": "[INCONCLUSIVE: parent-specific exceptions]",
       "INCONCLUSIVE": "[INCONCLUSIVE]", "TECHNICAL_ONLY": "[TECHNICAL-ONLY]"}


def fmt(x, nd=3):
    if x is None: return "n/a"
    if isinstance(x, float): return f"{x:.{nd}f}"
    if isinstance(x, list) and len(x) == 2 and all(isinstance(v, float) for v in x): return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"
    return str(x)


def generate(preset, root=None, out=None):
    from cor_research.jobs import runner as RN
    from cor_research.scoring import track_a as TA, track_b as TB
    root = root or E.RUN_ROOT; out = out or os.path.join(root, "RESULTS"); os.makedirs(out, exist_ok=True); C = PR.effective(preset)
    from cor_research.jobs import plan as PLN
    R = RN.load_all(root); S = RN.summary(root); incomplete = [r["id"] for r in R.values() if r["status"] != "done"]
    try: unreg = [j["id"] for j in PLN.plan(preset, root) if j["id"] not in R]          # review C M6: jobs the plan still needs but that are not registered
    except Exception as e: unreg = [f"planner error: {e}"]
    tracks_present = {r["track"] for r in R.values()}; complete = bool(R) and not incomplete and not unreg and {"A", "B", "C"} <= tracks_present
    IC = E.read_json(os.path.join(out, "INTEGRITY_CHECKS.json")) or {}; ic_fail = [k for k, v in (IC.get("checks") or {}).items() if v["status"] == "FAIL"]
    RI = E.read_json(os.path.join(root, "RUN_IDENTITY.json")) or {}; mixed = bool(RI.get("mixed_identity")); id_changes = RI.get("changes", [])   # review round 3 X3
    sc = os.path.join(root, "scores"); A = E.read_json(os.path.join(sc, f"track_a_{preset}.json")); B = E.read_json(os.path.join(sc, f"track_b_{preset}.json"))
    A1211 = E.read_json(os.path.join(sc, f"track_a_{preset}_1211.json")); Cm = E.read_json(os.path.join(root, "track_c", "track_c_math.json"))
    EL = E.read_json(os.path.join(root, "eligibility.json"), {}) or {}; tech = C["TECHNICAL_ONLY"]
    banner = (f"> Preset **{preset}** — {C['PRESETS'][preset]['label']}\n> Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from run root `{os.path.basename(root)}`.\n"
              + (f"> **INCOMPLETE: {len(incomplete)} registered jobs not done, {len(unreg)} planned jobs not yet registered, tracks present {sorted(tracks_present)}. No completion claim is made; every verdict below is PROVISIONAL.**\n" if not complete else "> All planned jobs of all three tracks are done.\n")
              + (f"> **INTEGRITY FAILURES: {', '.join(ic_fail)} (RESULTS/INTEGRITY_CHECKS.json).**\n" if ic_fail else "")
              + (f"> **MIXED-IDENTITY RUN: the numerical environment changed {len(id_changes)} time(s) during this run (RUN_IDENTITY.json: {json.dumps([c['diff'] for c in id_changes])[:600]}). Every verdict below is labelled MIXED-IDENTITY; per-attempt identities are in the registry.**\n" if mixed else "")
              + ("> **TECHNICAL-ONLY preset: nothing in this report is scientific evidence.**\n" if tech else "")
              + "> Phase 1.5, Step 1 (≈ 25% provisional). Conclusions remain provisional until the post-run review (RESULTS/REVIEW_AND_ADJUDICATION.md) is performed.\n\n")
    W = lambda name, txt: open(os.path.join(out, name), "w").write(txt)
    # ---- Track A
    ta = ["# Track A — causal generalisation across independent pretrained parents\n", banner, "## Parent denominator (all preregistered parents of this preset)\n",
          "| parent | status | E unit (seed) | M unit (seed) | screening (seed: class, KL, acc) |", "|---|---|---|---|---|"]
    for p in C["PRESETS"][preset]["parents"]:
        e = EL.get(p)
        if not e: ta.append(f"| {p} | NOT YET SCREENED / pretraining incomplete | | | |"); continue
        u = e.get("units", {}); scr = "; ".join(f"{s}: {v['cls']}, {v['KL']:.3f}, {v['acc']:.3f}{'' if v['competent'] else ' (noncompetent)'}" for s, v in (e.get("screening") or {}).items())
        ta.append(f"| {p} | {e['status']}{' aux_acc=' + fmt(e.get('aux_acc')) if e.get('aux_acc') is not None else ''} | {u.get('E', {}).get('seed', '—') if u.get('E', {}).get('eligible') else 'not eligible'} | "
                  f"{u.get('M', {}).get('seed', '—') if u.get('M', {}).get('eligible') else 'not eligible'} | {scr} |")
    if A and A.get("verdict") == "PROVENANCE_FAIL": ta += ["\n## PROVENANCE FAIL — no verdicts\n"] + [f"- {x}" for x in A["problems"]]
    elif A:
        ta += [f"\n## Frozen hypotheses at step {A['step']} (primary)\n", "| hypothesis | base | verdict | evaluable parents | parent verdicts | pair status counts | phi (parent-level mean, 95% CI) |", "|---|---|---|---|---|---|---|"]
        for k, h in A["hypotheses"].items():
            ph = h.get("phi") or {}; ta.append(f"| {k.split(':')[0]} | {k.split(':')[1]} | {LAB.get(h['verdict'], '')} {h['verdict']} | {h['n_parents_evaluable']} | "
                                               f"{', '.join(f'{p}: {v}' for p, v in h['parent_verdicts'].items())} | {', '.join(f'{s}={n}' for s, n in h['pair_status_counts'].items() if n)} | {fmt(ph.get('mean'))} {fmt(ph.get('ci95'))} |")
        ta += ["\n## Switching probability and competence (parent = unit)\n"]
        for base, st in A["stats"].items():
            sp = st.get("switch_probability") or {}; ta.append(f"- base {base}: switching probability {fmt(sp.get('mean'))} {fmt(sp.get('ci95'))} over {sp.get('n_parents', 0)} parents; per parent {sp.get('per_parent')}")
        if A1211: ta += ["\n## Secondary: historical horizon 1211 [EXPLORATORY: secondary analysis]\n"] + [f"- {k}: {h['verdict']}" for k, h in A1211["hypotheses"].items()]
    else: ta.append("\nTrack A scorer output not found (scorer not yet run).")
    W(FILES[1], "\n".join(ta) + "\n")
    # ---- Track B
    tb = ["# Track B — are EARLY and MIXTURE different algorithms?\n", banner, f"Frozen prediction (from development, made before confirmation): {C['TRACK_B']['frozen_prediction']}\n"]
    if B and B.get("verdict") == "PROVENANCE_FAIL": tb += ["## PROVENANCE FAIL — no classification"] + [f"- {x}" for x in B["problems"]]
    elif B:
        tb.append(f"**Headline (matched and final levels must agree): {B['verdict']}** (rule: {C['TRACK_B']['headline_rule']})\n")
        for lev, L in B["levels"].items():
            t = L["tests"]; c_ = t["HB_c"]["primary"]; d_ = t["HB_d"]["stat"]; e_ = t["HB_e"]["stat"]
            tb += [f"## Level: {lev}{' (precision-matched per parent)' if lev == 'matched' else ' (step 4000, unmatched)'}\n", f"- classification: **{L['classification']}**",
                   f"- units {L['n_units']}, evaluable {L['n_evaluable']}, parents with both classes {len(L['parents_with_both_classes'])}; censoring by class {L['censoring_by_class']} (gap {fmt(L['censoring_gap'])})",
                   f"- matched steps by class: {json.dumps(L['matched_step_by_class'])}",
                   f"- HB_c (primary, different data orders in both terms): {fmt(c_['mean'], 4)} CI {fmt(c_['ci'], 4)}, t p {fmt(c_.get('t_p'), 4)}, sign-flip p {fmt(c_.get('sign_flip_p'), 4)}; FUNCTIONAL_DIFF={t['HB_c']['FUNCTIONAL_DIFF']}, EQUIVALENT={t['HB_c']['EQUIVALENT']}; same-D 2x2 secondary {fmt(t['HB_c']['secondary_same_D_2x2']['mean'], 4)}",
                   "- HB_c_dir (rule-directed, Holm): " + "; ".join(f"{k}: {fmt(v['mean'], 4)} CI {fmt(v['ci'], 4)} holm={t['HB_c_dir']['holm_reject'][k]} native/induced same sign={t['HB_c_dir']['native_vs_induced'][k]['same_sign']}" for k, v in t["HB_c_dir"]["per_rule"].items()) + f"; DIRECTED_DIFF={t['HB_c_dir']['DIRECTED_DIFF']}, EQUIVALENT={t['HB_c_dir']['EQUIVALENT']}",
                   f"- HB_d locus (descriptive; OQI_2 normalised, E - M): {fmt(d_['mean'])} CI {fmt(d_['ci'])}; LOCUS_DIFF={t['HB_d']['LOCUS_DIFF']}; base-run null max logit diff {fmt(t['HB_d']['null_base_patch_max_logit_diff'], 6)}",
                   f"- HB_e precision (KL, E - M): {fmt(e_['mean'], 4)} CI {fmt(e_['ci'], 4)}; PRECISION_DIFF={t['HB_e']['PRECISION_DIFF']} (expected positive; not a discovery)",
                   "- HB_a [descriptive only] rule-mixture weight differences: " + "; ".join(f"{k}: {fmt(v['mean'], 4)}" for k, v in t["HB_a_descriptive"].items()),
                   f"- HB_b [descriptive only] fraction of models above the parametric-precision null: {fmt(t['HB_b_descriptive']['fraction_models_above_parametric_null'])}\n"]
        tb.append("An activation interchange that succeeds is not taken as algorithm identity; HB_d is a locus profile, not a test of what is computed; head-necessity differences are not taken as algorithm differences (mandate B4). B1-FUNCTIONAL means different decision functions with the internal mechanism NOT identified by this design.")
    else: tb.append("Track B scorer output not found.")
    W(FILES[2], "\n".join(tb) + "\n")
    # ---- Track C
    tc = ["# Track C — O/D mathematical status (independent of all training)\n", banner]
    if Cm:
        c = Cm["classification"]; o = c["original_OD"]
        tc += [f"- original O/D, source-level instantiation: **{o['source_level_instantiation']}** [VERIFIED-NUM: exact order-1 LP/Farkas certificate on GATE]; germ-level definition: **{o['germ_level_definition']}**",
               f"- theorem checks (Phase-1 statements, numerical): {json.dumps({k: v['holds'] for k, v in Cm['original']['theorems'].items()})}",
               f"- repaired O/D on GATE: **{c['repaired_OD_gate']}** [PROVEN + VERIFIED-NUM: R1 exact; factor-deletion destinations equal the independent marginalised-joint Bayes reference]",
               *[f"- {t}" for t in c["restricted_theorems"]], f"- neural prediction: **{c['neural_prediction']}** — no O/D neural experiment is run (mandate C5). Candidates considered:",
               *[f"  - {k}: {v}" for k, v in c.get("candidates_considered", {}).items()], f"- reasons: {json.dumps(c['reasons'])}",
               f"- T4 reference: closed form vs brute force max |diff| = {Cm['t4_reference']['closed_vs_brute_max_abs']:.2e}"]
    else: tc.append("Track C output not found (run notebook 03 or `python -m cor_research.mathematics.run_track_c`).")
    W(FILES[3], "\n".join(tc) + "\n")
    # ---- frozen verdicts, stats, negatives
    fv = ["# Frozen prediction verdicts\n", banner, "| id | track | verdict |", "|---|---|---|"]
    if A and "hypotheses" in A: fv += [f"| {k} | A | {h['verdict']} |" for k, h in A["hypotheses"].items()]
    if B and "levels" in B: fv.append(f"| Track B headline | B | {B['verdict']} |")
    if A and "fresh_only" in A: fv += [f"| {k} (fresh parents only) | A | {v['verdict']} |" for k, v in A["fresh_only"].items()]
    if mixed: fv = fv[:3] + [r_.replace(" |\n", " |") for r_ in fv[3:]] + ["", "**All verdicts in this table are MIXED-IDENTITY (see the banner).**"]
    if Cm: fv.append(f"| Track C classification | C | original (source-level) {Cm['classification']['original_OD']['source_level_instantiation']}; repaired {Cm['classification']['repaired_OD_gate']}; neural {Cm['classification']['neural_prediction']} |")
    W(FILES[4], "\n".join(fv) + "\n")
    st = ["# Statistical analysis\n", banner, "Parents are the independent units; pairs are nested within parents (two-stage bootstrap, 10 000 replicates; method-of-moments variance components).\n"]
    if A and "hypotheses" in A:
        for k, h in A["hypotheses"].items():
            if h.get("phi"): st.append(f"- {k}: phi mean {fmt(h['phi']['mean'])} CI {fmt(h['phi']['ci95'])}; parents {h['phi']['n_parents']}, pairs {h['phi']['n_pairs']}; tau2 {fmt((h['variance_components_between_unit'] or {}).get('tau2'), 4)}, sigma2 {fmt((h['variance_components_between_unit'] or {}).get('sigma2'), 4)}, ICC {fmt((h['variance_components_between_unit'] or {}).get('icc'))}")
    W(FILES[5], "\n".join(st) + "\n")
    ng = ["# Negative and inconclusive results (nothing is omitted)\n", banner]
    if A and "hypotheses" in A: ng += [f"- {k}: {h['verdict']}; censored pairs {json.dumps({s: n for s, n in h['pair_status_counts'].items() if s in ('MISSING', 'NONCOMPETENT', 'ANCHORS_NOT_EVALUABLE') and n})}" for k, h in A["hypotheses"].items() if h["verdict"] != "GENERALIZES"]
    ng += [f"- parent {p}: {e['status']}" for p, e in EL.items() if e["status"] != "SCREENED"] + [f"- parent {p}: no eligible unit for base {b}" for p, e in EL.items() if e["status"] == "SCREENED" for b in "EM" if not e["units"][b]["eligible"]]
    ng += [f"- FAILED job (kept, not replaced): {j}" for j in S["failed"]] + [f"- BLOCKED job: {j}" for j in S["blocked"]]
    W(FILES[6], "\n".join(ng) + "\n")
    fz = E.read_json(os.path.join(E.PKG, "protocols", "FREEZE.json")) or {}
    W(FILES[7], "\n".join(["# Methods and reproducibility\n", banner, f"- protocol version {C['version']}; freeze sha256 {fz.get('freeze_sha256', 'NO FREEZE FILE')}", "- protocol files: protocols/*/*.json (generated from cor_research/protocol.py)",
                           "- determinism: GPU --xla_gpu_deterministic_ops=true (run-to-run on the same GPU model/software stack; not bitwise equal to CPU); CPU single-threaded Eigen (bitwise)",
                           "- resume: every training job resumes from resume.pkl (parameters, AdamW state, data RNG bit-generator state, loss history); see tests and SMOKE_REPORT.json",
                           f"- environment: {json.dumps(E.read_json(os.path.join(root, 'environment_runtime.json'), {}))}"]) + "\n")
    cl = ["# Compute and execution log\n", banner, f"- registry: {S['n']} jobs; by phase {json.dumps(S['by_phase'])}", "", "| job | status | attempts | seconds | gpu MB max | max RSS MB |", "|---|---|---|---|---|---|"]
    for r in sorted(R.values(), key=lambda r: r["id"]):
        a = r["attempts"][-1] if r["attempts"] else {}; cl.append(f"| {r['id']} | {r['status']} | {len(r['attempts'])} | {fmt(a.get('seconds'), 0)} | {fmt(a.get('gpu_mem_mb_max'), 0)} | {fmt(a.get('max_rss_mb'), 0)} |")
    W(FILES[8], "\n".join(cl) + "\n")
    rv = sorted(glob.glob(os.path.join(root, "reviews_post_run", "*.md")))
    W(FILES[9], "\n".join(["# Review and adjudication\n", banner, "Pre-run design reviews (agent reviewers, before any Colab compute): see the package's reviews/ directory.",
                           ("Post-run reviews found: " + ", ".join(os.path.basename(f) for f in rv)) if rv else "**No post-run review has been performed.** The review bundle is in REVIEW_BUNDLE/ (scripts/make_review_bundle.py). Every conclusion above is provisional until it is reviewed. No review is claimed."]) + "\n")
    man = dict(preset=preset, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), complete=complete, incomplete_jobs=incomplete, technical_only=tech, freeze_sha256=fz.get("freeze_sha256"),
               evidence={os.path.relpath(f, root): E.sha256(f) for pat in ("measure/*/*.json", "trackB/*.json", "trackB/*.npz", "scores/*.json", "track_c/*.json", "eligibility.json", "registry/jobs/*.json") for f in sorted(glob.glob(os.path.join(root, pat)))})
    E.atomic_json(man, os.path.join(out, FILES[10]))
    syn = ["# Main research synthesis (generated from the completed results)\n", banner, "Three separate verdicts (mandate §6: they are not conflated):\n",
           f"1. Track A (training-state intervention, parent-level): " + (", ".join(f"{k}={h['verdict']}" for k, h in A["hypotheses"].items()) if A and "hypotheses" in A else "not scored"),
           f"2. Track B (distinguishable computation): " + (B["verdict"] if B and "verdict" in B else "not scored"),
           f"3. Track C (O/D): " + (f"original (source-level) {Cm['classification']['original_OD']['source_level_instantiation']}, repaired {Cm['classification']['repaired_OD_gate']}, neural {Cm['classification']['neural_prediction']}" if Cm else "not run"),
           "\nThe 28 questions of the mandate (§17) are answered by the reviewers from these files and the evidence manifest; this generator does not answer them.",
           "A positive Track A result does not prove a distinct algorithm; a positive Track B result does not prove the original O/D theory; a Bayesian-equivalent O/D computation does not show that a network implements O/D."]
    if not complete: syn.insert(2, f"**This synthesis is INCOMPLETE ({len(incomplete)} jobs not done, {len(unreg)} planned jobs unregistered) and claims no completed investigation; all verdicts are PROVISIONAL.**\n")
    W(FILES[0], "\n".join(syn) + "\n")
    return dict(complete=complete, n_jobs=len(R), incomplete=len(incomplete), unregistered=len(unreg), integrity_failures=ic_fail, out=out)


if __name__ == "__main__":
    preset = sys.argv[1]; r = generate(preset, out=sys.argv[2] if len(sys.argv) > 2 else None); print(json.dumps(r))
