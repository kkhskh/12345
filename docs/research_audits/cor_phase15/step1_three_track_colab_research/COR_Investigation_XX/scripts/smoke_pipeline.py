"""Complete small-scale technical pipeline (SOFTWARE VALIDATION ONLY, never evidence).  python scripts/smoke_pipeline.py RUN_ROOT [--concurrency 2]
Runs the 'smoke' preset (shortened schedules, forced units on the vendored development parent uP160) through the SAME planner, runner,
harness, composer, measurement, Track B battery, scorers and report generator used by the scientific presets, and checks:
  * a training job killed mid-run (FX_DIE_AFTER) resumes from its exact saved state and is bitwise identical to an uninterrupted run;
  * the registry records every job with hashes and resource measurements; a failed job stays failed (not replaced);
  * the scorers run and force TECHNICAL_ONLY verdicts; the report generator refuses completion claims while jobs are incomplete."""
import os, sys, json, time, shutil, subprocess
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG)
root = os.path.abspath(sys.argv[1]); os.makedirs(root, exist_ok=True)
os.environ.update(COR_RUN_ROOT=root, COR_PARENT_ROOT=os.path.join(root, "parents"), COR_DEVICE=os.environ.get("COR_DEVICE", "cpu"))
from cor_research import env as E
E.configure()
from cor_research.jobs import plan as PL, runner as RN
conc = int(sys.argv[sys.argv.index("--concurrency") + 1]) if "--concurrency" in sys.argv else 2
report = dict(started=time.time(), checks={})

# 1. pipeline (Track A + B); Track C is exercised separately (tests/test_track_c.py) because its LP certificates take minutes
R = RN.Runner(root, conc); s = R.run(lambda: PL.plan("smoke", root, tracks="AB"), require_freeze=False)
report["summary"] = s; report["checks"]["all_jobs_done"] = not s["failed"] and not s["blocked"] and all(set(v) <= {"done"} for v in s["by_phase"].values())

# 2. exact resume: rerun one continuation in a scratch copy, killed after its first post-branch snapshot, then resumed
J = {j["id"]: j for j in PL.plan("smoke", root, tracks="A")}; j = J["cont:uP160:E2001:fullB"]; tag = j["outputs"][0].split("/")[1]
alt = j["env"]["FX_TAGSUF"] + "_resumetest"; env = dict(os.environ, PYTHONPATH=PKG, **dict(j["env"], FX_TAGSUF=alt))
r1 = subprocess.run(j["argv"], cwd=root, env=dict(env, FX_DIE_AFTER="150"), capture_output=True, text=True)
r2 = subprocess.run(j["argv"], cwd=root, env=env, capture_output=True, text=True)
d0, d1 = os.path.join(root, "ft", tag), os.path.join(root, "ft", tag.replace(j["env"]["FX_TAGSUF"], alt))
same = all(E.sha256(os.path.join(d0, f)) == E.sha256(os.path.join(d1, f)) for f in sorted(os.listdir(d0)) if f.startswith("step") and f.endswith(".npz"))
report["checks"]["resume_exit_codes"] = [r1.returncode, r2.returncode]; report["checks"]["resume_log"] = E.read_json(os.path.join(d1, "resume_log.json"))
report["checks"]["resume_bitwise_identical"] = bool(r1.returncode == 3 and r2.returncode == 0 and same)

# 3. a deliberately failing job is recorded as failed and never replaced
bad = dict(PL.job("smoke:deliberate_failure", "A", "window", 99, [sys.executable, "-c", "import sys; sys.exit(7)"], {}, ["nonexistent/output.npz"]))
R2 = RN.Runner(root, 1); R2.run(lambda: [bad], require_freeze=False); rec = RN.load_all(root)["smoke:deliberate_failure"]
R2.run(lambda: [bad], require_freeze=False); rec2 = RN.load_all(root)["smoke:deliberate_failure"]
report["checks"]["failed_job_recorded_not_replaced"] = bool(rec["status"] == "failed" and rec2["status"] == "failed" and len(rec2["attempts"]) == 1 and rec["attempts"][0]["rc"] == 7)

# 4. scorers and reports (no freeze file needed for the smoke: frozen hashes are taken from the current code)
from cor_research.scoring import track_a as TA, track_b as TB, measure as MS, trackb_run as TBR
a = TA.score_all("smoke", root, frozen_measure_code=MS.code_hashes()); b = TB._np(TB.score("smoke", root, frozen_code=TBR.code_hashes()))
E.atomic_json(a, os.path.join(root, "scores", "track_a_smoke.json")); E.atomic_json(b, os.path.join(root, "scores", "track_b_smoke.json"))
report["checks"]["track_a_scored_not_provenance_fail"] = a.get("verdict") != "PROVENANCE_FAIL"; report["track_a_problems"] = a.get("problems")
report["checks"]["track_a_technical_only"] = a.get("verdict") != "PROVENANCE_FAIL" and all(h["verdict"] == "TECHNICAL_ONLY" for h in a["hypotheses"].values())
report["checks"]["track_b_technical_only"] = b.get("verdict") == "TECHNICAL_ONLY"; report["track_b_problems"] = b.get("problems")
from cor_research.reporting import report as RP
from cor_research.verification import integrity as IN
RP.generate("smoke", root, os.path.join(root, "RESULTS")); ic = IN.run("smoke", root); E.atomic_json(ic, os.path.join(root, "RESULTS", "INTEGRITY_CHECKS.json"))
res = RP.generate("smoke", root, os.path.join(root, "RESULTS")); report["integrity"] = {k: v["status"] for k, v in ic["checks"].items()}; report["checks"]["report_generated"] = sorted(os.listdir(os.path.join(root, "RESULTS")))
report["checks"]["report_refuses_completion"] = res["complete"] is False      # the deliberate failure keeps the run incomplete
report["seconds"] = time.time() - report["started"]; E.atomic_json(report, os.path.join(root, "SMOKE_REPORT.json"))
print(json.dumps(report["checks"], indent=1)); sys.exit(0 if all(v for k, v in report["checks"].items() if isinstance(v, bool)) else 1)
