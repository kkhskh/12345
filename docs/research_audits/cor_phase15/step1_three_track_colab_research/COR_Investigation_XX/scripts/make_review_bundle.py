"""Post-run adversarial review bundle (mandate §13).  python scripts/make_review_bundle.py PRESET  ->  $COR_RUN_ROOT/REVIEW_BUNDLE/
Contents: frozen hypotheses and protocols (+ FREEZE.json), raw predictions, observed results (scores, RESULTS), failed / blocked /
censored runs, controls, statistical analyses, causal-intervention evidence (graph audit, transplant provenance), Track C claims,
reproducibility reports (integrity checks, smoke report, environment), reviewer instructions and automated review checks.
This script does NOT perform a review and nothing it writes claims that a review happened."""
import os, sys, json, glob, shutil, time
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG)
from cor_research import env as E
preset = sys.argv[1]; root = E.RUN_ROOT; B = os.path.join(root, "REVIEW_BUNDLE"); os.makedirs(B, exist_ok=True)
cp = lambda src, dst: (os.makedirs(os.path.dirname(os.path.join(B, dst)), exist_ok=True), shutil.copy2(src, os.path.join(B, dst))) if os.path.exists(src) else None
for f in glob.glob(os.path.join(PKG, "protocols", "*", "*.json")) + [os.path.join(PKG, "protocols", "FREEZE.json")]: cp(f, os.path.join("frozen", os.path.relpath(f, PKG)))
for f in glob.glob(os.path.join(root, "scores", "*.json")) + glob.glob(os.path.join(root, "RESULTS", "*")): cp(f, os.path.join("observed", os.path.relpath(f, root)))
for f in [os.path.join(root, x) for x in ("eligibility.json", "benchmark.json", "track_c/track_c_math.json", "preflight/graph_audit.json", "preflight/PREFLIGHT.json", "RESULTS/INTEGRITY_CHECKS.json", "environment_history.json")]: cp(f, os.path.join("evidence", os.path.basename(f)))
for f in glob.glob(os.path.join(root, "ft", "tp_*", "provenance.json")): cp(f, os.path.join("evidence", "transplant_provenance", os.path.basename(os.path.dirname(f)) + ".json"))
from cor_research.jobs import runner as RN
R = RN.load_all(root); E.atomic_json({k: v for k, v in R.items() if v["status"] != "done"}, os.path.join(B, "failed_blocked_incomplete_jobs.json"))
A = E.read_json(os.path.join(root, "scores", f"track_a_{preset}.json")) or {}
checks = dict(n_jobs=len(R), n_not_done=sum(r["status"] != "done" for r in R.values()),
              hypotheses_with_censoring={k: {s: n for s, n in h["pair_status_counts"].items() if s in ("MISSING", "NONCOMPETENT", "ANCHORS_NOT_EVALUABLE") and n} for k, h in A.get("hypotheses", {}).items()},
              parents_inconclusive={k: [p for p, v in h["parent_verdicts"].items() if v == "PARENT_INCONCLUSIVE"] for k, h in A.get("hypotheses", {}).items()},
              technical_only=A.get("technical_only"))
E.atomic_json(checks, os.path.join(B, "AUTOMATED_REVIEW_CHECKS.json"))
open(os.path.join(B, "REVIEWER_INSTRUCTIONS.md"), "w").write(f"""# Reviewer instructions — COR Investigation XX ({preset})

Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.  No review has been performed by this package.  Reviews must be done by
independent agents (or people) who did not build the package; record each review as `reviews_post_run/<reviewer>.md` in the run root
and regenerate RESULTS (the review file names then appear in REVIEW_AND_ADJUDICATION.md).

Assign at least three reviewers: causal identification; mathematics and functional equivalence; reproducibility and integrity.
Each reviewer should, at minimum:
1. Check that every verdict in observed/RESULTS/FROZEN_PREDICTION_VERDICTS.md follows from frozen/ and observed/scores by the frozen rules
   (re-run `python -m cor_research.scoring.track_a {preset} x.json` on the run root; integrity check 16 does this automatically).
2. Read failed_blocked_incomplete_jobs.json and AUTOMATED_REVIEW_CHECKS.json: is any conclusion carried by a small or selected subset?
3. Track A: is the parent (not the pair) the unit of every claim?  Is the denominator (eligibility.json) complete?
4. Track B: is any functional difference explained by precision (HB_e) or by the noise null?  Is an interchange success read as identity?
5. Track C: does any claimed distinctive prediction differ from exact Bayes on the same declared generative model?
6. Answer the 28 questions of the mandate (§17) from the evidence, marking each as answered / not answerable from this run.
""")
print("review bundle written to", B)
