"""Write FREEZE_XIX.json: SHA-256 of every file that defines the confirmation (code, design, hypotheses, queues, reference cache, reviews,
design documents) plus the git HEAD at freeze.  Usage: freeze19.py"""
import os, json, hashlib, glob, subprocess, datetime
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, ".."))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
files = sorted(glob.glob(os.path.join(R, "scripts", "*.py")) + glob.glob(os.path.join(R, "scripts", "*.sh")) + glob.glob(os.path.join(R, "src", "*.py")) +
               [os.path.join(R, f) for f in ("logs/conf/design19.json", "logs/conf/queue_c1.txt", "logs/conf/queue_c2.txt", "logs/conf_spec19.json", "logs/refs19_94700000.npz",
                                             "logs/scorer_planted19.json", "logs/scorer_validation_summary19.json", "logs/graph_table19.csv", "DESIGN_XIX.md",
                                             "PRE_EXPERIMENT_IDENTIFIABILITY_REVIEWS.md", "sources/MANDATE_XIX.txt")] + glob.glob(os.path.join(R, "reviews", "*.md")))
X = os.path.join(R, ".."); deps = [os.path.join(X, "step1_pathway_selection_causal_identification", "src", "xviiim.py")] + [os.path.join(X, "step1_cross_family_relational_foundations", "src", f) for f in ("native.py", "ftlib.py", "metrics.py", "maskfwd.py", "relpairs.py", "genlib.py")]
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=R).stdout.strip()
F = dict(written_utc=datetime.datetime.utcnow().isoformat() + "Z", git_head_before_freeze_commit=head, confirmation_stream=94700000, data_orders=[821, 822, 823], parent="uP160",
         files={os.path.relpath(f, R): sha(f) for f in files}, external_dependencies={os.path.relpath(f, X): sha(f) for f in deps},
         rule="no confirmation run before this file is committed and pushed; before scoring only COMPUTE_LOG and file existence are monitored")
json.dump(F, open(os.path.join(R, "FREEZE_XIX.json"), "w"), indent=1); print(len(F["files"]), "files frozen; head", head)
