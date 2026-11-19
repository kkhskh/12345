"""Step-100 diagnostic of training-time arms: KL, accuracy and test-time route necessity of the step-100 state (before continuation).
Usage: s100nec19.py STREAM OUT.json DIR...   (DIR = run directories that contain step100.npz)"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src")); sys.path.insert(0, os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src"))
import xviiim as XM, xix_score as XS
stream, out = int(sys.argv[1]), sys.argv[2]; R = json.load(open(out)) if os.path.exists(out) else {}
for d in sys.argv[3:]:
    k = os.path.basename(os.path.normpath(d))
    if k not in R: R[k] = XS.necessity(XM.load_dir(d, "uP160", 100), stream)
    print(k[:60].ljust(60), {a: round(b, 3) for a, b in R[k].items()}, flush=True)
json.dump(R, open(out, "w"), indent=1)
