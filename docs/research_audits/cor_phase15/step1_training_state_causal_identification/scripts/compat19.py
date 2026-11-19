"""State-compatibility diagnostic for composed step-100 states (mandate A7): KL to Bayes, accuracy and test-time route necessity of the
step-100 state itself (before any continuation), for each composed tp_* directory and the matched B and I sources it names.
Usage: compat19.py STREAM OUT.json TPDIR...   (a composed state far worse than both sources signals an incompatible hybrid state)"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src")); sys.path.insert(0, os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src"))
import xviiim as XM, xix_score as XS
stream, out = int(sys.argv[1]), sys.argv[2]; R = json.load(open(out)) if os.path.exists(out) else {}; C = os.path.join(H, "..", "checkpoints", "ft")
def meas(d):
    if d not in R: R[d] = XS.necessity(XM.load_dir(os.path.join(C, d), "uP160", 100), stream)
    return R[d]
for tp in sys.argv[3:]:
    tp = os.path.basename(os.path.normpath(tp)); pv = json.load(open(os.path.join(C, tp, "provenance.json")))
    for d in (pv["B"], pv["I"], tp): meas(d)
    print(tp.ljust(30), {k: round(v, 3) for k, v in R[tp].items()}, "| src B", round(R[pv["B"]]["KL"], 3), "src I", round(R[pv["I"]]["KL"], 3), flush=True)
json.dump(R, open(out, "w"), indent=1)
