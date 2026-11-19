"""XIX scorer validation (mandate §9 F2): the causal-necessity classifier and the functional (residual-correlation) classifier on known models
across independent streams that were never used to set thresholds.  Models: 24 XIV step-1211 runs, 12 XVIII confirmation anchors, XVIII
screening models of other parents (pure, hybrid, incompetent), XVIII gsg hybrids.  For each stream: nec_E, nec_M, KL, acc, causal class,
functional score f (correlation of the residual-to-Bayes profile with the early minus mixture reference centroid; references = the 12 XVIII
confirmation anchors measured on that stream; a model is never its own reference).  Writes logs/scorer_validation19.json."""
import os, sys, json, glob
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src")); sys.path.insert(0, os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src"))
import numpy as np, jax
import xviiim as XM, xix_score as XS
X14 = os.path.join(H, "..", "..", "step1_cross_family_relational_foundations", "checkpoints", "ft"); X18 = os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "checkpoints", "ft")
REF = sorted(d for d in glob.glob(os.path.join(X18, "uP160_fx_L2[13]_W2[13]_D81[123]_stop1211_c18")) + glob.glob(os.path.join(X18, "uP160_fx_L2[13]_W2[13]_D81[123]_gofown[02]b1e9T100_stop1211_c18")))
VAL = sorted(d for d in glob.glob(os.path.join(X14, "*")) if os.path.exists(os.path.join(d, "step1211.npz")) and ("stop1211" in d or d.endswith("_base")))
VAL += sorted(glob.glob(os.path.join(X18, "uP16[2345]_fx_*_stop1211_scr"))) + sorted(glob.glob(os.path.join(X18, "uP160_fx_*_gsgown[02]T100_stop1211_c18")))
par = lambda d: os.path.basename(d)[:5]
out = {}
for stream in [int(s) for s in sys.argv[1:]]:
    refP = {d: XS.profile(XM.load_dir(d, "uP160", 1211), stream) for d in REF}
    refcls = {d: XS.causal_class(XS.necessity(XM.load_dir(d, "uP160", 1211), stream)) for d in REF}
    rows = {}
    for d in VAL + REF:
        M = XM.load_dir(d, par(d), 1211); nec = XS.necessity(M, stream); prof = refP[d] if d in refP else XS.profile(M, stream)
        others = {k: v for k, v in refP.items() if k != d}
        f = XS.functional_score(prof, {k: v for k, v in others.items() if refcls[k] == "EARLY"}, {k: v for k, v in others.items() if refcls[k] == "MIXTURE"})
        rows[os.path.basename(d)] = dict(**nec, causal=XS.causal_class(nec), f=f, functional=XS.functional_class(f), competent=XS.competent(nec), ref=d in refP)
        print(stream, os.path.basename(d)[:58].ljust(58), {k: round(v, 3) for k, v in nec.items()}, rows[os.path.basename(d)]["causal"], "f=%+.3f" % f, rows[os.path.basename(d)]["functional"], flush=True)
    out[stream] = rows
json.dump(out, open(os.path.join(H, "..", "logs", "scorer_validation19.json"), "w"), indent=1)
