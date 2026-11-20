"""XIX measurement: measure19.py RUN_DIR PARENT STREAM OUT.jsonl [STEP=1211]
Records necessity (nec_E, nec_M, KL, acc), causal class, competence, functional score f and class, and provenance:
sha256 of the measured checkpoint, sha256 of src/xix_score.py, sha256 of the reference set (the 12 XVIII c18 anchors' checkpoints).
Reference profiles for a stream are cached in logs/refs19_<stream>.npz together with the checkpoint hashes they came from."""
import os, sys, json, hashlib, glob, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src")); sys.path.insert(0, os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src"))
import numpy as np
import xviiim as XM, xix_score as XS
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
X18 = os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "checkpoints", "ft")
REF = sorted(glob.glob(os.path.join(X18, "uP160_fx_L2[13]_W2[13]_D81[123]_stop1211_c18")) + glob.glob(os.path.join(X18, "uP160_fx_L2[13]_W2[13]_D81[123]_gofown[02]b1e9T100_stop1211_c18")))
assert len(REF) == 12, REF
X14S = os.path.join(H, "..", "..", "step1_cross_family_relational_foundations", "src"); X18S = os.path.join(H, "..", "..", "step1_pathway_selection_causal_identification", "src")
DEPF = [os.path.join(H, "..", "src", "xix_score.py"), os.path.abspath(__file__), os.path.join(X18S, "xviiim.py")] + [os.path.join(X14S, f) for f in ("native.py", "ftlib.py", "metrics.py", "maskfwd.py", "relpairs.py", "genlib.py")]
DEP = {os.path.basename(f): sha(f) for f in DEPF}                         # pre-review C5: every measurement-path module (incl. the Bayes posterior)
def refs(stream):
    p = os.path.join(H, "..", "logs", f"refs19_{stream}.npz"); hs = [sha(os.path.join(d, "step1211.npz")) for d in REF] + ["code:" + hashlib.sha256(json.dumps(DEP, sort_keys=True).encode()).hexdigest()]
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if list(z["hashes"]) == hs: return z
    P, C = [], []
    for d in REF:
        M = XM.load_dir(d, "uP160", 1211); P.append(XS.profile(M, stream)); C.append(XS.causal_class(XS.necessity(M, stream)))
    assert sorted(C) == ["EARLY"] * 6 + ["MIXTURE"] * 6, C
    np.savez(p, profiles=np.array(P), classes=np.array(C), hashes=np.array(hs), names=np.array([os.path.basename(d) for d in REF])); return np.load(p, allow_pickle=True)
if __name__ == "__main__":
    d, parent, stream, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; step = int(sys.argv[5]) if len(sys.argv) > 5 else 1211
    t = time.time(); z = refs(stream); ck = os.path.join(d, f"step{step}.npz"); M = XM.load_dir(d, parent, step); n = XS.necessity(M, stream); prof = XS.profile(M, stream)
    names = list(z["names"]); me = os.path.basename(os.path.normpath(d))
    E = {k: v for k, v, c in zip(names, z["profiles"], z["classes"]) if c == "EARLY" and k != me}; Mx = {k: v for k, v, c in zip(names, z["profiles"], z["classes"]) if c == "MIXTURE" and k != me}
    f = XS.functional_score(prof, E, Mx)
    rec = dict(run=me, parent=parent, stream=stream, step=step, **n, causal=XS.causal_class(n), competent=XS.competent(n), f=f, functional=XS.functional_class(f),
               ckpt_sha256=sha(ck), dep_sha256=DEP, score_code_sha256=sha(os.path.join(H, "..", "src", "xix_score.py")), refs_sha256=hashlib.sha256("".join(z["hashes"]).encode()).hexdigest(), sec=time.time() - t)
    with open(out, "a") as fo: fo.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
