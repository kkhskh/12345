"""Track A measurement (v2): python -m cor_research.scoring.measure RUN_DIR PARENT STEP STREAM OUT.json
Records the frozen XIX primitives (xix_score.necessity: KL to exact Bayes and Bayes-argmax accuracy on 1500 natural episodes of the
measurement stream; necessity of the layer-0 own-question route own0 and the layer-2 route own2) and, v2 (review A M5), episode-bootstrap
standard errors of nec_E / nec_M (200 resamples; the per-episode terms are recomputed with the same calls and their means are asserted to
equal xix_score.necessity).  Labels are NOT trusted downstream.  Provenance: checkpoint sha256, parent checkpoint sha256, measurement-code
sha256 (every module on the measurement path incl. env.py and models/load.py).  Writes OUT atomically; refuses to overwrite a different record."""
import os, sys, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from cor_research import env as E
E.configure(); E.legacy_paths()
MEAS_FILES = ["step1_training_state_causal_identification/src/xix_score.py", "step1_pathway_selection_causal_identification/src/xviiim.py"] + \
    [f"step1_cross_family_relational_foundations/src/{f}" for f in ("native.py", "ftlib.py", "metrics.py", "maskfwd.py", "relpairs.py", "genlib.py", "nnlib.py")]


def code_hashes():
    """sha256 of EVERY vendored legacy module (review C r2 N20: the import graph is wide) plus the XX measurement path"""
    import glob
    h = {os.path.relpath(f, E.LEGACY): E.sha256(f) for f in sorted(glob.glob(os.path.join(E.LEGACY, "**", "*.py"), recursive=True))}
    for f in ("scoring/measure.py", "models/load.py", "env.py"): h[f] = E.sha256(os.path.join(E.PKG, "cor_research", f))
    return h


def measure(run_dir, parent, step, stream, n_boot=200):
    from cor_research.models import load as LD
    import numpy as np, xix_score as XS
    t = time.time(); M = LD.load(run_dir, parent, step); n = XS.necessity(M, stream); NV = E.patch_parents()
    X, q, PB = XS._nat(stream); i = np.arange(len(q)); MF = XS.MF
    def per(L): P = XS.XM.RP.sm(np.asarray(L)[i, q]); return (PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1)
    k0 = per(MF.forward(M, X)); kE = per(MF.forward(M, X, biases=dict([MF.edges("own0")]))); kM = per(MF.forward(M, X, biases=dict([MF.edges("own2")])))
    assert abs(k0.mean() - n["KL"]) < 1e-6 and abs((kE - k0).mean() - n["nec_E"]) < 1e-6 and abs((kM - k0).mean() - n["nec_M"]) < 1e-6, "per-episode terms disagree with xix_score"
    rng = np.random.default_rng(stream + 13); B = rng.integers(0, len(q), (n_boot, len(q)))
    se = dict(nec_E_se=float((kE - k0)[B].mean(1).std(ddof=1)), nec_M_se=float((kM - k0)[B].mean(1).std(ddof=1)), KL_se=float(k0[B].mean(1).std(ddof=1)))
    return dict(run=os.path.basename(os.path.normpath(run_dir)), parent=parent, step=int(step), stream=int(stream), **{k: float(v) for k, v in n.items()}, **se,
                ckpt_sha256=E.sha256(os.path.join(run_dir, f"step{step}.npz")), parent_sha256=E.sha256(NV.parent_path(parent)), code_sha256=code_hashes(), seconds=time.time() - t)


if __name__ == "__main__":
    d, parent, step, stream, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    rec = measure(d, parent, step, stream)
    if os.path.exists(out):
        old = json.load(open(out))
        if {k: v for k, v in old.items() if k != "seconds"} != {k: v for k, v in rec.items() if k != "seconds"}: raise SystemExit(f"{out}: existing record differs; refusing to overwrite")
    else: E.atomic_json(rec, out)
    print(json.dumps({k: rec[k] for k in ("run", "step", "KL", "acc", "nec_E", "nec_M", "nec_E_se", "nec_M_se")}))
