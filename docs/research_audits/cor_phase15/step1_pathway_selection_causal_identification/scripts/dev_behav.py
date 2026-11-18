"""XVIII development: behavioural (attention-free) diagnostics on XIV step-1211 checkpoints: power-mean exponent rho, R2, slopes, KL.
Usage: dev_behav.py OUT SPEC...   (SPEC as in XIV src/xc.py).  Development stream 86_100_000 (XIII-C metric sets)."""
import json, os, sys, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "step1_cross_family_relational_foundations", "src"))
import xc, metrics as MT
S = MT.sets(86_100_000); out = sys.argv[1]
for sp in sys.argv[2:]:
    t0 = time.time(); M = xc.load(sp, 1211); b, Ln = MT.behaviour(M, S); b.update(MT.rho_fit(Ln, S)); b.update(unit=sp, sec=time.time() - t0)
    open(out, "a").write(json.dumps(b) + "\n"); print(sp, {k: round(v, 3) for k, v in b.items() if isinstance(v, float)}, flush=True)
