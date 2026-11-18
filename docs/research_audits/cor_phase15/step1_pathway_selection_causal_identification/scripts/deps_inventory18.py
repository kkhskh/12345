"""XVIII dependency inventory -> logs/DEPENDENCIES.json: software versions; every Python module imported from outside this package when the
measurement library and the training harness's imports are loaded (path, sha256); external data files (pretrained parents uP160-uP165,
the XIII-B evaluation file read at the end of training, XIV checkpoints used in development).  Paths relative to cor_phase15/."""
import hashlib, json, os, sys, platform, glob
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); ROOT = os.path.abspath(os.path.join(PKG, ".."))
sys.path.insert(0, os.path.join(PKG, "src")); sys.path.insert(0, os.path.join(ROOT, "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax, optax, scipy
import xviiim, native as NV, maskfwd, metrics, ftlib, relpairs, genlib, xc
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)
ext = {n: dict(path=rel(m.__file__), sha256=sha(m.__file__)) for n, m in list(sys.modules.items()) if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(ROOT) and not os.path.abspath(m.__file__).startswith(PKG)}
data = {}
for p in ("uP160", "uP161", "uP162", "uP163", "uP164", "uP165"): f = NV.parent_path(p); data[rel(f)] = sha(f)
ev = os.path.join(NV.X13B, "logs", "eval_conf13.npz"); data[rel(ev)] = sha(ev)
X14 = os.path.join(ROOT, "step1_cross_family_relational_foundations", "checkpoints", "ft")
dev = {rel(f): sha(f) for f in sorted(glob.glob(os.path.join(X14, "*", "step1211.npz")))}
out = dict(software=dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__, optax=optax.__version__, scipy=scipy.__version__),
           env={"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}, external_python_modules=ext,
           sibling_packages_required=sorted({v["path"].split(os.sep)[0] for v in ext.values()} | {k.split(os.sep)[0] for k in data}),
           external_data_files=dict(parents_and_eval=data, xiv_step1211_checkpoints_used_in_development=dev))
json.dump(out, open(os.path.join(PKG, "logs", "DEPENDENCIES.json"), "w"), indent=1); print(len(ext), "modules", out["sibling_packages_required"])
