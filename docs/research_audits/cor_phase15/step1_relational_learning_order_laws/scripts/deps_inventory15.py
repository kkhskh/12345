"""XV dependency inventory -> logs/DEPENDENCIES.json: software versions; every Python module imported from outside this package (path,
sha256) when the XV scripts' imports are loaded; external data files (none beyond modules: XV trains everything it scores; the XIV
gatelib loader is used as code only).  Paths relative to cor_phase15/."""
import hashlib, json, os, sys, platform
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); ROOT = os.path.abspath(os.path.join(PKG, ".."))
for p in ("src", "scripts"): sys.path.insert(0, os.path.join(PKG, p))
for p in ("step1_foundational_atlas_continuation/src", "step1_cross_family_relational_foundations/src", "step1_latent_reliability/src"): sys.path.insert(0, os.path.join(ROOT, p))
import numpy as np, jax, optax, scipy, sklearn
import xvtask, xvdiag, xv_math_core, train_tasks, gatelib, nets, bayes
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)
ext = {n: dict(path=rel(m.__file__), sha256=sha(m.__file__)) for n, m in list(sys.modules.items()) if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(ROOT) and not os.path.abspath(m.__file__).startswith(PKG)}
out = dict(software=dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__, optax=optax.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
           env={"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}, external_python_modules=ext,
           sibling_packages_required=sorted({v["path"].split(os.sep)[0] for v in ext.values()}), external_data_files={})
json.dump(out, open(os.path.join(PKG, "logs", "DEPENDENCIES.json"), "w"), indent=1); print(len(ext), out["sibling_packages_required"])
