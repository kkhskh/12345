"""XVII dependency inventory -> logs/DEPENDENCIES.json: software versions; every Python module imported from outside this package (path,
sha256) when the XVI scripts' imports are loaded; external data files (the six XV checkpoints the continuations start from, and the XV
snapshots used by the exploratory original-schedule diagnostics).  Paths relative to cor_phase15/."""
import hashlib, json, os, sys, platform, glob
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); ROOT = os.path.abspath(os.path.join(PKG, ".."))
for p in ("src", "scripts"): sys.path.insert(0, os.path.join(PKG, p))
for p in ("step1_foundational_atlas_continuation/src", "step1_cross_family_relational_foundations/src", "step1_latent_reliability/src", "step1_relational_mechanism_deepening/src"): sys.path.insert(0, os.path.join(ROOT, p))
import numpy as np, jax, optax, scipy, sklearn
import xvitask, xvidiag, xviislot, xvi_math_core, train_tasks, gatelib, nets, bayes, fwd3
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)
ext = {n: dict(path=rel(m.__file__), sha256=sha(m.__file__)) for n, m in list(sys.modules.items()) if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(ROOT) and not os.path.abspath(m.__file__).startswith(PKG)}
XVI = os.path.join(ROOT, "step1_relational_acquisition_limits")
data = {rel(f): sha(f) for f in [os.path.join(XVI, "confirm_outputs", r, f"step{k}.npz") for r, k in (("WA_cont_C_s202", 4000), ("WB_low_s503", 2500))]}
data[rel(os.path.join(ROOT, "step1_relational_learning_order_laws", "confirm_outputs", "C_s202", "step4000.npz"))] = sha(os.path.join(ROOT, "step1_relational_learning_order_laws", "confirm_outputs", "C_s202", "step4000.npz"))
out = dict(software=dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__, optax=optax.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
           env={"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}, external_python_modules=ext,
           sibling_packages_required=sorted({v["path"].split(os.sep)[0] for v in ext.values()} | {"step1_relational_learning_order_laws", "step1_relational_acquisition_limits"}),
           external_data_files=dict(development_model_files=data, note="development replays regenerate XVI/XV runs from these packages"))
json.dump(out, open(os.path.join(PKG, "logs", "DEPENDENCIES.json"), "w"), indent=1); print(len(ext), out["sibling_packages_required"])
