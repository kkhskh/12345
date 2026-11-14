"""Dependency inventory for Investigation XIII -> logs/DEPENDENCIES.json.
(1) software versions; (2) Python modules imported from OUTSIDE this package (path + sha256); (3) every external data file read by XIII scripts:
development parents (pre.npz), development fine-tuned checkpoints (all step*.npz, incl. trajectories used by transition_predictor), XII DAS E subspaces
used for development score_contrast / l1abl_mix, the development evaluation set.  The fresh cohort (uP160-165) lives INSIDE this package and is
covered by MANIFEST.sha256.  Paths relative to cor_phase15/."""
import hashlib, json, os, sys, glob, platform
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); ROOT = os.path.abspath(os.path.join(PKG, ".."))
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, optax, sklearn, scipy
import native as NV, ftlib as F, genlib as GL, xii_bayes
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)
out = dict(software=dict(python=platform.python_version(), jax=jax.__version__, jaxlib=__import__("jaxlib").__version__, numpy=np.__version__, optax=optax.__version__,
                         scikit_learn=sklearn.__version__, scipy=scipy.__version__, platform=platform.platform()),
           env={"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"})
out["external_python_modules"] = {n: dict(path=rel(m.__file__), sha256=sha(m.__file__)) for n, m in list(sys.modules.items())
                                  if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(ROOT) and not os.path.abspath(m.__file__).startswith(PKG)}
dev = set()
for f in glob.glob(os.path.join(PKG, "logs", "*_dev*.json")) + glob.glob(os.path.join(PKG, "logs", "*_pre.json")):
    try: dev.update(k for k in json.load(open(f)) if ":" in k and not k.startswith("NULL"))
    except Exception: pass
files = set()
for sp in sorted(dev):
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; tag = f"{parent}_{spec}_s4000_seed{int(seed or 0)}{suf}"
    files.add(NV.parent_path(parent))
    try: files.update(glob.glob(os.path.join(F.model_dir(tag), "step*.npz")))
    except FileNotFoundError: pass
    files.update(glob.glob(os.path.join(ROOT, "step1_native_relational_algorithm_validation", "logs", "das", f"{tag}_E_L2_q7_k6*_R.npy")))
    files.update(glob.glob(os.path.join(ROOT, "step1_causal_equivalence", "logs", "das", f"{tag}_E_L2_q7_k6*_R.npy")))
files.add(NV.EVAL_DEV)
missing = [rel(f) for f in files if not os.path.exists(f)]
out["development_models"] = sorted(dev); out["external_data_files"] = {rel(f): sha(f) for f in sorted(files) if os.path.exists(f)}
out["missing"] = missing; out["n_external_data_files"] = len(out["external_data_files"])
out["sibling_packages_required"] = sorted({p.split(os.sep)[0] for p in list(out["external_data_files"]) + [v["path"] for v in out["external_python_modules"].values()]})
json.dump(out, open(os.path.join(PKG, "logs", "DEPENDENCIES.json"), "w"), indent=1)
print(json.dumps(dict(software=out["software"], n_dev_models=len(dev), n_files=out["n_external_data_files"], missing=missing, siblings=out["sibling_packages_required"]), indent=1))
