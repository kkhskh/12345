"""Complete dependency inventory for Investigation XII (copied from XI; model list extended to the fresh cohort and XII dev set) (logs/DEPENDENCIES.json + DEPENDENCIES.md table source).
(1) software: interpreter and package versions;  (2) every Python module imported from OUTSIDE this package (path + sha256);
(3) every external data file read by the XI scripts: parent checkpoints (pre.npz), fine-tuned checkpoints (all step*.npz used, incl. dynamics),
    Investigation X E-subspaces (*_E_L2_q7_k6*_R.npy), evaluation sets, Investigation X trajectory file.  Paths are relative to cor_phase15/."""
import hashlib, importlib, json, os, sys, glob, platform
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); ROOT = os.path.abspath(os.path.join(PKG, ".."))
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, optax, sklearn, scipy
import native as NV, ftlib as F, genlib as GL
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)
out = dict(software=dict(python=platform.python_version(), jax=jax.__version__, jaxlib=__import__("jaxlib").__version__, numpy=np.__version__, optax=optax.__version__,
                         scikit_learn=sklearn.__version__, scipy=scipy.__version__, platform=platform.platform()),
           env={"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"})
mods = {}
for name, m in list(sys.modules.items()):
    f = getattr(m, "__file__", None)
    if f and os.path.abspath(f).startswith(ROOT) and not os.path.abspath(f).startswith(PKG): mods[name] = dict(path=rel(f), sha256=sha(f))
out["external_python_modules"] = mods
files = set()
specs = ["xP81:full_c1:0:", "xP81:full_c1:1:", "xP81:c1:0:", "yP130:full_c1:0:_eval_conf9", "yP130:full_c1:1:", "yP130:c1:0:_eval_conf9", "yP131:full_c1:0:_eval_conf9",
         "yP131:c1:0:_eval_conf9", "yP132:full_c1:0:_eval_conf9", "yP134:full_c1:0:_eval_conf9"] + [f"wP14{i}:{s}:0:_eval_conf10" for i in range(4) for s in ("full_c1", "c1")]
ext_only = True   # XII: fresh-cohort checkpoints live INSIDE this package and are covered by MANIFEST.sha256, not by this inventory
for sp in specs:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; tag = f"{parent}_{spec}_s4000_seed{int(seed or 0)}{suf}"
    files.add(NV.parent_path(parent)); files.update(glob.glob(os.path.join(F.model_dir(tag), "step*.npz")))
    files.update(glob.glob(os.path.join(ROOT, "step1_causal_equivalence", "logs", "das", f"{tag}_E_L2_q7_k6*_R.npy")))
files.add(NV.EVAL_DEV); files.update(glob.glob(os.path.join(ROOT, "step1_native_causal_construction", "logs", "decoy*", "*_full_c1_*.json"))); files.add(os.path.join(ROOT, "step1_causal_equivalence", "logs", "eval_conf10.npz")); files.add(os.path.join(ROOT, "step1_causal_equivalence", "logs", "traj_alg_conf.json"))
files.add(os.path.join(ROOT, "step1_causal_equivalence", "logs", "traj_alg.json"))
missing = [rel(f) for f in files if not os.path.exists(f)]
out["external_data_files"] = {rel(f): sha(f) for f in sorted(files) if os.path.exists(f)}
out["missing"] = missing; out["n_external_data_files"] = len(out["external_data_files"])
out["sibling_packages_required"] = sorted({p.split(os.sep)[0] for p in list(out["external_data_files"]) + [v["path"] for v in mods.values()]})
json.dump(out, open(os.path.join(PKG, "logs", "DEPENDENCIES.json"), "w"), indent=1)
print(json.dumps(dict(software=out["software"], n_modules=len(mods), n_files=out["n_external_data_files"], missing=missing, siblings=out["sibling_packages_required"]), indent=1))
