"""Dependency inventory for Investigation XIII-D -> logs/DEPENDENCIES.json.
(1) software versions; (2) Python modules imported from OUTSIDE this package (path + sha256); (3) every external data file read:
the XIII cohort parents (pre.npz), the XIII / XIII-B seed fine-tunes used as diagonal references and development units (every step*.npz of every
such unit directory, since trajectories read several steps), and the XIII-B evaluation set eval_conf13.npz (read by ft_factorial's final
evaluation).  Unit specs are collected from every logs/*.json(l) of this package.  Paths relative to cor_phase15/."""
import hashlib, json, os, sys, glob, platform, re
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); ROOT = os.path.abspath(os.path.join(PKG, ".."))
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, optax, scipy
import native as NV, ftlib as F, genlib as GL, xii_bayes, relpairs, metrics, xc
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)
out = dict(software=dict(python=platform.python_version(), jax=jax.__version__, jaxlib=__import__("jaxlib").__version__, numpy=np.__version__, optax=optax.__version__,
                         scipy=scipy.__version__, platform=platform.platform()), env={"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"})
out["external_python_modules"] = {n: dict(path=rel(m.__file__), sha256=sha(m.__file__)) for n, m in list(sys.modules.items())
                                  if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(ROOT) and not os.path.abspath(m.__file__).startswith(PKG)}
specs = set(); pat = re.compile(r"uP16\d:(?:s:\d+|fx:\d+:\d+:\d+(?::[A-Za-z0-9_.-]+)?)")
for f in glob.glob(os.path.join(PKG, "logs", "*.json")) + glob.glob(os.path.join(PKG, "logs", "*.jsonl")) + glob.glob(os.path.join(PKG, "logs", "*", "*.json")) + glob.glob(os.path.join(PKG, "logs", "queue_*.txt")):
    specs.update(pat.findall(open(f).read()))
parents = set(s.split(":")[0] for s in specs) | {"uP160", "uP161", "uP163", "uP164", "uP165"}
files = set(NV.parent_path(p) for p in parents)
for sp in sorted(specs):
    try: ud = xc.unit_dir(sp)
    except Exception: continue
    if os.path.abspath(ud).startswith(PKG): continue   # this package's own runs are in the archive
    files.update(glob.glob(os.path.join(ud, "step*.npz")) + glob.glob(os.path.join(ud, "*.json")))
files.update(glob.glob(os.path.join(NV.X13C, "checkpoints", "adapt3", "*.npz")))   # XIII-C adapted models read by c_carrier/adapt4 development
files.add(os.path.join(NV.X13B, "logs", "eval_conf13.npz"))
missing = [rel(f) for f in files if not os.path.exists(f)]
out["reference_units"] = sorted(specs); out["external_data_files"] = {rel(f): sha(f) for f in sorted(files) if os.path.exists(f)}
out["missing"] = missing; out["n_external_data_files"] = len(out["external_data_files"])
out["sibling_packages_required"] = sorted({p.split(os.sep)[0] for p in list(out["external_data_files"]) + [v["path"] for v in out["external_python_modules"].values()]})
json.dump(out, open(os.path.join(PKG, "logs", "DEPENDENCIES.json"), "w"), indent=1)
print(json.dumps(dict(software=out["software"], n_units=len(specs), n_files=out["n_external_data_files"], missing=missing, siblings=out["sibling_packages_required"]), indent=1))
