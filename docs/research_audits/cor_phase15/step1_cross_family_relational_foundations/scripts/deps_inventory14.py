"""Dependency inventory for Investigation XIV -> logs/DEPENDENCIES.json.
(1) software versions; (2) Python modules imported from OUTSIDE this package (path + sha256), including the GATE modules of the atlas
continuation / deepening / latent-reliability packages; (3) every external data file read: uP160 parent, every checkpoint file of every
T4 unit referenced in this package's logs and queues that lives outside it (XIII-E / XIII-D / XIII-B), the XIII-B evaluation set, and the
three development GATE models.  Paths relative to cor_phase15/."""
import hashlib, json, os, sys, glob, platform, re
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..")); ROOT = os.path.abspath(os.path.join(PKG, ".."))
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, optax, scipy, sklearn
import native as NV, ftlib as F, genlib as GL, xii_bayes, relpairs, metrics, xc, maskfwd, gatelib
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)
out = dict(software=dict(python=platform.python_version(), jax=jax.__version__, jaxlib=__import__("jaxlib").__version__, numpy=np.__version__, optax=optax.__version__,
                         scipy=scipy.__version__, sklearn=sklearn.__version__, platform=platform.platform()), env={"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"})
out["external_python_modules"] = {n: dict(path=rel(m.__file__), sha256=sha(m.__file__)) for n, m in list(sys.modules.items())
                                  if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(ROOT) and not os.path.abspath(m.__file__).startswith(PKG)}
specs = set(); pat = re.compile(r"uP16\d:(?:s:\d+|fx:\d+:\d+:\d+(?::[A-Za-z0-9_.-]+)?)")
for f in glob.glob(os.path.join(PKG, "logs", "*.json")) + glob.glob(os.path.join(PKG, "logs", "*.jsonl")) + glob.glob(os.path.join(PKG, "logs", "queue_*.txt")):
    specs.update(pat.findall(open(f).read()))
# branch sources named in the training queue (FX_BRANCH=<tag>:<k>)
for l in open(os.path.join(PKG, "logs", "queue_I_train.txt")):
    for t in re.findall(r"FX_BRANCH=([A-Za-z0-9_.-]+):\d+", l):
        p = t.split("_"); specs.add(f"{p[0]}:fx:{p[2][1:]}:{p[3][1:]}:{p[4][1:]}" + (":" + "_".join(p[5:]) if len(p) > 5 else ""))
files = {NV.parent_path("uP160"), os.path.join(NV.X13B, "logs", "eval_conf13.npz")} | set(gatelib.DEV_FINALS.values())
for sp in sorted(specs):
    for d in (os.path.join(NV.X13E, "checkpoints", "ft"), os.path.join(NV.X13D, "checkpoints", "ft"), os.path.join(NV.X13C, "checkpoints", "ft")):
        p = sp.split(":"); tag = f"{p[0]}_fx_L{p[2]}_W{p[3]}_D{p[4]}" + (f"_{p[5]}" if len(p) > 5 else "") if p[1] == "fx" else None
        if tag and os.path.isdir(os.path.join(d, tag)):
            files.update(f for f in glob.glob(os.path.join(d, tag, "*")) if not f.endswith("resume.pkl")); break
missing = [rel(f) for f in files if not os.path.exists(f)]
out["reference_units"] = sorted(specs); out["external_data_files"] = {rel(f): sha(f) for f in sorted(files) if os.path.exists(f)}
out["missing"] = missing; out["n_external_data_files"] = len(out["external_data_files"])
out["sibling_packages_required"] = sorted({p.split(os.sep)[0] for p in list(out["external_data_files"]) + [v["path"] for v in out["external_python_modules"].values()]})
json.dump(out, open(os.path.join(PKG, "logs", "DEPENDENCIES.json"), "w"), indent=1)
print(json.dumps(dict(n_units=len(specs), n_files=out["n_external_data_files"], missing=missing, siblings=out["sibling_packages_required"]), indent=1))
