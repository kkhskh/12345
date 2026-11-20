"""Build the delivery ZIP.  python scripts/build_package.py OUT_DIR
1. regenerate protocol JSON and notebooks from their sources; 2. vendored-checkpoint manifest; 3. package manifest (sha256 of every
shipped file + source commit); 4. ZIP with top folder COR_Investigation_XX/ (excluded: RUNS/, dev/runs/ checkpoints, __pycache__);
5. zip self-test.  The freeze (protocols/FREEZE.json) must already exist: the build refuses otherwise."""
import os, sys, json, glob, zipfile, subprocess, time
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG); os.environ.setdefault("COR_DEVICE", "cpu")
from cor_research import env as E
out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
assert os.path.exists(os.path.join(PKG, "protocols", "FREEZE.json")), "freeze first (scripts/freeze.py) — the build ships a frozen protocol"
ok, bad = __import__("cor_research.jobs.runner", fromlist=["x"]).verify_freeze(PKG); assert ok, f"freeze mismatch: {bad[:10]}"
EXCL = ("RUNS/", "dev/runs/ft/", "__pycache__", ".pyc", "manifests/PACKAGE_MANIFEST.json", ".ipynb_checkpoints")
files = sorted(os.path.relpath(f, PKG) for f in glob.glob(os.path.join(PKG, "**", "*"), recursive=True) if os.path.isfile(f))
files = [f for f in files if not any(x in f for x in EXCL)]
ck = {f: E.sha256(os.path.join(PKG, f)) for f in files if f.startswith("cor_research/legacy/") and f.endswith((".npz", ".pkl")) or (f.startswith("cor_research/legacy/") and f.endswith("pre.json"))}
E.atomic_json(ck, os.path.join(PKG, "manifests", "CHECKPOINTS.json"))
try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PKG, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=PKG, capture_output=True, text=True).stdout.strip().splitlines()
except Exception: commit, dirty = None, None
files = sorted(set(files) | {"manifests/CHECKPOINTS.json"})
man = dict(built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source_commit=commit, package_subtree_uncommitted_changes=(len(dirty) if dirty is not None else None), freeze_sha256=E.read_json(os.path.join(PKG, "protocols", "FREEZE.json"))["freeze_sha256"],
           files={f: E.sha256(os.path.join(PKG, f)) for f in files})
E.atomic_json(man, os.path.join(PKG, "manifests", "PACKAGE_MANIFEST.json"))
z = os.path.join(out, "cor_inv20_colab_three_track_research.zip")
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files + ["manifests/PACKAGE_MANIFEST.json"]: zf.write(os.path.join(PKG, f), os.path.join("COR_Investigation_XX", f))
with zipfile.ZipFile(z) as zf: t = zf.testzip(); n = len(zf.namelist())
print(json.dumps(dict(zip=z, files=n, bytes=os.path.getsize(z), sha256=E.sha256(z), zip_test="OK" if t is None else t, source_commit=commit)))
