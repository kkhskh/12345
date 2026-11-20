"""Build the XIX archives (mandate §17).  Usage: build_archive19.py OUTDIR
1. MANIFEST.sha256 for the package (excluding resume.pkl, logs/claims, __pycache__, the manifest itself).
2. cor_inv19_training_state_causal_identification.zip: the whole package under
   cor_phase15/step1_training_state_causal_identification/ plus every external dependency in logs/DEPENDENCIES19.json under its own
   cor_phase15/ path (the archive is NOT standalone with respect to the parent pretraining history; it contains what XIX's code imports
   and reads).  Split into parts of <= 290 MB (.zip.part001 ...) with a parts manifest; the parts are concatenated back and verified.
3. cor_inv19_audit.zip: everything except checkpoints, plus the step-1211 checkpoints of the confirmation anchors (small audit archive).
Writes OUTDIR/ARCHIVE_PARTS19.json."""
import os, sys, json, hashlib, zipfile, glob
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "..")); X = os.path.abspath(os.path.join(R, "..")); OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
NAME = "step1_training_state_causal_identification"; PART = 290 * 1024 * 1024
def sha(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()
skip = lambda rel: rel.endswith("resume.pkl") or rel.startswith("logs/claims") or "__pycache__" in rel or rel == "MANIFEST.sha256" or rel.endswith(".tmp")
files = sorted(os.path.relpath(os.path.join(dp, f), R) for dp, _, fs in os.walk(R) for f in fs)
files = [f for f in files if not skip(f)]
with open(os.path.join(R, "MANIFEST.sha256"), "w") as m:
    for f in files: m.write(f"{sha(os.path.join(R, f))}  {f}\n")
files.append("MANIFEST.sha256"); deps = json.load(open(os.path.join(R, "logs", "DEPENDENCIES19.json")))["files"]
full = os.path.join(OUT, "cor_inv19_training_state_causal_identification.zip")
with zipfile.ZipFile(full, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
    for f in files: z.write(os.path.join(R, f), f"cor_phase15/{NAME}/{f}")
    for f in deps: z.write(os.path.join(X, f), f"cor_phase15/{f}")
parts = []; i = 0
with open(full, "rb") as src:
    while True:
        b = src.read(PART)
        if not b: break
        i += 1; p = f"{full}.part{i:03d}"; open(p, "wb").write(b); parts.append(dict(file=os.path.basename(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest()))
rebuilt = os.path.join(OUT, "rebuilt_check.zip")
with open(rebuilt, "wb") as o:
    for p in parts: o.write(open(os.path.join(OUT, p["file"]), "rb").read())
ok_rebuild = sha(rebuilt) == sha(full); ok_zip = zipfile.ZipFile(rebuilt).testzip() is None; os.remove(rebuilt)
audit = os.path.join(OUT, "cor_inv19_audit.zip")
keep = lambda f: not f.startswith("checkpoints/") or (f.endswith("step1211.npz") and ("_full" in f) and "82" in f.split("/")[2])
with zipfile.ZipFile(audit, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
    for f in files:
        if keep(f): z.write(os.path.join(R, f), f"cor_phase15/{NAME}/{f}")
info = dict(full=os.path.basename(full), full_bytes=os.path.getsize(full), full_sha256=sha(full), n_files=len(files), n_dependency_files=len(deps), parts=parts,
            reconstruction_verified=ok_rebuild, zip_test_ok=ok_zip, audit=os.path.basename(audit), audit_bytes=os.path.getsize(audit), audit_sha256=sha(audit),
            reassemble="cat cor_inv19_training_state_causal_identification.zip.part* > cor_inv19_training_state_causal_identification.zip && sha256sum it against full_sha256",
            standalone=False, note="contains the XIX package and every file XIX's code imports or reads from earlier COR packages; it does not contain the full earlier packages")
json.dump(info, open(os.path.join(OUT, "ARCHIVE_PARTS19.json"), "w"), indent=1); print(json.dumps({k: v for k, v in info.items() if k != "parts"}, indent=1), len(parts), "parts")
