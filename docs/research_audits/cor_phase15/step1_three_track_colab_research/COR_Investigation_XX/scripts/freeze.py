"""Freeze: hash every protocol file and every code file the confirmation depends on into protocols/FREEZE.json.
python scripts/freeze.py [--force].  Refuses to overwrite an existing freeze unless --force (a re-freeze is recorded in
protocols/FREEZE_HISTORY.json with both versions; the history is never edited)."""
import os, sys, json, glob, time, hashlib
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, PKG)
os.environ.setdefault("COR_DEVICE", "cpu")
from cor_research import env as E
from cor_research.scoring import measure as MS, trackb_run as TBR
def files():
    fs = sorted(glob.glob(os.path.join(PKG, "protocols", "*", "*.json"))) + sorted(glob.glob(os.path.join(PKG, "configurations", "*.json")))
    fs += sorted(f for f in glob.glob(os.path.join(PKG, "cor_research", "**", "*.py"), recursive=True))
    fs += sorted(glob.glob(os.path.join(PKG, "scripts", "*.py"))) + sorted(glob.glob(os.path.join(PKG, "tests", "**", "*.py"), recursive=True))
    fs += [os.path.join(PKG, f) for f in ("requirements.txt", "requirements-gpu.txt", os.path.join("protocols", "DESIGN_XX.md"))] + sorted(glob.glob(os.path.join(PKG, "tests", "fixtures", "*.json")))
    return [os.path.relpath(f, PKG) for f in fs if "__pycache__" not in f and not f.endswith("freeze.py")]
if __name__ == "__main__":
    out = os.path.join(PKG, "protocols", "FREEZE.json"); old = E.read_json(out)
    if old and "--force" not in sys.argv: raise SystemExit("protocols/FREEZE.json exists; use --force to record a re-freeze (both versions are kept)")
    fz = dict(frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), files={f: E.sha256(os.path.join(PKG, f)) for f in files()},
              measure_code_sha256=MS.code_hashes(), trackb_code_sha256=TBR.code_hashes())   # review C r2 N1: the FULL Track B code-hash set
    fz["freeze_sha256"] = hashlib.sha256(json.dumps({k: fz[k] for k in ("files", "measure_code_sha256", "trackb_code_sha256")}, sort_keys=True).encode()).hexdigest()
    if old:
        hp = os.path.join(PKG, "protocols", "FREEZE_HISTORY.json"); H = E.read_json(hp, []) or []; H.append(dict(replaced=old, by=fz["freeze_sha256"], at=fz["frozen_at_utc"])); E.atomic_json(H, hp)
    E.atomic_json(fz, out); print("frozen", len(fz["files"]), "files; freeze_sha256", fz["freeze_sha256"])
