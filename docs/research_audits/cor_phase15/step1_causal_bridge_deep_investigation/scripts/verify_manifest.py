"""Verify MANIFEST.json against the files on disk. Exit 1 on any mismatch or missing file."""
import hashlib, json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
man = json.loads((root / "MANIFEST.json").read_text())
bad = [f for f, h in man.items() if not (root / f).is_file() or hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
print("manifest entries:", len(man), "mismatches:", len(bad))
for f in bad:
    print("MISMATCH", f)
sys.exit(1 if bad else 0)
