"""Write MANIFEST.json: sha256 of every package file except the manifest and logs."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
               and "logs" not in p.parts and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)
man = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root / "MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
print(len(man), "files")
