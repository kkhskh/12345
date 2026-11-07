"""Write MANIFEST.json: sha256 of every tracked file in the package (checkpoints excluded)."""
import hashlib, json, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
files = sorted(p for p in root.rglob("*") if p.is_file() and "checkpoints" not in p.parts and p.name != "MANIFEST.json"
               and not p.name.endswith(".stdout") and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
man = {"base_commit_at_manifest_time": head, "files": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(root / "MANIFEST.json").write_text(json.dumps(man, indent=1))
print(len(files), "files")
