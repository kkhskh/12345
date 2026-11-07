"""Write manifest.json for the files currently in this package.

The manifest lists every file except itself. Cache files are omitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "manifest.json":
            continue
        if "__pycache__" in rel or rel.endswith(".pyc") or rel.startswith(".pytest_cache/"):
            continue
        files.append({
            "path": rel,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        })
    payload = {"files": files, "count": len(files)}
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(files)} entries")


if __name__ == "__main__":
    main()
