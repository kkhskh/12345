"""Check manifest.json against the files actually in this package.

Run from the package root:

    python scripts/verify_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.json"


def listed_files():
    payload = json.loads(MANIFEST.read_text())
    entries = payload["files"]
    return {entry["path"]: entry["sha256"] for entry in entries}


def on_disk():
    found = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "manifest.json":
            continue
        if "__pycache__" in rel or rel.endswith(".pyc") or ".pytest_cache/" in f"{rel}/" or rel.startswith(".pytest_cache/"):
            continue
        found.append(rel)
    return set(found)


def main():
    recorded = listed_files()
    present = on_disk()
    missing = sorted(set(recorded) - present)
    extra = sorted(present - set(recorded))
    bad_hash = []
    for rel, digest in sorted(recorded.items()):
        if "__pycache__" in rel or rel.endswith(".pyc") or rel.startswith(".pytest_cache/"):
            raise SystemExit(f"manifest lists a cache file: {rel}")
        path = ROOT / rel
        if not path.is_file():
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            bad_hash.append(rel)
    if missing or extra or bad_hash:
        print("missing", missing)
        print("extra", extra)
        print("bad_hash", bad_hash)
        return 1
    print(f"manifest ok: {len(recorded)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
