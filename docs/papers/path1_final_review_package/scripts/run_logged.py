"""Run one command and record it in logs/: command, cwd, environment, exit code, output, file hashes.

Usage (from the package root):
    python scripts/run_logged.py LABEL -- python scripts/verify_paper_claims.py

Writes logs/LABEL.txt (human-readable) and logs/LABEL.json (machine-readable). The JSON records
the SHA-256 of every package file before and after the command, and lists files the command
created, changed or deleted. The exit code of this script is the exit code of the command.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
LOGS = PKG / "logs"
ENV_KEYS = ("PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONSAFEPATH", "MPLBACKEND", "PATH", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV")
PACKAGES = ("numpy", "scipy", "matplotlib", "pytest")


def tree_hashes() -> dict[str, str]:
    out = {}
    for p in sorted(PKG.rglob("*")):
        rel = p.relative_to(PKG)
        if not p.is_file() or rel.parts[0] == "logs" or "__pycache__" in rel.parts or p.name == ".DS_Store":
            continue
        out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def versions() -> dict[str, str]:
    v = {}
    for name in PACKAGES:
        try:
            v[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            v[name] = "not installed"
    return v


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    label, cmd = argv[0], argv[2:]
    if cmd[0] == "python":
        cmd = [sys.executable] + cmd[1:]
    LOGS.mkdir(exist_ok=True)
    before = tree_hashes()
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)
    elapsed = time.time() - t0
    after = tree_hashes()
    rec = {
        "label": label,
        "command": cmd,
        "cwd": str(Path.cwd()),
        "package_root": str(PKG),
        "started_utc": started,
        "elapsed_seconds": round(elapsed, 2),
        "exit_code": proc.returncode,
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions(),
            "env": {k: os.environ.get(k) for k in ENV_KEYS},
        },
        "files_created": sorted(set(after) - set(before)),
        "files_changed": sorted(k for k in set(after) & set(before) if after[k] != before[k]),
        "files_deleted": sorted(set(before) - set(after)),
        "sha256_after": after,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    (LOGS / f"{label}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    with open(LOGS / f"{label}.txt", "w", encoding="utf-8") as fh:
        fh.write(f"command:   {' '.join(cmd)}\ncwd:       {rec['cwd']}\nstarted:   {started}\n")
        fh.write(f"python:    {sys.executable} ({rec['environment']['python_version']})\n")
        fh.write(f"packages:  {rec['environment']['packages']}\n")
        fh.write(f"env:       {rec['environment']['env']}\n")
        fh.write(f"exit code: {proc.returncode}\nelapsed:   {rec['elapsed_seconds']} s\n")
        fh.write(f"created:   {rec['files_created']}\nchanged:   {rec['files_changed']}\ndeleted:   {rec['files_deleted']}\n")
        fh.write("\n--- stdout ---\n" + proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    tail = (proc.stdout.strip().splitlines() or [""])[-1]
    print(f"[{label}] exit {proc.returncode} in {elapsed:.1f} s: {tail}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
