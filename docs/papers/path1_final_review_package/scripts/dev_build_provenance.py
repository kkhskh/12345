"""Development-only: copy raw result files out of the original archives and record provenance.

This script is NOT needed to reproduce the paper from the release package. It documents how
data/raw/, data/protocols/ and data/archival/console_excerpts/ were populated. It opens the
original archives read-only (zipfile) and never writes next to them.

Usage (from the package root, inside the original workspace):
    python scripts/dev_build_provenance.py --downloads ~/Downloads --repo ../../.. \
        --transcript <session transcript .jsonl>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

ARCHIVES = {
    "fiber": "08r2_fiber_results.zip",
    "spec": "fiber_spec.zip",
    "fixed_js": "fiber_cost_fixed_js.zip",
    "nst": "null_state_transfer.zip",
}
SECONDARY_ARCHIVES = {
    "fixed_js_verified_csv": "fixed_js_verified_data.zip",
}

PROTOCOL_FILES = [
    "docs/experiments/08r2_fiber_spec_preregistration.md",
    "docs/experiments/08r2_fiber_cost_preregistration.md",
    "docs/experiments/08r2_fiber_cost_fixed_js_preregistration.md",
    "docs/experiments/08r2_null_state_transfer_preregistration.md",
    "docs/experiments/08r2_state_semantics_preregistration.md",
    "docs/experiments/08r2_state_semantics_mean_level_correction.md",
    "docs/experiments/08r2_symmetric_target_preregistration.md",
    "docs/experiments/08r2_symmetric_target_census_audit.md",
    "docs/experiments/08r2_symmetric_target_fresh_preregistration.md",
    "docs/experiments/08r2_visible_null_factorial_preregistration.md",
    "configs/toy_othello_08r2_fiber.json",
    "configs/toy_othello_08r2_fiber_spec.json",
    "configs/toy_othello_08r2_fiber_cost_fixed_js.json",
    "configs/toy_othello_08r2_null_state_transfer.json",
    "configs/toy_othello_08r2_state_semantics.json",
    "configs/toy_othello_08r2_symmetric_target_fresh.json",
    "configs/toy_othello_08r2_visible_null_factorial.json",
]

# (excerpt file, transcript line index, start marker, end marker or None, description)
CONSOLE_EXCERPTS = [
    ("sem_full_run.txt", 1549, "08R2-STATE-SEMANTICS\n  verdict=LEGAL_MASK", "08R2-STATE-SEMANTICS STAGE 6", "STATE-SEMANTICS full 48-pair run (Colab console)"),
    ("sem_stage_share_subgroups.txt", 1583, "08R2-SYMMETRIC-TARGET STAGE 0 CORRECTION", "08R2-SYMMETRIC-TARGET STAGE 1", "Stage-share subgroup correction (Colab console)"),
    ("symmetric_target_census.txt", 1583, "08R2-SYMMETRIC-TARGET STAGE 2 CENSUS", None, "Leftover-validation symmetric-target census (Colab console)"),
    ("symmetric_target_census_audit.txt", 1614, "08R2-SYMMETRIC-TARGET CENSUS AUDIT", None, "Census audit (Colab console)"),
    ("fresh_cohort.txt", 1633, "08R2-SYMMETRIC-TARGET-FRESH STAGE 1", None, "FRESH cohort generation and triplet census (Colab console)"),
    ("fresh_full_run.txt", 1637, "08R2-SYMMETRIC-TARGET-FRESH\n  verdict=SYMMETRIC_TARGET_SWITCHING", None, "FRESH 32-triplet run (Colab console)"),
    ("fresh_anchor_audit.txt", 1656, "08R2-ANCHOR-INTERACTION-AUDIT", None, "FRESH anchor-interaction audit (Colab console)"),
    ("vnf_stage1.txt", 1684, "08R2-VISIBLE-NULL-FACTORIAL STAGE 1", "Traceback", "VNF stage 1 summary before an unrelated serialization crash (Colab console)"),
    ("vnf_cohort.txt", 1695, "08R2-VISIBLE-NULL-FACTORIAL STAGE 2", None, "VNF cohort (Colab console)"),
    ("vnf_rowspace_map.txt", 1700, "08R2-VISIBLE-NULL-FACTORIAL STAGE 3", "Moving model", "VNF rowspace ridge fit (Colab console)"),
    ("vnf_full_run.txt", 1703, "08R2-VISIBLE-NULL-FACTORIAL\n  verdict=TARGET_SPECIFIC", None, "VNF 32-triplet run (Colab console)"),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def transcript_text(lines: list[str], i: int) -> str:
    o = json.loads(lines[i])
    m = o.get("message", o)
    c = m.get("content")
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return str(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--transcript", required=True)
    a = ap.parse_args()
    downloads = Path(a.downloads).expanduser().resolve()
    repo = Path(a.repo).expanduser().resolve()
    transcript = Path(a.transcript).expanduser().resolve()

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Originals opened read-only. smoke/ members excluded. No checkpoint, probe weights, or sealed confirmation data are included.",
        "archives": {},
        "secondary_archives": {},
        "protocols": [],
        "console_excerpts": [],
        "not_found": [],
    }
    for study, name in ARCHIVES.items():
        src = downloads / name
        if not src.is_file():
            raise SystemExit(f"missing original archive: {src}")
        entry = {"original_path": str(src), "archive_sha256": sha256_file(src), "archive_bytes": src.stat().st_size, "members": []}
        dest_root = PKG / "data" / "raw" / study
        with zipfile.ZipFile(src) as zf:
            for info in zf.infolist():
                if info.is_dir() or "/smoke/" in "/" + info.filename or info.filename.startswith("smoke/"):
                    continue
                data = zf.read(info)
                rel = Path(info.filename).name
                out = dest_root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
                entry["members"].append({
                    "member": info.filename,
                    "package_path": str(out.relative_to(PKG)),
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                    "crc32_archive": f"{info.CRC:08x}",
                })
        manifest["archives"][study] = entry
    for key, name in SECONDARY_ARCHIVES.items():
        src = downloads / name
        if src.is_file():
            with zipfile.ZipFile(src) as zf:
                members = [{"member": i.filename, "bytes": i.file_size, "sha256": sha256_bytes(zf.read(i))} for i in zf.infolist() if not i.is_dir()]
            manifest["secondary_archives"][key] = {
                "original_path": str(src), "archive_sha256": sha256_file(src), "members": members,
                "use": "Earlier exported verification table. Superseded by recomputation from data/raw/fixed_js; not shipped.",
            }

    for rel in PROTOCOL_FILES:
        src = repo / rel
        if not src.is_file():
            manifest["not_found"].append({"item": rel, "searched": str(src)})
            continue
        out = PKG / "data" / "protocols" / Path(rel).name
        out.write_bytes(src.read_bytes())
        try:
            blob = subprocess.run(["git", "-C", str(repo), "hash-object", rel], capture_output=True, text=True, check=True).stdout.strip()
            tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel], capture_output=True, text=True).returncode == 0
        except Exception as exc:  # git missing is a provenance gap, recorded explicitly
            blob, tracked = f"git_error:{exc}", False
        manifest["protocols"].append({"repo_path": rel, "package_path": str(out.relative_to(PKG)), "sha256": sha256_file(out), "git_blob": blob, "git_tracked": bool(tracked)})

    lines = transcript.read_text(encoding="utf-8").splitlines()
    for fname, idx, start, end, desc in CONSOLE_EXCERPTS:
        text = transcript_text(lines, idx)
        s = text.find(start)
        if s < 0:
            raise SystemExit(f"excerpt start marker not found for {fname} at line {idx}")
        e = text.find(end, s + len(start)) if end else -1
        chunk = text[s:e if e > 0 else len(text)]
        chunk = chunk.split("</user_query>")[0].rstrip() + "\n"
        out = PKG / "data" / "archival" / "console_excerpts" / fname
        out.write_text(chunk, encoding="utf-8")
        manifest["console_excerpts"].append({
            "package_path": str(out.relative_to(PKG)), "sha256": sha256_file(out), "description": desc,
            "source": "verbatim Colab console output pasted by the experimenter into the research session transcript",
            "transcript_file": transcript.name, "transcript_line_index": idx,
        })

    for study, pattern in [
        ("STATE-SEMANTICS (SEM)", "results/08r2/state_semantics/*"),
        ("SYMMETRIC-TARGET-FRESH (FRESH)", "results/08r2/symmetric_target_fresh/*"),
        ("VISIBLE-NULL-FACTORIAL (VNF)", "results/08r2/visible_null_factorial/*"),
        ("08R2 validation pair tables", "results/08r2/validation/*"),
    ]:
        manifest["not_found"].append({
            "item": study, "expected": pattern,
            "searched": ["~/Downloads (all *.zip listed and checked by name)", "~/Desktop", "~/Documents", "~/Library/CloudStorage/Dropbox", "repository artifacts/", "system-wide find on -xdev for state_semantic*, anchor_interaction*, symmetric_target*, visible_null*"],
            "result": "No result arrays found. Only runner scripts exist locally. No Google Drive mount is present on this machine.",
        })

    (PKG / "data" / "provenance_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    n = sum(len(v["members"]) for v in manifest["archives"].values())
    print(f"copied {n} raw members, {len(manifest['protocols'])} protocol files, {len(manifest['console_excerpts'])} console excerpts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
