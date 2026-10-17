"""08R1 evaluation helpers that do not load Othello-GPT.

Hook scoring, negative-control construction, prefix-overlap, and the
completion validator live here so they can be unit-tested without torch.
Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


STATUSES = (
    "PASS",
    "FAIL",
    "PENDING",
    "BLOCKED_MISSING_ARTIFACT",
    "NOT_APPLICABLE",
    "SYNTHETIC_UNIT_TEST",
)

REQUIRED_REAL_STAGES = (
    "checkpoint_identity",
    "hook_blocks.6.hook_resid_mid",
    "hook_blocks.6.hook_resid_post",
    "hook_blocks.7.hook_resid_post",
    "noop_identity",
    "final_resid_donor_logit",
    "mid_post_readback",
    "corrected_probability_metrics",
    "probe_consistency_08d",
    "wrong_donor_control",
    "wrong_square_control",
    "untrained_control",
    "bootstrap_cis",
)

REQUIRED_REAL_FILES = (
    "08r1_real_model_status.json",
    "08r1_real_hook_verification.csv",
    "08r1_real_mid_post_diagnostics.csv",
    "08r1_real_corrected_probability_metrics.csv",
    "08r1_real_negative_controls.csv",
    "08r1_real_consistency_checks.csv",
    "08r1_real_untrained_control.csv",
    "08r1_real_bootstrap_results.csv",
)

REQUIRED_SOURCE_MARKERS = (
    "hook_obj.name",
    "choose_wrong_donors",
    "wrong_square_linear_edit",
    "BLOCKED_MISSING_ARTIFACT",
    "evaluate_completion",
    "prefix_overlap_by_length",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_hook_event(
    expected_name: str,
    observed_names: list[str],
    fire_count: int,
    shape: tuple[int, ...] | None,
    expected_shape: tuple[int, ...] | None,
    readback,
    donor,
    other_tokens_unchanged: bool,
    atol: float = 1e-5,
) -> dict:
    """Pure check used by the real runner and by unit tests. Never always-true."""
    names = [str(n) for n in observed_names]
    name_match = bool(names) and all(n == expected_name for n in names)
    fired = int(fire_count) > 0
    shape_ok = shape is not None and expected_shape is not None and tuple(shape) == tuple(expected_shape)
    if readback is None or donor is None:
        readback_ok = False
    else:
        a = np.asarray(readback, dtype=np.float64)
        b = np.asarray(donor, dtype=np.float64)
        readback_ok = a.shape == b.shape and bool(np.allclose(a, b, atol=atol))
    ok = bool(fired and name_match and shape_ok and readback_ok and other_tokens_unchanged)
    reason = []
    if not fired:
        reason.append("hook_did_not_fire")
    if not name_match:
        reason.append(f"name_mismatch expected={expected_name} observed={names}")
    if not shape_ok:
        reason.append(f"shape_mismatch got={shape} expected={expected_shape}")
    if not readback_ok:
        reason.append("readback_ne_donor")
    if not other_tokens_unchanged:
        reason.append("unrelated_tokens_changed")
    return {
        "expected_hook": expected_name,
        "observed_hooks": "|".join(names) if names else "",
        "fire_count": int(fire_count),
        "name_match": int(name_match),
        "shape_ok": int(shape_ok),
        "readback_ok": int(readback_ok),
        "other_tokens_unchanged": int(bool(other_tokens_unchanged)),
        "ok": int(ok),
        "fail_reason": ";".join(reason),
        "csv_hook": names[0] if names else "",
        "validation_kind": "VERIFIED_ON_REAL_MODEL" if ok else "FAIL",
    }


def choose_wrong_donors(rows: list[dict], pairs: list[dict], seed: int) -> list[int | None]:
    """Pick a real other prefix: same player/occupancy bucket, different legal set."""
    rng = np.random.default_rng(int(seed))
    out: list[int | None] = []
    for p in pairs:
        rec = rows[p["recip"]]
        don = rows[p["donor"]]
        bucket = (int(rec["player"]), int(rec["occupied"]) // 4)
        rec_legal = set(rec["legal"])
        don_legal = set(don["legal"])
        cands = []
        for i, r in enumerate(rows):
            if i in (p["recip"], p["donor"]):
                continue
            if not r.get("legal") or r.get("terminal"):
                continue
            if (int(r["player"]), int(r["occupied"]) // 4) != bucket:
                continue
            legal = set(r["legal"])
            if legal == don_legal or legal == rec_legal:
                continue
            cands.append(i)
        if not cands:
            out.append(None)
        else:
            out.append(int(rng.choice(cands)))
    return out


def intended_disagreed_squares(rec_lab, don_lab) -> list[int]:
    rec_lab = np.asarray(rec_lab)
    don_lab = np.asarray(don_lab)
    return [int(j) for j in range(min(len(rec_lab), len(don_lab))) if rec_lab[j] != don_lab[j]]


def wrong_square_linear_edit(x, rec_lab, don_lab, models, budget: float, rng, center=(27, 28, 35, 36)):
    """Apply a matched-norm probe direction to a square that is not an intended change.

    This is a target-specificity control, not a claim that the edited board is a
    legal natural Othello state.
    """
    x = np.asarray(x, dtype=np.float64)
    rec_lab = np.asarray(rec_lab)
    don_lab = np.asarray(don_lab)
    intended = intended_disagreed_squares(rec_lab, don_lab)
    intended_set = set(intended)
    cands = [j for j in range(64) if j not in intended_set and j not in center]
    if not cands:
        return None
    j = int(np.asarray(cands)[int(rng.integers(0, len(cands)))])
    clf = models[j]
    if not hasattr(clf, "coef_"):
        return None
    coef = np.asarray(clf.coef_, dtype=np.float64)
    if coef.ndim != 2 or coef.shape[0] < 2:
        return None
    src = int(rec_lab[j])
    tgt = 0 if src != 0 else 1
    if tgt >= coef.shape[0]:
        tgt = int(np.argmax(np.arange(coef.shape[0]) != src))
    vec = coef[tgt] - coef[src]
    nrm = float(np.linalg.norm(vec))
    if nrm <= 1e-12:
        return None
    step = vec * (float(budget) / nrm)
    return {
        "edited": x + step,
        "wrong_square": j,
        "intended_squares": intended,
        "edit_norm": float(np.linalg.norm(step)),
        "natural_state": 0,
        "control_kind": "target_specificity",
    }


def prefix_overlap_by_length(
    holdout_seqs: list[list[int]],
    existing_seqs: list[list[int]],
    max_len: int = 20,
    reference: str = "existing_games",
) -> list[dict]:
    """Count holdout prefixes that also appear in previous games, by prefix length."""
    existing_at: dict[int, set[tuple[int, ...]]] = {}
    for seq in existing_seqs:
        upto = min(len(seq), max_len)
        for L in range(1, upto + 1):
            existing_at.setdefault(L, set()).add(tuple(seq[:L]))
    rows = []
    for L in range(1, max_len + 1):
        hold = [tuple(seq[:L]) for seq in holdout_seqs if len(seq) >= L]
        exist = existing_at.get(L, set())
        overlap = sum(1 for p in hold if p in exist)
        rows.append(
            {
                "prefix_length": L,
                "n_holdout_prefixes": len(hold),
                "n_overlap": int(overlap),
                "frac_overlap": float(overlap / len(hold)) if hold else float("nan"),
                "reference": reference,
                "note": "Shared openings are expected at small L. Long-prefix overlap is the leakage concern.",
                "validation_kind": "VERIFIED_FROM_REAL_SAVED_DATA",
                "used_in_method_selection": 0,
            }
        )
    return rows


def dist_stats(values) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


def record_hf_revision(repo: str) -> dict:
    out = {"hf_repo": repo, "hf_revision": None, "hf_revision_source": None}
    try:
        from huggingface_hub import model_info

        info = model_info(repo)
        out["hf_revision"] = getattr(info, "sha", None)
        out["hf_revision_source"] = "huggingface_hub.model_info"
    except Exception as exc:
        out["hf_revision_error"] = str(exc)
    return out


def is_scientific_pass(status: str) -> bool:
    return status == "PASS"


def evaluate_completion(
    stages: dict[str, str],
    file_ok: dict[str, bool],
    synthetic_counted: bool = False,
    numeric_ok: bool = True,
) -> dict:
    """Decide 08R1 status from explicit stage codes. PENDING/BLOCKED/SYNTHETIC are not passes."""
    missing = [s for s in REQUIRED_REAL_STAGES if s not in stages]
    required = {s: stages.get(s, "PENDING") for s in REQUIRED_REAL_STAGES}
    fails = [s for s, st in required.items() if st == "FAIL"]
    blocked = [s for s, st in required.items() if st == "BLOCKED_MISSING_ARTIFACT"]
    pending = [s for s, st in required.items() if st in {"PENDING", "SYNTHETIC_UNIT_TEST"}] + missing
    passed = [s for s, st in required.items() if st == "PASS"]
    files_missing = [k for k in REQUIRED_REAL_FILES if not file_ok.get(k)]
    if synthetic_counted:
        decision = "08R1_INCOMPLETE"
        reason = "synthetic rows were counted as scientific results"
    elif fails:
        decision = "REAL_MODEL_VALIDATION_FAILED"
        reason = "required stage FAIL: " + ",".join(fails)
    elif not numeric_ok:
        decision = "08R1_INCOMPLETE"
        reason = "required scientific CSVs lack real numeric results or contain synthetic rows"
    elif blocked or pending or files_missing:
        decision = "08R1_INCOMPLETE"
        reason = (
            "blocked=" + ",".join(blocked)
            + "; pending=" + ",".join(pending)
            + "; missing_files=" + ",".join(files_missing)
        )
    elif len(passed) == len(REQUIRED_REAL_STAGES) and not files_missing:
        decision = "08R1_COMPLETE"
        reason = "all required real-model stages PASS"
    else:
        decision = "08R1_INCOMPLETE"
        reason = "unclassified incomplete"
    return {
        "decision": decision,
        "reason": reason,
        "n_required": len(REQUIRED_REAL_STAGES),
        "n_pass": len(passed),
        "n_fail": len(fails),
        "n_blocked": len(blocked),
        "n_pending": len(pending),
        "official_0A_letter": "C / PARTIAL",
        "scientific_passes_only": [s for s in passed],
        "files_missing": files_missing,
    }


def local_source_readiness(runner_text: str, helper_text: str, tests_passed: bool) -> dict:
    """Static readiness: required scientific paths exist. Not 08R1_COMPLETE."""
    blob = runner_text + "\n" + helper_text
    missing = [m for m in REQUIRED_SOURCE_MARKERS if m not in blob]
    hardcoded = "row_labels_match_edited_sites" in runner_text and '"ok": 1' in runner_text
    # The old defect was a dict with check=row_labels... and ok: 1 hardcoded.
    old_hardcode = (
        'check": "row_labels_match_edited_sites"' in runner_text
        and runner_text.split('row_labels_match_edited_sites', 1)[-1][:400].count('"ok": 1')
    )
    if missing or old_hardcode or not tests_passed:
        return {
            "decision": "08R1_INCOMPLETE — IMPLEMENTATION_DEFECTS_REMAIN",
            "missing_markers": missing,
            "hardcoded_ok": bool(old_hardcode),
            "tests_passed": tests_passed,
            "official_0A_letter": "C / PARTIAL",
        }
    return {
        "decision": "READY_FOR_COLAB_REAL_RUN",
        "missing_markers": [],
        "hardcoded_ok": False,
        "tests_passed": tests_passed,
        "official_0A_letter": "C / PARTIAL",
        "note": "Local readiness only. 08R1_COMPLETE requires a successful real-model Colab run.",
    }


def nonempty_csv(path: Path, required_cols: Iterable[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    header = lines[0].split(",")
    return all(col in header for col in required_cols)


def inspect_real_csv_has_numeric(path: Path, col: str) -> bool:
    if not path.exists():
        return False
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return False
    if any(str(r.get("validation_kind", "")).startswith("SYNTHETIC") for r in rows):
        return False
    if any(r.get("provenance") == "synthetic" for r in rows):
        return False
    vals = []
    for r in rows:
        try:
            vals.append(float(r[col]))
        except (KeyError, TypeError, ValueError):
            continue
    return bool(vals) and any(v == v for v in vals)


def real_outputs_scientifically_usable(results_dir: Path) -> tuple[bool, str]:
    """Fail closed if a required real CSV is empty, synthetic, or missing numeric results."""
    checks = (
        ("08r1_real_corrected_probability_metrics.csv", "became_legal_delta_mass"),
        ("08r1_real_corrected_probability_metrics.csv", "total_legal_mass_target_after"),
        ("08r1_real_bootstrap_results.csv", "ci95_low"),
        ("08r1_real_bootstrap_results.csv", "ci95_high"),
        ("08r1_real_hook_verification.csv", "ok"),
        ("08r1_real_consistency_checks.csv", "mean"),
        ("08r1_real_negative_controls.csv", "natural_state"),
    )
    for name, col in checks:
        path = Path(results_dir) / name
        if not nonempty_csv(path, [col]):
            return False, f"{name} missing schema column {col}"
        if name.endswith("corrected_probability_metrics.csv") or name.endswith("bootstrap_results.csv"):
            if not inspect_real_csv_has_numeric(path, col):
                return False, f"{name} has no real numeric {col}"
    return True, "ok"
