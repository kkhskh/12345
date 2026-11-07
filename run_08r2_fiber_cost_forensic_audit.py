"""Read-only forensic audit of 08R2-FIBER-COST protocol compliance.

Does not load Othello-GPT, the probe, or the confirmation split.
Does not overwrite 08r2/fiber_cost artifacts.
Does not change COST scientific results.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.othello_r2 import OFFICIAL_0A_LETTER
from src.othello_r2_fiber import js_divergence, softmax
from src.othello_r2_fiber_spec import slice_boot
from src.othello_runutil import write_json


FORENSIC_DIRNAME = "fiber_cost_forensic"
FIRST_DEVIATION_COMMIT = "63433f2b217fe1ca413bbe103e07ab9b506d794e"
FIXED_JS_TARGET = 0.01
FIXED_JS_TOL = max(0.0001, 0.02 * FIXED_JS_TARGET)
NOT_ID = "NOT_IDENTIFIABLE_FROM_SAVED_DATA"
VERDICT_DEVIATION = "FROZEN_COST_PROTOCOL_DEVIATION_CONFIRMED"
SEVEN = ("correct_null", "wrong_null_0", "wrong_null_1", "wrong_null_2", "random_null_0", "random_null_1", "random_null_2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only 08R2-FIBER-COST forensic audit.")
    p.add_argument("--cost-dir", required=True, help="Existing 08r2/fiber_cost directory.")
    p.add_argument("--out-dir", default=None, help="Defaults to sibling fiber_cost_forensic/.")
    return p.parse_args()


def isolate_forensic_dir(cost_dir: Path, explicit: str | None) -> Path:
    if explicit:
        out = Path(explicit)
    elif cost_dir.name == "fiber_cost" and cost_dir.parent.name == "08r2":
        out = cost_dir.parent / FORENSIC_DIRNAME
    else:
        out = cost_dir.parent / FORENSIC_DIRNAME
    if "fiber_cost" in out.parts and out.name == "fiber_cost":
        raise SystemExit("forensic audit refuses to write inside the original fiber_cost directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value, default=float("nan")) -> float:
    try:
        if value in (None, "", NOT_ID):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def git_trace() -> str:
    lines = [
        "08R2-FIBER-COST commit trace",
        "",
        "First COST implementation commit:",
        FIRST_DEVIATION_COMMIT,
        "  Add 08R2-FIBER-COST to test JS-matched target specificity of probe-null residuals.",
        "",
        "That commit introduced Experiment B, not a fixed JS=0.01 design:",
        "  prereg step 2: t = JS(Editor A, unscaled correct-null)",
        "  prereg step 3: scale only wrong/random directions to t",
        "  config js_atol=0.01 is a matching tolerance, not a JS target",
        "  correct_null rows are written with reason=reference and alpha=||P_N(h_D-h_E)||",
        "",
        "Later commits did not change the estimand:",
        "  1a3e1e2 path discovery",
        "  966addd reconstruct frozen pairs if SPEC folder missing",
        "  ddceca1 incomplete JS matching is INCONCLUSIVE, not a missing-file failure",
        "",
    ]
    try:
        out = subprocess.check_output(
            ["git", "log", "--oneline", "--", "docs/experiments/08r2_fiber_cost_preregistration.md", "run_08r2_fiber_cost.py", "src/othello_r2_fiber_cost.py", "configs/toy_othello_08r2_fiber_cost.json"],
            text=True,
        )
        lines.extend(["git log --oneline (COST files):", out.rstrip(), ""])
    except Exception as exc:
        lines.append(f"git log unavailable: {exc}")
    return "\n".join(lines) + "\n"


def direction_type(cell: str) -> str:
    if cell == "correct_null":
        return "correct"
    if cell.startswith("wrong_null_"):
        return "wrong"
    if cell.startswith("random_null_"):
        return "random"
    return cell


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cost_dir = Path(args.cost_dir)
    if not cost_dir.is_dir():
        raise SystemExit(f"missing COST directory {cost_dir}")
    if cost_dir.name != "fiber_cost":
        raise SystemExit(f"refusing to audit a directory not named fiber_cost: {cost_dir}")
    out_dir = isolate_forensic_dir(cost_dir, args.out_dir)

    match_rows = read_csv(cost_dir / "fiber_cost_js_match.csv")
    inter_rows = read_csv(cost_dir / "fiber_cost_interventions.csv")
    dose_rows = read_csv(cost_dir / "fiber_cost_dose_search.csv") if (cost_dir / "fiber_cost_dose_search.csv").is_file() else []
    status = json.loads((cost_dir / "fiber_cost_completion_status.json").read_text(encoding="utf-8"))
    attempts = json.loads((cost_dir / "fiber_cost_attempt_accounting.json").read_text(encoding="utf-8"))
    manifest = json.loads((cost_dir / "fiber_cost_pair_manifest.json").read_text(encoding="utf-8"))
    prereg = json.loads((cost_dir / "fiber_cost_preregistration.json").read_text(encoding="utf-8")) if (cost_dir / "fiber_cost_preregistration.json").is_file() else {}

    pairs = manifest.get("pairs") or []
    pair_meta = {}
    for p in pairs:
        pair_meta[(p["recip_game"], p["target_game"])] = p
    wrong_by_pair = {}
    for p, w in zip(pairs, manifest.get("wrong_donors") or []):
        wrong_by_pair[(p["recip_game"], p["target_game"])] = [d.get("game_id") for d in w.get("donors", [])]

    inter_ix = {}
    cat_ix = {}
    for row in inter_rows:
        key = (row.get("recip_game"), row.get("target_game"), row.get("cell"))
        inter_ix[key] = row
        if row.get("r2_type"):
            cat_ix[(row.get("recip_game"), row.get("target_game"))] = row.get("r2_type")

    max_scale = defaultdict(lambda: float("nan"))
    for row in dose_rows:
        key = (row.get("recip_game"), row.get("target_game"), row.get("cell"))
        a = fnum(row.get("alpha"))
        if a == a:
            prev = max_scale[key]
            max_scale[key] = a if prev != prev else max(prev, a)

    logits = {}
    logit_path = cost_dir / "fiber_cost_full_logits.npz"
    if logit_path.is_file():
        pack = np.load(logit_path)
        logits = {k: np.asarray(pack[k], dtype=np.float64) for k in pack.files}

    match_ix = {(r.get("recip_game"), r.get("target_game"), r.get("cell")): r for r in match_rows}

    audit_rows = []
    js_rows = []
    js_discrepancies = []
    n_fixed_window = 0
    n_recomputed = 0
    n_logits_missing = 0

    for p in pairs:
        rg, tg = p["recip_game"], p["target_game"]
        cat = cat_ix.get((rg, tg)) or p.get("r2_type")
        donors = ["correct"] + (wrong_by_pair.get((rg, tg)) or [None, None, None]) + ["random", "random", "random"]
        for cell, donor in zip(SEVEN, donors):
            m = match_ix.get((rg, tg, cell), {})
            iv = inter_ix.get((rg, tg, cell), {})
            actual_target = fnum(m.get("target_js"), fnum(iv.get("target_js")))
            achieved = fnum(m.get("js"), fnum(iv.get("js_to_editor")))
            alpha = fnum(m.get("alpha"), fnum(iv.get("alpha")))
            scaled_l2 = fnum(m.get("l2"), fnum(iv.get("l2_to_editor")))
            if cell == "correct_null":
                original_l2 = scaled_l2 if scaled_l2 == scaled_l2 else alpha
                scale_note = "unscaled_spec_displacement"
            else:
                original_l2 = NOT_ID
                scale_note = "unit_direction_times_alpha"
            reason = m.get("reason") or iv.get("reason") or NOT_ID
            status_cal = "ok" if str(m.get("ok", iv.get("valid", ""))) in {"1", "1.0"} or reason == "reference" else str(reason)
            abs_err_var = abs(achieved - actual_target) if achieved == achieved and actual_target == actual_target else float("nan")
            rel_err_var = abs_err_var / actual_target if abs_err_var == abs_err_var and actual_target not in (0, float("nan")) else float("nan")
            abs_err_fixed = abs(achieved - FIXED_JS_TARGET) if achieved == achieved else float("nan")
            in_fixed = int(abs_err_fixed <= FIXED_JS_TOL) if abs_err_fixed == abs_err_fixed else 0
            n_fixed_window += in_fixed
            audit_rows.append(
                {
                    "pair_id": f"{rg}|{tg}",
                    "recip_game": rg,
                    "target_game": tg,
                    "category": cat,
                    "direction_type": direction_type(cell),
                    "cell": cell,
                    "donor_id": donor if donor is not None else NOT_ID,
                    "frozen_fixed_js_target": FIXED_JS_TARGET,
                    "actual_calibration_target_js": actual_target if actual_target == actual_target else NOT_ID,
                    "original_direction_l2": original_l2,
                    "selected_intervention_scale": alpha if alpha == alpha else NOT_ID,
                    "actual_scaled_intervention_l2": scaled_l2 if scaled_l2 == scaled_l2 else NOT_ID,
                    "achieved_js_saved": achieved if achieved == achieved else NOT_ID,
                    "abs_js_error_vs_actual_target": abs_err_var if abs_err_var == abs_err_var else NOT_ID,
                    "rel_js_error_vs_actual_target": rel_err_var if rel_err_var == rel_err_var else NOT_ID,
                    "abs_js_error_vs_fixed_0.01": abs_err_fixed if abs_err_fixed == abs_err_fixed else NOT_ID,
                    "in_fixed_0.01_window": in_fixed,
                    "calibration_status": status_cal,
                    "failure_reason": reason,
                    "scale_meaning": scale_note,
                    "max_tested_scale": max_scale[(rg, tg, cell)] if max_scale[(rg, tg, cell)] == max_scale[(rg, tg, cell)] else (NOT_ID if cell != "correct_null" else alpha),
                    "l2_to_editor_intervention": fnum(iv.get("l2_to_editor")) if iv else NOT_ID,
                }
            )

            ed_key = f"{rg}__{tg}__editor_A"
            iv_key = f"{rg}__{tg}__{cell}"
            recomputed = float("nan")
            disc = float("nan")
            ident = "recomputed"
            if ed_key not in logits or iv_key not in logits:
                ident = NOT_ID
                n_logits_missing += 1
            else:
                recomputed = float(js_divergence(softmax(logits[ed_key]), softmax(logits[iv_key])))
                n_recomputed += 1
                if achieved == achieved:
                    disc = abs(recomputed - achieved)
                    js_discrepancies.append(disc)
            js_rows.append(
                {
                    "pair_id": f"{rg}|{tg}",
                    "cell": cell,
                    "direction_type": direction_type(cell),
                    "log_convention": "natural_log_kl_row",
                    "normalization": "shift_1e-12_then_renormalize",
                    "saved_js": achieved if achieved == achieved else NOT_ID,
                    "recomputed_js": recomputed if recomputed == recomputed else NOT_ID,
                    "abs_discrepancy": disc if disc == disc else NOT_ID,
                    "actual_calibration_target_js": actual_target if actual_target == actual_target else NOT_ID,
                    "frozen_fixed_js_target": FIXED_JS_TARGET,
                    "in_fixed_0.01_window_saved": in_fixed,
                    "in_fixed_0.01_window_recomputed": int(abs(recomputed - FIXED_JS_TARGET) <= FIXED_JS_TOL) if recomputed == recomputed else NOT_ID,
                    "identifiability": ident,
                }
            )

    contrasts = [r for r in inter_rows if r.get("cell") == "pair_contrast" and str(r.get("valid")) in {"1", "1.0"}]
    complete_ids = {(r["recip_game"], r["target_game"]) for r in contrasts}
    failed_match = [r for r in match_rows if r.get("cell") != "correct_null" and str(r.get("reason")) not in {"ok", "reference"}]
    fail_pairs = sorted({(r["recip_game"], r["target_game"]) for r in failed_match})
    incomplete_ids = [pid for pid in [(p["recip_game"], p["target_game"]) for p in pairs] if pid not in complete_ids]

    fail_acct = []
    by_pair_fail = Counter((r["recip_game"], r["target_game"]) for r in failed_match)
    for r in failed_match:
        rg, tg = r["recip_game"], r["target_game"]
        fail_acct.append(
            {
                "recip_game": rg,
                "target_game": tg,
                "category": cat_ix.get((rg, tg)) or pair_meta.get((rg, tg), {}).get("r2_type"),
                "cell": r.get("cell"),
                "direction_type": direction_type(r.get("cell", "")),
                "reason": r.get("reason"),
                "actual_calibration_target_js": r.get("target_js"),
                "achieved_js": r.get("js"),
                "selected_or_max_scale": r.get("alpha"),
                "n_failed_directions_in_pair": by_pair_fail[(rg, tg)],
                "pair_complete": int((rg, tg) in complete_ids),
            }
        )

    def pair_target_js(pid):
        m = match_ix.get((pid[0], pid[1], "correct_null"), {})
        return fnum(m.get("target_js"))

    def pair_l2(pid):
        m = match_ix.get((pid[0], pid[1], "correct_null"), {})
        return fnum(m.get("l2"))

    complete_js = [pair_target_js(pid) for pid in complete_ids]
    incomplete_js = [pair_target_js(pid) for pid in incomplete_ids]
    complete_l2 = [pair_l2(pid) for pid in complete_ids]
    incomplete_l2 = [pair_l2(pid) for pid in incomplete_ids]
    n_incomp_a = sum(1 for pid in incomplete_ids if (cat_ix.get(pid) or pair_meta.get(pid, {}).get("r2_type")) == "A_legal_diff")
    n_incomp_b = len(incomplete_ids) - n_incomp_a
    n_comp_a = sum(1 for pid in complete_ids if (cat_ix.get(pid) or pair_meta.get(pid, {}).get("r2_type")) == "A_legal_diff")

    def slice_rows(rows, mask=None):
        g = rows if mask is None else [r for r in rows if r.get("r2_type") == mask]
        if not g:
            return {"n": 0}
        sl = {
            "n": len(g),
            "mean_delta_specific": float(np.nanmean([fnum(r["delta_specific"]) for r in g])),
            "mean_delta_random": float(np.nanmean([fnum(r["delta_random"]) for r in g])),
            "mean_dG": float(np.nanmean([fnum(r.get("dG_wrong")) for r in g])),
            "mean_dQ": float(np.nanmean([fnum(r.get("dQ_wrong")) for r in g])),
            "mean_l2_correct": float(np.nanmean([fnum(r.get("l2_correct")) for r in g])),
            "mean_js_correct": float(np.nanmean([fnum(r.get("js_correct_vs_editor")) for r in g])),
            "frac_correct_wins_wrong": float(np.mean([fnum(r["delta_specific"]) > 0 for r in g])),
            "frac_correct_wins_random": float(np.mean([fnum(r["delta_random"]) > 0 for r in g])),
        }
        sl["delta_specific_ci"] = slice_boot([fnum(r["delta_specific"]) for r in g], [r["recip_game"] for r in g], 2000, 82)
        sl["delta_random_ci"] = slice_boot([fnum(r["delta_random"]) for r in g], [r["recip_game"] for r in g], 2000, 83)
        return sl

    subset = {
        "pooled": slice_rows(contrasts),
        "A": slice_rows(contrasts, "A_legal_diff"),
        "B": slice_rows(contrasts, "B_similar_board_legal_change"),
        "condition": "conditional_on_JS_matching_feasibility_n35_of_48",
        "coverage": len(complete_ids) / 48 if pairs else float("nan"),
        "coverage_gate_90_passed": 0,
    }

    disc = np.asarray(js_discrepancies, dtype=np.float64) if js_discrepancies else np.array([])
    js_summary = {
        "n_direction_records": len(audit_rows),
        "n_recomputed_from_logits": n_recomputed,
        "n_logits_missing": n_logits_missing,
        "log_convention": "natural_log; JS=0.5 KL(p||m)+0.5 KL(q||m); m=0.5(p+q); +1e-12 then renormalize",
        "max_abs_discrepancy": float(np.max(disc)) if disc.size else NOT_ID,
        "mean_abs_discrepancy": float(np.mean(disc)) if disc.size else NOT_ID,
        "p50_abs_discrepancy": float(np.median(disc)) if disc.size else NOT_ID,
        "p95_abs_discrepancy": float(np.quantile(disc, 0.95)) if disc.size else NOT_ID,
        "n_in_fixed_0.01_window_saved": n_fixed_window,
        "fixed_js_target": FIXED_JS_TARGET,
        "fixed_js_tolerance": FIXED_JS_TOL,
    }

    fail_by_cell = Counter(r["cell"] for r in failed_match)
    fail_by_reason = Counter(r.get("reason") for r in failed_match)
    fail_by_type = Counter(direction_type(r["cell"]) for r in failed_match)
    fail_mult = Counter(by_pair_fail.values())

    (out_dir / "fiber_cost_commit_trace.txt").write_text(git_trace(), encoding="utf-8")
    write_csv(out_dir / "fiber_cost_calibration_audit.csv", audit_rows)
    write_csv(out_dir / "fiber_cost_js_recomputation.csv", js_rows)
    write_csv(out_dir / "fiber_cost_failure_accounting.csv", fail_acct)

    report = [
        "# 08R2-FIBER-COST Forensic Protocol-Compliance Report",
        "",
        "Read-only. No model execution. Original `fiber_cost/` artifacts were not overwritten.",
        "",
        "## Official letters preserved",
        "",
        "- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`",
        "- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`",
        "- `08R2_VALIDATION_COMPLETE — NEGATIVE`",
        f"- `0A {OFFICIAL_0A_LETTER}`",
        "- `SECTION_5_3 = PARTIAL`",
        "- `OBSTRUCTION_READY = NO`",
        "",
        f"Forensic verdict: `{VERDICT_DEVIATION}`",
        "",
        "## 1. Critical protocol answers",
        "",
        "A. JS target for each pair: the observed `JS(Editor A, unscaled correct-null)`. It is pair-specific.",
        "B. It is not 0.01. Config `js_atol=0.01` is a matching tolerance around that pair-specific t. Repo prereg: `|JS-t| ≤ max(0.01, 0.05 t)`.",
        "C. Exactly 48 rows are `reference` because each pair writes the correct donor once and does not search its scale.",
        "D. Correct-donor scale is `α = ||P_N(h_D − h_E)||`, the original SPEC displacement. No bisection.",
        "E. `mean_l2_correct` is that unscaled correct-null residual L2, not a JS-calibrated scale.",
        "F. No. Only the six wrong/random directions are calibrated. The correct donor is the reference.",
        f"G. First commit introducing this design: `{FIRST_DEVIATION_COMMIT}`.",
        "",
        "The requested frozen fixed-JS protocol (all seven directions to 0.01, first scale reaching the target, 90% complete-pair coverage, secondary 0.005/0.02) was not implemented.",
        "The in-repo preregistration was authored in the same commit as the runner and encodes Experiment B. That document is not independent evidence of a 0.01 protocol.",
        "",
        "## 6. Which experiment was executed",
        "",
        "Executed: **Experiment B — correct-donor-reference design**.",
        "Not executed: **Experiment A — frozen fixed-JS 0.01**.",
        "",
        "These are different estimands. The finished COST run is an exploratory output-JS-reference analysis.",
        "It must not be relabeled as the preregistered fixed-JS study.",
        "",
        "## 3. Independent JS recomputation",
        "",
        f"- Direction records: {len(audit_rows)} (expected 336).",
        f"- Recomputed from saved 61-way logits vs Editor A: {n_recomputed}.",
        f"- Missing logits (failed controls are not patched/saved): {n_logits_missing}.",
        f"- Convention: natural-log KL; JS = 0.5 KL(p||m)+0.5 KL(q||m); +1e-12 then renormalize.",
        f"- Max |saved−recomputed|: {js_summary['max_abs_discrepancy']}.",
        f"- Mean |saved−recomputed|: {js_summary['mean_abs_discrepancy']}.",
        f"- Directions in the requested window JS=0.01±{FIXED_JS_TOL}: {n_fixed_window} / {len(audit_rows)}.",
        "",
        "Equality to the pair-specific donor-reference JS is not equality to 0.01.",
        "",
        "## 4. Missingness",
        "",
        f"- Attempted pairs: {attempts.get('pairs')}.",
        f"- Complete pairs: {attempts.get('valid')} / {len(complete_ids)} contrast rows.",
        f"- Incomplete pairs: {len(incomplete_ids)}.",
        f"- Direction-level JS failures: {len(failed_match)} (reasons {dict(fail_by_reason)}).",
        f"- Failures by type: {dict(fail_by_type)}.",
        f"- Failures by cell: {dict(fail_by_cell)}.",
        f"- Failures per incomplete pair, multiplicity: {dict(fail_mult)}.",
        f"- Incomplete A/B: A={n_incomp_a} B={n_incomp_b}. Complete A={n_comp_a} B={len(complete_ids)-n_comp_a}.",
        f"- Mean donor-reference JS complete vs incomplete: {float(np.nanmean(complete_js)) if complete_js else NOT_ID} vs {float(np.nanmean(incomplete_js)) if incomplete_js else NOT_ID}.",
        f"- Mean unscaled correct L2 complete vs incomplete: {float(np.nanmean(complete_l2)) if complete_l2 else NOT_ID} vs {float(np.nanmean(incomplete_l2)) if incomplete_l2 else NOT_ID}.",
        "",
        "24 direction failures are not 24 failed pairs. They sit inside 13 incomplete pairs. A pair is incomplete if any of its six controls fail.",
        "Correct donors have zero calibration failures; all 48 are references.",
        "",
        "## 5. 35-pair complete-case contrasts (exploratory, feasibility-conditional)",
        "",
        f"- Coverage: {len(complete_ids)}/48 = {subset['coverage']:.4f}. 90% coverage gate: FAILED.",
        f"- Pooled Δ_JS: {subset['pooled']}",
        f"- A: {subset['A']}",
        f"- B: {subset['B']}",
        "",
        "These CIs are conditional on matching feasibility. They are not unbiased for the original 48-pair population.",
        "They do not pass the requested 90% coverage gate. They do not convert COST to SUPPORTED.",
        "",
        "## 7. What survives",
        "",
        "Survives:",
        "- SPEC L2-matched result `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`.",
        "- COST official letter `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`.",
        "- Engineering identity: frozen pair hash, Editor A budget, invariance, no confirmation access.",
        "- Descriptive fact that 35 pairs were matched to the correct donor's observed JS, not to 0.01.",
        "",
        "Does not survive as a preregistered claim:",
        "- Any statement that COST tested equal disruption at JS=0.01.",
        "- Any statement that all seven directions were calibrated identically.",
        "- Any use of `mean_l2_correct≈18.16` as a JS-calibrated dose.",
        "- Promotion of the 35-pair Δ_JS≈0.28 to a supported equal-JS effect.",
        "",
        f"Saved prereg keys from the run: {prereg}",
        "",
    ]
    (out_dir / "fiber_cost_forensic_protocol_report.md").write_text("\n".join(report), encoding="utf-8")
    write_json(
        out_dir / "fiber_cost_forensic_status.json",
        {
            "status": "FORENSIC_AUDIT_COMPLETE",
            "verdict": VERDICT_DEVIATION,
            "executed_experiment": "B_correct_donor_reference",
            "requested_experiment": "A_fixed_js_0.01",
            "first_deviation_commit": FIRST_DEVIATION_COMMIT,
            "official_cost_verdict_preserved": "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
            "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
            "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
            "used_confirmation": 0,
            "model_executed": 0,
            "n_audit_rows": len(audit_rows),
            "n_complete_pairs": len(complete_ids),
            "n_incomplete_pairs": len(incomplete_ids),
            "n_direction_failures": len(failed_match),
            "n_in_fixed_0.01_window": n_fixed_window,
            "coverage": subset["coverage"],
            "coverage_gate_90_passed": 0,
            "js_recompute": js_summary,
            "fail_by_type": dict(fail_by_type),
            "fail_by_cell": dict(fail_by_cell),
            "fail_by_reason": dict(fail_by_reason),
            "fail_multiplicity": {str(k): v for k, v in fail_mult.items()},
            "subset_conditional": subset,
            "wrote_into_original_fiber_cost": 0,
        },
    )
    print("08R2-FIBER-COST FORENSIC AUDIT")
    print(f"  verdict={VERDICT_DEVIATION}")
    print(f"  executed=Experiment B (correct-donor-reference)")
    print(f"  requested=Experiment A (fixed JS=0.01)")
    print(f"  first_deviation_commit={FIRST_DEVIATION_COMMIT}")
    print(f"  audit_rows={len(audit_rows)} complete_pairs={len(complete_ids)} direction_fail={len(failed_match)}")
    print(f"  n_in_fixed_0.01_window={n_fixed_window}")
    print(f"  js_recompute={js_summary}")
    print(f"  fail_by_type={dict(fail_by_type)} fail_by_cell={dict(fail_by_cell)}")
    print(f"  coverage={subset['coverage']:.4f} coverage_90=0")
    print(f"  out_dir={out_dir}")
    print("  TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE preserved")
    print("  TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED preserved")
    print("  official letter unchanged: C / PARTIAL")


if __name__ == "__main__":
    main()
