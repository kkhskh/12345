"""08R2-FIBER: probe-visible vs probe-invisible causal decomposition.

Read-only with respect to 08R2 validation labels. No new editor. No confirmation.
Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.othello_r1 import cluster_bootstrap, intervention_probability_metrics, kl_row, legality_sets, total_legal_mass
from src.othello_r2 import (
    OFFICIAL_0A_LETTER,
    PROTOCOL_ABD,
    SITE,
    LinearBoardDecoder,
    pair_manifest_hash,
    require_no_confirmation,
    sklearn_proba_pack,
)
from src.othello_runutil import write_json


R2_DIRNAME = "08r2"
FIBER_DIRNAME = "fiber"
PRIMARY = {"A_legal_diff", "B_similar_board_legal_change"}
STATUS_ENGINEERING = "ENGINEERING_ONLY"
STATUS_REAL_REQUIRED = "REAL_MODEL_REQUIRED"
STATUS_VALIDATION_FAILED = "FIBER_VALIDATION_FAILED"
STATUS_COMPLETE = "FIBER_DIAGNOSTIC_COMPLETE"
VERDICT_SUPPORTED = "PROBE_LOGIT_INSUFFICIENCY_SUPPORTED"
VERDICT_NOT_ESTABLISHED = "PROBE_LOGIT_INSUFFICIENCY_NOT_ESTABLISHED"
VERDICT_INCONCLUSIVE = "FIBER_DIAGNOSTIC_INCONCLUSIVE"
VERDICT_ENGINEERING = "FIBER_ENGINEERING_FAILED"

REQUIRED_FIBER_FILES = (
    "08r2_fiber_dependency_manifest.json",
    "08r2_fiber_probe_contrast_parity.csv",
    "08r2_fiber_svd_report.json",
    "08r2_fiber_matched_pairs.csv",
    "08r2_fiber_factorial_interventions.csv",
    "08r2_fiber_full_logits.npz",
    "08r2_fiber_probe_probabilities.npz",
    "08r2_fiber_natural_anchor.csv",
    "08r2_fiber_controls.csv",
    "08r2_fiber_bootstrap.csv",
    "08r2_fiber_completion_status.json",
    "08r2_fiber_scientific_report.md",
)

DEFAULT_TOLS = {
    "rank_tolerances": (1e-6, 1e-8, 1e-10),
    "primary_rank_tol": 1e-8,
    "probe_proba_atol_f64": 1e-8,
    "probe_proba_atol_f32": 1e-5,
    "parity_proba_atol": 1e-6,
    "behavior_floor": 0.05,
    "js_same_atol": 1e-8,
    "js_meaningful": 0.001,
    "repro_dp_atol": 0.01,
    "repro_l2_atol": 0.25,
    "null_component_min_norm": 1e-8,
}


def isolate_fiber_results_dir(requested: str | Path) -> dict:
    requested = Path(requested)
    if requested.name == FIBER_DIRNAME and requested.parent.name == R2_DIRNAME:
        out = requested
    elif requested.name == R2_DIRNAME:
        out = requested / FIBER_DIRNAME
    else:
        out = requested / R2_DIRNAME / FIBER_DIRNAME
    return {
        "requested": str(requested),
        "results_dir": str(out),
        "nested": int(out != requested),
        "fiber": 1,
        "reason": "08R2-FIBER writes only under 08r2/fiber/",
    }


def reject_unexpected_preprocessing(pack: dict) -> dict:
    banned = [k for k in pack if k in {"scaler", "pca", "preprocessor", "pipeline", "transform"}]
    return {
        "unexpected_preprocessing": banned,
        "ok": len(banned) == 0,
        "note": "08c pack is hook/target/method/models; no fitted scaler.",
    }


def independent_contrasts(decoder: LinearBoardDecoder) -> dict:
    """Independent class-logit contrasts. Wδ=0 preserves every probe probability."""
    weights = []
    biases = []
    rows = []
    square_kind = []
    for j, classes in enumerate(decoder.class_sets):
        present = [int(c) for c in classes]
        if len(present) <= 1:
            square_kind.append({"square": j, "classes": present, "n_contrasts": 0, "kind": "constant"})
            continue
        kind = "binary" if len(present) == 2 else "multiclass"
        square_kind.append({"square": j, "classes": present, "n_contrasts": len(present) - 1, "kind": kind})
        ref = present[0]
        for cls in present[1:]:
            w = np.asarray(decoder.coef[j, cls] - decoder.coef[j, ref], dtype=np.float64)
            b = float(decoder.intercept[j, cls] - decoder.intercept[j, ref])
            weights.append(w)
            biases.append(b)
            rows.append({"square": j, "ref": ref, "cls": int(cls), "kind": kind, "bias": b})
    if not weights:
        W = np.zeros((0, decoder.d_model), dtype=np.float64)
        b = np.zeros((0,), dtype=np.float64)
    else:
        W = np.stack(weights, axis=0).astype(np.float64)
        b = np.asarray(biases, dtype=np.float64)
    return {
        "W": W,
        "biases": b,
        "rows": rows,
        "squares": square_kind,
        "n_contrasts": int(W.shape[0]),
        "d_model": int(decoder.d_model),
        "n_binary": int(sum(s["kind"] == "binary" for s in square_kind)),
        "n_multiclass": int(sum(s["kind"] == "multiclass" for s in square_kind)),
        "n_constant": int(sum(s["kind"] == "constant" for s in square_kind)),
    }


def contrast_proba(x: np.ndarray, decoder: LinearBoardDecoder, contrasts: dict) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    p = np.zeros((x.shape[0], 64, 3), dtype=np.float64)
    by_sq: dict[int, list[int]] = {}
    for i, row in enumerate(contrasts["rows"]):
        by_sq.setdefault(int(row["square"]), []).append(i)
    W = contrasts["W"]
    b = contrasts["biases"]
    for j, classes in enumerate(decoder.class_sets):
        present = [int(c) for c in classes]
        if len(present) <= 1:
            p[:, j, present[0] if present else 0] = 1.0
            continue
        z = np.zeros((x.shape[0], 3), dtype=np.float64)
        for i in by_sq.get(j, []):
            z[:, int(contrasts["rows"][i]["cls"])] = x @ W[i] + b[i]
        live = z[:, present]
        live = live - live.max(axis=1, keepdims=True)
        e = np.exp(live)
        sm = e / e.sum(axis=1, keepdims=True)
        for k, cls in enumerate(present):
            p[:, j, cls] = sm[:, k]
    return p


def max_abs_proba(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def contrast_parity_report(decoder: LinearBoardDecoder, models, x: np.ndarray, contrasts: dict, atol: float) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    sk = sklearn_proba_pack(models, x)
    dec = decoder.proba_np(x)
    con = contrast_proba(x, decoder, contrasts)
    pred_sk = np.stack([m.predict(x) for m in models], axis=1)
    pred_dec = decoder.predict_np(x)
    pred_con = np.argmax(con, axis=-1)
    max_sk_dec = max_abs_proba(sk, dec)
    max_sk_con = max_abs_proba(sk, con)
    max_dec_con = max_abs_proba(dec, con)
    ok = (
        max_sk_dec <= atol
        and max_sk_con <= atol
        and max_dec_con <= atol
        and float((pred_sk == pred_dec).mean()) == 1.0
        and float((pred_sk == pred_con).mean()) == 1.0
        and float((pred_sk == pred_dec).all(axis=1).mean()) == 1.0
    )
    return {
        "n": int(x.shape[0]),
        "max_abs_sklearn_vs_decoder": max_sk_dec,
        "max_abs_sklearn_vs_contrast": max_sk_con,
        "max_abs_decoder_vs_contrast": max_dec_con,
        "square_agree_sklearn_decoder": float((pred_sk == pred_dec).mean()),
        "board_agree_sklearn_decoder": float((pred_sk == pred_dec).all(axis=1).mean()),
        "square_agree_sklearn_contrast": float((pred_sk == pred_con).mean()),
        "board_agree_sklearn_contrast": float((pred_sk == pred_con).all(axis=1).mean()),
        "atol": float(atol),
        "ok": int(ok),
    }


def svd_probe_geometry(W: np.ndarray, primary_tol: float, tols: list[float] | tuple[float, ...]) -> dict:
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] == 0:
        return {
            "shape": list(W.shape),
            "numerical_rank": 0,
            "nullspace_dim": int(W.shape[1]) if W.ndim == 2 else 0,
            "ok": 0,
            "reason": "empty_contrast_matrix",
        }
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    smax = float(s[0]) if s.size else 0.0
    sensitivity = []
    for tol in tols:
        r = int(np.sum(s > float(tol) * smax)) if smax > 0 else 0
        sensitivity.append({"tol": float(tol), "rank": r, "nullspace_dim": int(W.shape[1] - r)})
    rank = int(np.sum(s > float(primary_tol) * smax)) if smax > 0 else 0
    rank = max(0, min(rank, vt.shape[0], W.shape[1]))
    vr = vt[:rank].T if rank else np.zeros((W.shape[1], 0), dtype=np.float64)
    p_r = vr @ vr.T if rank else np.zeros((W.shape[1], W.shape[1]), dtype=np.float64)
    p_n = np.eye(W.shape[1], dtype=np.float64) - p_r
    recon = (u[:, :rank] * s[:rank]) @ vt[:rank] if rank else np.zeros_like(W)
    gram = vr.T @ vr if rank else np.zeros((0, 0))
    eye_r = np.eye(rank, dtype=np.float64) if rank else gram
    report = {
        "shape": [int(W.shape[0]), int(W.shape[1])],
        "singular_values": [float(v) for v in s],
        "rank_tolerance": float(primary_tol),
        "rank_tolerance_kind": "relative_to_largest_singular_value",
        "numerical_rank": rank,
        "rowspace_dim": rank,
        "nullspace_dim": int(W.shape[1] - rank),
        "sensitivity": sensitivity,
        "orthogonality_error": float(np.linalg.norm(gram - eye_r)) if rank else 0.0,
        "reconstruction_error": float(np.linalg.norm(W - recon)),
        "projection_idempotence_error_R": float(np.linalg.norm(p_r @ p_r - p_r)),
        "projection_idempotence_error_N": float(np.linalg.norm(p_n @ p_n - p_n)),
        "complement_error": float(np.linalg.norm(p_r + p_n - np.eye(W.shape[1]))),
        "nullspace_image_error": float(np.linalg.norm(W @ p_n)),
        "has_meaningful_nullspace": int(W.shape[1] - rank > 0),
        "V_r": vr,
        "P_R": p_r,
        "P_N": p_n,
        "ok": 1,
    }
    return report


def project_delta(delta: np.ndarray, p_r: np.ndarray, p_n: np.ndarray) -> dict:
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    d_r = p_r @ d
    d_n = p_n @ d
    return {
        "delta": d,
        "delta_R": d_r,
        "delta_N": d_n,
        "l2": float(np.linalg.norm(d)),
        "l2_R": float(np.linalg.norm(d_r)),
        "l2_N": float(np.linalg.norm(d_n)),
        "split_error": float(np.linalg.norm(d - d_r - d_n)),
    }


def factorial_activations(h_e: np.ndarray, h_d: np.ndarray, p_r: np.ndarray, p_n: np.ndarray) -> dict:
    h_e = np.asarray(h_e, dtype=np.float64).reshape(-1)
    h_d = np.asarray(h_d, dtype=np.float64).reshape(-1)
    parts = project_delta(h_d - h_e, p_r, p_n)
    cells = {
        "H00": h_e,
        "H10": h_e + parts["delta_R"],
        "H01": h_e + parts["delta_N"],
        "H11": h_e + parts["delta_R"] + parts["delta_N"],
    }
    return {**parts, "cells": cells, "h11_minus_donor": float(np.linalg.norm(cells["H11"] - h_d))}


def probe_invariance(decoder: LinearBoardDecoder, a: np.ndarray, b: np.ndarray, atol: float) -> dict:
    pa = decoder.proba_np(a)
    pb = decoder.proba_np(b)
    err = max_abs_proba(pa, pb)
    return {
        "max_abs_proba": err,
        "ok": int(err <= atol),
        "atol": float(atol),
        "same_labels": int(np.array_equal(decoder.predict_np(a).reshape(-1), decoder.predict_np(b).reshape(-1))),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_row(p, m) + 0.5 * kl_row(q, m)


def y_functional(probs: np.ndarray, rec_legal, tgt_legal) -> dict:
    parts = legality_sets(rec_legal, tgt_legal)
    gain = total_legal_mass(probs, parts["became_legal"])
    loss = total_legal_mass(probs, parts["became_illegal"])
    stay_l = total_legal_mass(probs, parts["stayed_legal"])
    stay_i = total_legal_mass(probs, parts["stayed_illegal"])
    y = float(gain - loss) if gain == gain and loss == loss else float("nan")
    return {
        "Y": y,
        "became_legal_mass": float(gain) if gain == gain else float("nan"),
        "became_illegal_mass": float(loss) if loss == loss else float("nan"),
        "stayed_legal_mass": float(stay_l) if stay_l == stay_l else float("nan"),
        "stayed_illegal_mass": float(stay_i) if stay_i == stay_i else float("nan"),
    }


def causal_from_base(base_probs: np.ndarray, edited_probs: np.ndarray, rec_legal, tgt_legal) -> dict:
    mets = intervention_probability_metrics(base_probs, edited_probs, rec_legal, tgt_legal)
    y = y_functional(edited_probs, rec_legal, tgt_legal)
    return {**mets, **y, "kl_to_base": kl_row(base_probs, edited_probs)}


def factorial_effects(y00: float, y10: float, y01: float, y11: float) -> dict:
    return {
        "E_R": float(y10 - y00),
        "E_N": float(y01 - y00),
        "I": float(y11 - y10 - y01 + y00),
        "E_R_given_N": float(y11 - y01),
        "E_N_given_R": float(y11 - y10),
        "Y00": float(y00),
        "Y10": float(y10),
        "Y01": float(y01),
        "Y11": float(y11),
    }


def random_nullspace_direction(p_n: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    raw = p_n @ rng.normal(size=p_n.shape[0])
    nrm = float(np.linalg.norm(raw))
    if nrm <= 1e-15:
        raise RuntimeError("random_nullspace_direction_vanished")
    return raw / nrm


def unit_null(delta_n: np.ndarray, min_norm: float) -> np.ndarray | None:
    v = np.asarray(delta_n, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if nrm < float(min_norm):
        return None
    return v / nrm


def training_distance(h: np.ndarray, mean: np.ndarray, pca_basis: np.ndarray) -> dict:
    z = np.asarray(h, dtype=np.float64).reshape(-1) - np.asarray(mean, dtype=np.float64).reshape(-1)
    basis = np.asarray(pca_basis, dtype=np.float64)
    proj = (z @ basis) @ basis.T
    resid = z - proj
    return {
        "l2_to_train_mean": float(np.linalg.norm(z)),
        "pca_residual_l2": float(np.linalg.norm(resid)),
        "note": "Diagnostic distance from training activation stats, not manifold membership.",
    }


def pair_demonstrates_insufficiency(js_00_01: float, js_10_11: float, e_n: float, floor: float, js_floor: float) -> bool:
    return bool(
        (js_00_01 == js_00_01 and js_00_01 >= js_floor)
        or (js_10_11 == js_10_11 and js_10_11 >= js_floor)
        or (e_n == e_n and abs(e_n) >= floor)
    )


def decide_verdict(n_valid: int, n_demo: int, n_matched: int, nullspace_dim: int, parity_ok: bool, hooks_ok: bool) -> str:
    if not parity_ok or not hooks_ok:
        return VERDICT_ENGINEERING
    if nullspace_dim <= 0:
        return VERDICT_INCONCLUSIVE
    if n_valid == 0:
        return VERDICT_INCONCLUSIVE
    if n_demo > 0:
        return VERDICT_SUPPORTED
    return VERDICT_NOT_ESTABLISHED


def load_validation_primary_pairs(path: Path) -> tuple[list[dict], dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = [p for p in doc.get("pairs", []) if p.get("r2_type") in PRIMARY]
    flags = {
        "n_primary": int(len(pairs)),
        "n_a": int(sum(p["r2_type"] == "A_legal_diff" for p in pairs)),
        "n_b": int(sum(p["r2_type"] == "B_similar_board_legal_change" for p in pairs)),
        "n_d_in_primary_filter": 0,
        "pair_manifest_sha256_saved": doc.get("pair_manifest_sha256"),
        "used_confirmation": int(doc.get("used_confirmation") or 0),
    }
    return pairs, flags


def verify_pairs_against_sampler(saved: list[dict], recomputed: list[dict]) -> dict:
    saved_p = [p for p in saved if p.get("r2_type") in PRIMARY]
    rec_p = [p for p in recomputed if p.get("r2_type") in PRIMARY]
    h_saved = pair_manifest_hash(saved_p)
    h_rec = pair_manifest_hash(rec_p)
    return {
        "saved_hash": h_saved,
        "recomputed_hash": h_rec,
        "ok": int(h_saved == h_rec),
        "n_saved": int(len(saved_p)),
        "n_recomputed": int(len(rec_p)),
    }


def historical_editor_a_row(causal_rows: list[dict], pair: dict, budget: float = 16.0) -> dict | None:
    hits = []
    for r in causal_rows:
        if r.get("method") != "editor_A_fullspace":
            continue
        if str(r.get("recip_game")) != str(pair.get("recip_game")):
            continue
        if str(r.get("target_game")) != str(pair.get("target_game")):
            continue
        if str(r.get("r2_type")) != str(pair.get("r2_type")):
            continue
        try:
            if abs(float(r.get("budget", 0)) - float(budget)) > 1e-9:
                continue
        except (TypeError, ValueError):
            continue
        hits.append(r)
    return hits[0] if hits else None


def reproduction_ok(repro: dict, hist: dict | None, tols: dict) -> dict:
    if hist is None:
        return {"ok": 0, "reason": "historical_editor_A_row_missing"}
    checks = {
        "board_exact_repro": float(repro.get("board_exact", 0)),
        "board_exact_hist": float(hist.get("board_exact", 0)),
        "dP_gain_abs_diff": abs(float(repro.get("dP_gain", np.nan)) - float(hist.get("dP_gain", np.nan))),
        "dP_loss_abs_diff": abs(float(repro.get("dP_loss", np.nan)) - float(hist.get("dP_loss", np.nan))),
        "l2_abs_diff": abs(float(repro.get("delta_norm", np.nan)) - float(hist.get("delta_norm", np.nan))),
    }
    ok = (
        checks["board_exact_repro"] == 1.0
        and checks["board_exact_hist"] == 1.0
        and checks["dP_gain_abs_diff"] <= tols["repro_dp_atol"]
        and checks["dP_loss_abs_diff"] <= tols["repro_dp_atol"]
        and checks["l2_abs_diff"] <= tols["repro_l2_atol"]
    )
    return {"ok": int(ok), "reason": "ok" if ok else "reproduction_outside_tolerance", **checks}


def slice_boot(values, clusters, n_boot: int, seed: int) -> dict:
    out = cluster_bootstrap(np.asarray(values, dtype=np.float64), list(clusters), n_boot, seed)
    return {k: out[k] for k in ("n", "n_clusters", "point", "boot_mean", "ci95_low", "ci95_high")}


def evaluate_fiber_completeness(out_dir: str | Path, *, real_model: bool = True) -> dict:
    out_dir = Path(out_dir)
    failures = []
    files = {name: (out_dir / name).is_file() for name in REQUIRED_FIBER_FILES}
    for name, exists in files.items():
        if not exists:
            failures.append(f"missing_file:{name}")
        elif (out_dir / name).stat().st_size == 0:
            failures.append(f"empty_file:{name}")
    status = {}
    if files.get("08r2_fiber_completion_status.json"):
        status = json.loads((out_dir / "08r2_fiber_completion_status.json").read_text(encoding="utf-8"))
    if status.get("used_confirmation"):
        failures.append("confirmation_accessed")
    if real_model:
        if status.get("checkpoint_loaded") != 1:
            failures.append("checkpoint_not_loaded")
        if status.get("probe_loaded") != 1:
            failures.append("probe_not_loaded")
        if status.get("parity_ok") != 1:
            failures.append("probe_parity_failed")
        if status.get("nullspace_ok") != 1:
            failures.append("nullspace_missing_or_failed")
        if status.get("hooks_ok") != 1:
            failures.append("hooks_failed")
        if status.get("full_outputs_saved") != 1:
            failures.append("full_outputs_not_saved")
        n_valid = int(status.get("n_valid") or 0)
        n_cells = int(status.get("n_factorial_cells") or 0)
        if n_valid > 0 and n_cells < 4 * n_valid:
            failures.append("missing_factorial_cells")
        if n_valid == 0 and status.get("status") == STATUS_COMPLETE:
            failures.append("complete_without_valid_pairs")
        if int(status.get("n_control_rows") or 0) < 1 and status.get("status") == STATUS_COMPLETE:
            failures.append("controls_omitted")
        if status.get("synthetic_scientific_rows"):
            failures.append("synthetic_scientific_rows")
    if "fiber" not in Path(out_dir).parts:
        failures.append("fiber_outputs_not_isolated")
    return {
        "ok": int(not failures),
        "failures": failures,
        "files": files,
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "section_5_3_status": "PARTIAL",
        "obstruction_ready": "NO",
    }


def write_scientific_report(path: Path, payload: dict) -> None:
    svd = payload.get("svd", {})
    slices = payload.get("slices", {})
    a = slices.get("A", {})
    b = slices.get("B", {})
    pooled = slices.get("pooled", {})
    ident = payload.get("identifiability", {})
    lines = [
        "# 08R2-FIBER Scientific Report",
        "",
        "Probe-visible versus probe-invisible decomposition of the Editor A vs full-donor gap.",
        "This is an exploratory mechanistic follow-up on the already-examined 08R2 validation pairs.",
        "It is not sealed confirmation and not a new editor competition.",
        "",
        "Official labels unchanged:",
        "",
        "- `08R2_VALIDATION_COMPLETE — NEGATIVE`",
        "- Original 0A: **C / PARTIAL**",
        "- `SECTION_5_3 = PARTIAL`",
        "- `OBSTRUCTION_READY = NO`",
        "",
        f"Status: `{payload.get('status')}`",
        f"Verdict: `{payload.get('verdict')}`",
        "",
        "## Predefined tolerances",
        "",
        f"{payload.get('tolerances')}",
        "",
        "## A. Probe contrast rank",
        "",
        f"- Contrast matrix shape: {svd.get('shape')}",
        f"- Numerical rank / rowspace dim: {svd.get('numerical_rank')}",
        f"- Nullspace dim: {svd.get('nullspace_dim')}",
        f"- Primary relative tolerance: {svd.get('rank_tolerance')}",
        f"- Singular-value sensitivity: {svd.get('sensitivity')}",
        f"- Orthogonality error: {svd.get('orthogonality_error')}",
        f"- Reconstruction error: {svd.get('reconstruction_error')}",
        f"- Idempotence errors: R={svd.get('projection_idempotence_error_R')} N={svd.get('projection_idempotence_error_N')}",
        f"- W P_N residual: {svd.get('nullspace_image_error')}",
        "",
        "## B. Probe-equivalent activations with different downstream decisions",
        "",
        f"- Matched primary pairs: {payload.get('n_matched')}",
        f"- Valid (invariance held): {payload.get('n_valid')}",
        f"- Invalid invariance: {payload.get('n_invalid')}",
        f"- Pairs demonstrating insufficiency: {payload.get('n_demo')}",
        f"- Category A demo: {a.get('n_demo')} / {a.get('n_valid')} valid",
        f"- Category B demo: {b.get('n_demo')} / {b.get('n_valid')} valid",
        "",
        "A pair demonstrates insufficiency only if the complete probe probability vectors agree within the predefined atol and the downstream distribution or Y-functional still moves by a predefined floor.",
        "",
        "## C–E. Factorial contributions",
        "",
        f"- Mean E_R (probe-visible): {pooled.get('mean_E_R')} median={pooled.get('median_E_R')} CI={pooled.get('E_R_ci')}",
        f"- Mean E_N (probe-invisible): {pooled.get('mean_E_N')} median={pooled.get('median_E_N')} CI={pooled.get('E_N_ci')}",
        f"- Mean interaction I: {pooled.get('mean_I')} median={pooled.get('median_I')} CI={pooled.get('I_ci')}",
        f"- Conditional E_R | N : {pooled.get('mean_E_R_given_N')}",
        f"- Conditional E_N | R : {pooled.get('mean_E_N_given_R')}",
        f"- A: E_R={a.get('mean_E_R')} E_N={a.get('mean_E_N')} I={a.get('mean_I')} targets={a.get('n_target_games')}",
        f"- B: E_R={b.get('mean_E_R')} E_N={b.get('mean_E_N')} I={b.get('mean_I')} targets={b.get('n_target_games')}",
        "",
        "The Transformer is not assumed linear in activation space. Interaction is reported, not absorbed into main effects.",
        "",
        "## F–G. Natural-anchor null perturbations",
        "",
        f"- Natural-anchor rows: {payload.get('n_anchor')}",
        f"- Mean |ΔY| at ε=1 around donor: {payload.get('anchor_eps1_mean_abs_dY')}",
        f"- Same-norm random-null |ΔY| at ε=1: {payload.get('random_eps1_mean_abs_dY')}",
        f"- Probe invariance failures on anchors: {payload.get('n_anchor_invariance_fail')}",
        "",
        "## H. Alignment versus generic disruption",
        "",
        f"- Mean JS(H00, H01): {pooled.get('mean_js_00_01')}",
        f"- Mean JS(H10, H11): {pooled.get('mean_js_10_11')}",
        f"- Mean |E_N|: {pooled.get('mean_abs_E_N')}",
        f"- Wrong-donor null mean |ΔY| on the intended legal set: {payload.get('wrong_donor_mean_abs_dY')}",
        "",
        "Legal-set Y movement is not the full output distribution. JS is the full-distribution comparison. A large JS with small Y is generic disruption.",
        "",
        "## I. A versus B",
        "",
        f"- A pairs use {a.get('n_target_games')} target games. Recipient-game CIs do not establish generalization across arbitrary targets.",
        f"- B pairs use {b.get('n_target_games')} target games.",
        "",
        "## J. History / distribution-shift confound",
        "",
        f"- Prefix-length and occupancy contrasts are in `08r2_fiber_matched_pairs.csv`.",
        f"- Training-stat diagnostic distances are recorded per cell. They are not proof of manifold membership.",
        f"- Off-manifold hybrids (H01, H10) can inflate effects relative to small natural-anchor moves.",
        "",
        "## K. What a stronger germ would need",
        "",
        ident.get(
            "stronger_germ",
            "Tested evidence concerns whether R(h) determines F_x(h) on these interventions. "
            "A stronger local state would need whatever residual information still moves F_x when R is held fixed. "
            "That information is not identified as a unique game-state coordinate by this diagnostic.",
        ),
        "",
        "## Claims this does not support",
        "",
        "- The nullspace is not shown to be a unique mathematical game state.",
        "- The board probe is not shown to be wrong about the board.",
        "- COR / Section 5.3 is not confirmed or refuted. Confirmation remains sealed.",
        "- Obstruction energy is not computed.",
        "- No new state editor is selected.",
        "",
        f"## Verdict",
        "",
        f"`{payload.get('verdict')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_safe(path: Path, obj) -> None:
    write_json(path, obj)


def require_no_confirm_pairs(pairs: list[dict], holdout_ids: set[str] | None = None) -> None:
    require_no_confirmation(pairs, holdout_ids)
    if any(p.get("used_confirmation") for p in pairs):
        raise RuntimeError("confirmation flag set on a fiber pair")
