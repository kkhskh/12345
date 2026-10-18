"""Read-only 08R2-FIBER specificity audit. No model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.othello_r2_fiber import (
    DEFAULT_TOLS,
    PRIMARY,
    factorial_effects,
    js_divergence,
    slice_boot,
    softmax,
)
from src.othello_runutil import write_json


ORIGINAL = (
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
CELLS = ("H00", "H10", "H01", "H11")
EPSILONS = (1.0, 2.0, 4.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only FIBER natural-anchor and specificity audit.")
    p.add_argument(
        "--fiber-dir",
        default="/content/drive/MyDrive/ml_research_artifacts/toy_othello/results/08r2/fiber",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=82)
    return p.parse_args()


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_logit_key(key: str) -> tuple[str, str, str] | None:
    if "__" not in key:
        return None
    recip, rest = key.split("__", 1)
    if "__" not in rest:
        return None
    target, cell = rest.rsplit("__", 1)
    return recip, target, cell


def check_simplex(arr: np.ndarray) -> dict:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "finite": bool(np.isfinite(x).all()),
        "nonnegative": bool((x >= -1e-12).all()),
        "sum": float(x.sum()),
        "normalized": bool(abs(float(x.sum()) - 1.0) <= 1e-6),
        "n": int(x.size),
    }


def load_npz_maps(path: Path) -> dict[tuple[str, str, str], np.ndarray]:
    out = {}
    if not path.is_file():
        return out
    blob = np.load(path, allow_pickle=False)
    for key in blob.files:
        parsed = parse_logit_key(key)
        if parsed:
            out[parsed] = np.asarray(blob[key], dtype=np.float64)
    return out


def verify_logits(logit_map: dict) -> dict:
    if not logit_map:
        return {"present": 0, "n_vectors": 0, "n_ok": 0, "max_abs_sum_error": float("nan")}
    bad = 0
    max_sum_err = 0.0
    for arr in logit_map.values():
        p = softmax(arr)
        chk = check_simplex(p)
        max_sum_err = max(max_sum_err, abs(chk["sum"] - 1.0))
        if not (chk["finite"] and chk["nonnegative"] and chk["normalized"]):
            bad += 1
    return {
        "present": 1,
        "n_vectors": int(len(logit_map)),
        "n_ok": int(len(logit_map) - bad),
        "n_bad": int(bad),
        "max_abs_sum_error": float(max_sum_err),
    }


def verify_probe_arrays(probe_map: dict, atol: float) -> dict:
    if not probe_map:
        return {"present": 0, "invariance_max_err_01": float("nan"), "invariance_max_err_10": float("nan")}
    err01 = []
    err10 = []
    pairs = {(r, t) for r, t, _ in probe_map}
    missing = 0
    for recip, target in pairs:
        got = {c: probe_map.get((recip, target, c)) for c in CELLS}
        if any(v is None for v in got.values()):
            missing += 1
            continue
        for arr in got.values():
            if arr.shape != (64, 3) or not np.isfinite(arr).all():
                missing += 1
        err01.append(float(np.max(np.abs(got["H01"] - got["H00"]))))
        err10.append(float(np.max(np.abs(got["H10"] - got["H11"]))))
    return {
        "present": 1,
        "n_pair_keys": int(len(pairs)),
        "n_incomplete_probe_pairs": int(missing),
        "invariance_max_err_01": float(np.max(err01)) if err01 else float("nan"),
        "invariance_max_err_10": float(np.max(err10)) if err10 else float("nan"),
        "invariance_mean_err_01": float(np.mean(err01)) if err01 else float("nan"),
        "invariance_ok": int(bool(err01) and max(err01) <= atol and max(err10) <= atol),
    }


def recompute_from_logits(logit_map: dict, matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, m in matched.iterrows():
        recip, target = str(m["recip_game"]), str(m["target_game"])
        probs = {}
        missing = []
        for cell in CELLS:
            arr = logit_map.get((recip, target, cell))
            if arr is None:
                missing.append(cell)
                continue
            probs[cell] = softmax(arr)
        if missing:
            rows.append({"recip_game": recip, "target_game": target, "raw_logits_complete": 0, "missing_cells": ",".join(missing)})
            continue
        y_csv = {c: float(m[f"Y{c[1:]}"]) if f"Y{c[1:]}" in m else float("nan") for c in CELLS}
        # Factorial CSV Y is used for identity; JS is recomputed from raw 61-way.
        fx_csv = factorial_effects(float(m["Y00"]), float(m["Y10"]), float(m["Y01"]), float(m["Y11"]))
        identity_err = abs((fx_csv["Y11"] - fx_csv["Y00"]) - (fx_csv["E_R"] + fx_csv["E_N"] + fx_csv["I"]))
        rows.append(
            {
                "recip_game": recip,
                "target_game": target,
                "r2_type": m["r2_type"],
                "raw_logits_complete": 1,
                "js_00_01_recomputed": js_divergence(probs["H00"], probs["H01"]),
                "js_10_11_recomputed": js_divergence(probs["H10"], probs["H11"]),
                "js_00_10_recomputed": js_divergence(probs["H00"], probs["H10"]),
                "js_00_11_recomputed": js_divergence(probs["H00"], probs["H11"]),
                "js_00_01_saved": float(m.get("js_00_01", np.nan)),
                "js_10_11_saved": float(m.get("js_10_11", np.nan)),
                "E_R_recomputed": fx_csv["E_R"],
                "E_N_recomputed": fx_csv["E_N"],
                "I_recomputed": fx_csv["I"],
                "identity_err": identity_err,
                "Y00": fx_csv["Y00"],
                "Y10": fx_csv["Y10"],
                "Y01": fx_csv["Y01"],
                "Y11": fx_csv["Y11"],
            }
        )
    return pd.DataFrame(rows)


def factorial_integrity(fact: pd.DataFrame) -> dict:
    cells = fact[fact["cell"].isin(CELLS)] if "cell" in fact.columns else fact
    pairs = cells.groupby(["recip_game", "target_game"])["cell"].nunique() if len(cells) else pd.Series(dtype=int)
    y_check = []
    if {"became_legal_mass", "became_illegal_mass", "Y"}.issubset(cells.columns):
        y_check = ( _num(cells["Y"]) - (_num(cells["became_legal_mass"]) - _num(cells["became_illegal_mass"])) ).abs()
    return {
        "n_factorial_rows": int(len(cells)),
        "n_pairs_with_four_cells": int((pairs == 4).sum()) if len(pairs) else 0,
        "n_pairs_missing_cells": int((pairs != 4).sum()) if len(pairs) else 0,
        "max_Y_vs_mass_err": float(np.nanmax(y_check)) if len(y_check) else float("nan"),
    }


def _boot(df: pd.DataFrame, col: str, n_boot: int, seed: int) -> dict | None:
    if df.empty or col not in df.columns:
        return None
    vals = _num(df[col]).tolist()
    clusters = df["recip_game"].astype(str).tolist()
    return slice_boot(vals, clusters, n_boot, seed)


def summarize_anchor(anchor: pd.DataFrame, n_boot: int, seed: int) -> dict:
    donor = anchor[anchor.get("family", pd.Series(dtype=str)) == "donor_null_direction"].copy() if "family" in anchor.columns else anchor.iloc[0:0]
    rand = anchor[anchor.get("family", pd.Series(dtype=str)) == "random_nullspace"].copy() if "family" in anchor.columns else anchor.iloc[0:0]
    out = {"by_eps": {}, "n_donor_rows": int(len(donor)), "n_random_rows": int(len(rand))}
    if donor.empty:
        return {**out, "present": 0}
    donor["abs_dY"] = _num(donor.get("dY_vs_donor")).abs()
    donor["js"] = _num(donor.get("js_vs_donor"))
    donor["inv_ok"] = _num(donor.get("invariance_ok")).fillna(0)
    donor["l2"] = _num(donor.get("l2_to_donor"))
    if not rand.empty:
        rand["abs_dY"] = _num(rand.get("dY_vs_donor")).abs()
        rand["js"] = _num(rand.get("js_vs_donor"))
    for eps in EPSILONS:
        d = donor[np.isclose(_num(donor["eps"]), eps)]
        r = rand[np.isclose(_num(rand["eps"]), eps)] if len(rand) else rand
        plus = d[np.isclose(_num(d.get("sign", 0)), 1.0)]
        minus = d[np.isclose(_num(d.get("sign", 0)), -1.0)]
        fd_rows = []
        if len(plus) and len(minus):
            merged = plus.merge(minus, on=["recip_game", "target_game", "r2_type"], suffixes=("_p", "_m"))
            merged["fd"] = (_num(merged["dY_vs_donor_p"]) - _num(merged["dY_vs_donor_m"])) / (2.0 * eps)
            merged["abs_dY_pair"] = 0.5 * (_num(merged["abs_dY_p"]) + _num(merged["abs_dY_m"]))
            merged["js_pair"] = 0.5 * (_num(merged["js_p"]) + _num(merged["js_m"]))
            fd_rows = merged
        pair_cmp = []
        if len(d) and len(r):
            d_pair = d.groupby(["recip_game", "target_game", "r2_type"], as_index=False).agg(
                abs_dY=("abs_dY", "mean"), js=("js", "mean"), n=("abs_dY", "size")
            )
            r_pair = r.groupby(["recip_game", "target_game", "r2_type"], as_index=False).agg(
                abs_dY=("abs_dY", "mean"), js=("js", "mean"), n_draws=("draw", "nunique") if "draw" in r.columns else ("abs_dY", "size")
            )
            cmp = d_pair.merge(r_pair, on=["recip_game", "target_game", "r2_type"], suffixes=("_donor", "_rand"))
            cmp["abs_dY_diff"] = _num(cmp["abs_dY_donor"]) - _num(cmp["abs_dY_rand"])
            cmp["js_diff"] = _num(cmp["js_donor"]) - _num(cmp["js_rand"])
            cmp["donor_outperforms"] = _num(cmp["abs_dY_donor"]) > _num(cmp["abs_dY_rand"])
            pair_cmp = cmp
        def slice_block(df, r2=None):
            g = df if r2 is None else df[df["r2_type"] == r2]
            name = "pooled" if r2 is None else ("A" if r2 == "A_legal_diff" else "B")
            if g.empty:
                return {"slice": name, "n": 0}
            return {
                "slice": name,
                "n_pairs": int(g[["recip_game", "target_game"]].drop_duplicates().shape[0]),
                "n_recipient_games": int(g["recip_game"].nunique()),
                "n_target_games": int(g["target_game"].nunique()),
                "n_rows": int(len(g)),
                "mean_abs_dY": float(np.nanmean(_num(g["abs_dY"]))) if "abs_dY" in g else float("nan"),
                "median_abs_dY": float(np.nanmedian(_num(g["abs_dY"]))) if "abs_dY" in g else float("nan"),
                "mean_js": float(np.nanmean(_num(g["js"]))) if "js" in g else float("nan"),
                "median_js": float(np.nanmedian(_num(g["js"]))) if "js" in g else float("nan"),
                "mean_l2": float(np.nanmean(_num(g["l2"]))) if "l2" in g and np.isfinite(_num(g["l2"])).any() else float("nan"),
                "invariance_fail": int((g["inv_ok"] == 0).sum()) if "inv_ok" in g else 0,
                "max_invariance_err": float(np.nanmax(_num(g.get("invariance_err")))) if "invariance_err" in g else float("nan"),
                "abs_dY_ci": _boot(g, "abs_dY", n_boot, seed) if "abs_dY" in g else None,
                "js_ci": _boot(g, "js", n_boot, seed + 1) if "js" in g else None,
            }
        cmp_block = None
        if isinstance(pair_cmp, pd.DataFrame) and len(pair_cmp):
            cmp_block = {
                "n_pairs": int(len(pair_cmp)),
                "n_random_draws_mean": float(np.nanmean(_num(pair_cmp["n_draws"]))) if "n_draws" in pair_cmp else float("nan"),
                "mean_abs_dY_donor": float(np.nanmean(_num(pair_cmp["abs_dY_donor"]))),
                "mean_abs_dY_rand": float(np.nanmean(_num(pair_cmp["abs_dY_rand"]))),
                "mean_abs_dY_diff": float(np.nanmean(_num(pair_cmp["abs_dY_diff"]))),
                "median_abs_dY_diff": float(np.nanmedian(_num(pair_cmp["abs_dY_diff"]))),
                "mean_js_diff": float(np.nanmean(_num(pair_cmp["js_diff"]))),
                "frac_donor_outperforms": float(pair_cmp["donor_outperforms"].mean()),
                "abs_dY_diff_ci": _boot(pair_cmp, "abs_dY_diff", n_boot, seed + 2),
                "A": {
                    "frac_outperforms": float(pair_cmp[pair_cmp["r2_type"] == "A_legal_diff"]["donor_outperforms"].mean()) if (pair_cmp["r2_type"] == "A_legal_diff").any() else float("nan"),
                    "n": int((pair_cmp["r2_type"] == "A_legal_diff").sum()),
                    "n_targets": int(pair_cmp.loc[pair_cmp["r2_type"] == "A_legal_diff", "target_game"].nunique()),
                },
                "B": {
                    "frac_outperforms": float(pair_cmp[pair_cmp["r2_type"] == "B_similar_board_legal_change"]["donor_outperforms"].mean()) if (pair_cmp["r2_type"] == "B_similar_board_legal_change").any() else float("nan"),
                    "n": int((pair_cmp["r2_type"] == "B_similar_board_legal_change").sum()),
                    "n_targets": int(pair_cmp.loc[pair_cmp["r2_type"] == "B_similar_board_legal_change", "target_game"].nunique()),
                },
            }
        fd_block = None
        if isinstance(fd_rows, pd.DataFrame) and len(fd_rows):
            fd_block = {
                "mean_fd": float(np.nanmean(_num(fd_rows["fd"]))),
                "median_fd": float(np.nanmedian(_num(fd_rows["fd"]))),
                "fd_ci": _boot(fd_rows, "fd", n_boot, seed + 3),
            }
        out["by_eps"][str(eps)] = {
            "donor": slice_block(d),
            "donor_A": slice_block(d, "A_legal_diff"),
            "donor_B": slice_block(d, "B_similar_board_legal_change"),
            "random": slice_block(r.assign(inv_ok=1, l2=np.nan) if len(r) and "inv_ok" not in r else r) if len(r) else {"n": 0},
            "paired_vs_random": cmp_block,
            "finite_difference": fd_block,
        }
    return {**out, "present": 1}


def summarize_wrong_donor(controls: pd.DataFrame, matched: pd.DataFrame, n_boot: int, seed: int) -> dict:
    w = controls[controls.get("cell", pd.Series(dtype=str)) == "wrong_donor_nullspace"].copy() if "cell" in controls.columns else controls.iloc[0:0]
    rnull = controls[controls.get("cell", pd.Series(dtype=str)) == "random_nullspace"].copy() if "cell" in controls.columns else controls.iloc[0:0]
    if w.empty:
        return {"present": 0, "wrong_donor_js": "NOT_IDENTIFIABLE_FROM_SAVED_DATA"}
    w["dY"] = _num(w.get("dY_vs_H00"))
    w["Y"] = _num(w.get("Y"))
    w["dP_gain"] = _num(w.get("dP_gain"))
    w["dP_loss"] = _num(w.get("dP_loss"))
    w["kl"] = _num(w.get("kl_to_base"))
    w["l2"] = _num(w.get("l2_applied", w.get("l2_to_editor_A")))
    m = matched.copy()
    cmp = w.merge(m[["recip_game", "target_game", "r2_type", "E_N", "l2_N", "Y01", "Y00"]], on=["recip_game", "target_game", "r2_type"], how="inner")
    cmp["correct_minus_wrong_dY"] = _num(cmp["E_N"]) - _num(cmp["dY"])
    cmp["l2_ratio"] = _num(cmp["l2"]) / _num(cmp["l2_N"])
    l2_matched = bool(len(cmp) and (cmp["l2_ratio"] - 1.0).abs().max() <= 0.05)
    rn = None
    if not rnull.empty:
        rnull["dY"] = _num(rnull.get("dY_vs_H00"))
        rn = {
            "n": int(len(rnull)),
            "n_pairs": int(rnull[["recip_game", "target_game"]].drop_duplicates().shape[0]) if {"recip_game", "target_game"}.issubset(rnull.columns) else int(len(rnull)),
            "mean_abs_dY": float(np.nanmean(_num(rnull["dY"]).abs())),
            "mean_l2": float(np.nanmean(_num(rnull["matched_l2"]))) if "matched_l2" in rnull.columns else float(np.nanmean(_num(rnull.get("l2_to_editor_A")))),
        }
    return {
        "present": 1,
        "n": int(len(cmp)),
        "n_recipient_games": int(cmp["recip_game"].nunique()) if len(cmp) else 0,
        "n_target_games": int(cmp["target_game"].nunique()) if len(cmp) else 0,
        "mean_wrong_dY": float(np.nanmean(cmp["dY"])) if len(cmp) else float("nan"),
        "mean_correct_E_N": float(np.nanmean(cmp["E_N"])) if len(cmp) else float("nan"),
        "mean_correct_minus_wrong": float(np.nanmean(cmp["correct_minus_wrong_dY"])) if len(cmp) else float("nan"),
        "median_correct_minus_wrong": float(np.nanmedian(cmp["correct_minus_wrong_dY"])) if len(cmp) else float("nan"),
        "mean_wrong_dP_gain": float(np.nanmean(cmp["dP_gain"])) if len(cmp) else float("nan"),
        "mean_wrong_dP_loss": float(np.nanmean(cmp["dP_loss"])) if len(cmp) else float("nan"),
        "mean_wrong_kl": float(np.nanmean(cmp["kl"])) if len(cmp) else float("nan"),
        "mean_wrong_l2": float(np.nanmean(cmp["l2"])) if len(cmp) else float("nan"),
        "mean_correct_l2_N": float(np.nanmean(cmp["l2_N"])) if len(cmp) else float("nan"),
        "cost_matched": int(l2_matched),
        "wrong_donor_js": "NOT_IDENTIFIABLE_FROM_SAVED_DATA",
        "diff_ci": _boot(cmp, "correct_minus_wrong_dY", n_boot, seed) if len(cmp) else None,
        "A_n_targets": int(cmp.loc[cmp["r2_type"] == "A_legal_diff", "target_game"].nunique()) if len(cmp) else 0,
        "B_n_targets": int(cmp.loc[cmp["r2_type"] == "B_similar_board_legal_change", "target_game"].nunique()) if len(cmp) else 0,
        "hybrid_random": rn,
        "note": "Wrong-donor rows were scored on the intended target legal set. Full-distribution JS for wrong-donor was not saved.",
    }


def ci_excludes_zero(ci: dict | None) -> bool | None:
    if not ci:
        return None
    lo, hi = ci.get("ci95_low"), ci.get("ci95_high")
    if lo != lo or hi != hi:
        return None
    return bool(lo > 0 or hi < 0)


def decide_claims(verif: dict, anchor: dict, wrong: dict, tols: dict) -> dict:
    claim_a = "SUPPORTED" if verif.get("n_demo_saved", 0) > 0 and verif.get("identity_max_err", 1) <= 1e-9 else (
        "INCONCLUSIVE" if verif.get("n_pairs_four_cells", 0) < 32 else "SUPPORTED"
    )
    if verif.get("n_pairs_four_cells") == 32 and verif.get("probe_invariance_ok"):
        claim_a = "SUPPORTED"
    eps1 = (anchor.get("by_eps") or {}).get("1.0") or {}
    d1 = eps1.get("donor") or {}
    claim_b = "INCONCLUSIVE"
    if anchor.get("present") and d1.get("n_pairs"):
        mean_dy = d1.get("mean_abs_dY", 0) or 0
        mean_js = d1.get("mean_js", 0) or 0
        dy_ci = ci_excludes_zero(d1.get("abs_dY_ci"))
        js_ci = ci_excludes_zero(d1.get("js_ci"))
        inv_fail = int(d1.get("invariance_fail") or 0)
        if inv_fail:
            claim_b = "INCONCLUSIVE"
        elif (mean_dy >= tols["behavior_floor"] and dy_ci) or (mean_js >= tols["js_meaningful"] and js_ci):
            claim_b = "SUPPORTED"
        elif mean_dy < tols["behavior_floor"] and mean_js < tols["js_meaningful"]:
            claim_b = "NOT_SUPPORTED"
        else:
            claim_b = "INCONCLUSIVE"
    cmp1 = eps1.get("paired_vs_random") or {}
    claim_c = "INCONCLUSIVE"
    random_better = None
    if cmp1.get("n_pairs"):
        random_better = bool(cmp1.get("mean_abs_dY_diff", 0) > 0 and ci_excludes_zero(cmp1.get("abs_dY_diff_ci")))
    wrong_better = None
    if wrong.get("present"):
        wrong_better = bool(wrong.get("mean_correct_minus_wrong", 0) > 0 and ci_excludes_zero(wrong.get("diff_ci")))
        if not wrong.get("cost_matched"):
            wrong_better = None if wrong_better is False else wrong_better
    if random_better is False:
        claim_c = "NOT_SUPPORTED"
    elif random_better is True and wrong_better is True and wrong.get("cost_matched"):
        claim_c = "SUPPORTED"
    elif random_better is True and wrong_better is False:
        claim_c = "NOT_SUPPORTED"
    elif random_better is True and wrong.get("present") and not wrong.get("cost_matched"):
        claim_c = "INCONCLUSIVE"
    elif random_better is True and wrong_better is None:
        claim_c = "INCONCLUSIVE"
    return {
        "A_continuous_probe_insufficiency": claim_a,
        "B_natural_anchor_sensitivity": claim_b,
        "C_target_specific_missing_state": claim_c,
        "C_not_inferred_from_A_or_B": 1,
    }


def _fmt(ci) -> str:
    if not ci:
        return "n/a"
    return f"n={ci['n']} clusters={ci['n_clusters']} point={ci['point']:.4f} CI95=[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]"


def write_report(path: Path, payload: dict) -> None:
    claims = payload["claims"]
    v = payload["verification"]
    a = payload["anchor"]
    w = payload["wrong_donor"]
    svd = payload.get("svd", {})
    lines = [
        "# 08R2-FIBER Specificity and Natural-Anchor Audit",
        "",
        "Read-only analysis of saved FIBER artifacts. No model rerun. Original files were not overwritten.",
        "",
        "Official labels unchanged:",
        "",
        "- `08R2_VALIDATION_COMPLETE — NEGATIVE`",
        "- Original 0A: **C / PARTIAL**",
        "- `SECTION_5_3 = PARTIAL`",
        "- `OBSTRUCTION_READY = NO`",
        "",
        "## 1. Raw integrity",
        "",
        f"- Files present: {payload['file_counts']}",
        f"- Factorial pairs with H00/H10/H01/H11: {v.get('n_pairs_four_cells')}",
        f"- Missing factorial cells: {v.get('n_pairs_missing_cells')}",
        f"- Logit vectors: {v.get('logits')}",
        f"- Recomputed JS max |saved-raw| 00-01: {v.get('js_00_01_max_abs_err')}",
        f"- Recomputed JS max |saved-raw| 10-11: {v.get('js_10_11_max_abs_err')}",
        f"- Y vs became_legal-became_illegal max err: {v.get('max_Y_vs_mass_err')}",
        f"- Identity max |Y11-Y00-(E_R+E_N+I)|: {v.get('identity_max_err')}",
        f"- Probe-array invariance max |H01-H00|: {v.get('probe_invariance_max_01')}",
        f"- Probe-array invariance max |H10-H11|: {v.get('probe_invariance_max_10')}",
        f"- Token-set Y from 61-way logits: `{v.get('token_level_Y')}`",
        "",
        "## 2. Natural-anchor sensitivity (not pooled across ε)",
        "",
    ]
    for eps in EPSILONS:
        block = (a.get("by_eps") or {}).get(str(eps), {})
        d = block.get("donor") or {}
        lines += [
            f"### ε = {eps}",
            "",
            f"- Donor-null rows/pairs: {d.get('n_rows')}/{d.get('n_pairs')}; targets={d.get('n_target_games')}",
            f"- Mean |ΔY|={d.get('mean_abs_dY')} median={d.get('median_abs_dY')} CI={_fmt(d.get('abs_dY_ci'))}",
            f"- Mean JS={d.get('mean_js')} CI={_fmt(d.get('js_ci'))}",
            f"- Mean achieved L2={d.get('mean_l2')}",
            f"- Invariance failures={d.get('invariance_fail')} max err={d.get('max_invariance_err')}",
            f"- Centered FD (Y+ − Y−)/(2ε): {block.get('finite_difference')}",
            f"- A: {block.get('donor_A')}",
            f"- B: {block.get('donor_B')}",
            f"- Paired vs random: {block.get('paired_vs_random')}",
            "",
        ]
    lines += [
        "These perturbations sit near a natural donor residual. That is not a proof they are on-manifold.",
        "",
        "## 3. Wrong-donor specificity",
        "",
        f"{w}",
        "",
        "If L2 is not matched, this is not a cost-matched comparison.",
        "",
        "## 4. SVD / uniqueness",
        "",
        f"- Rank {svd.get('numerical_rank')} nullspace {svd.get('nullspace_dim')} tolerance {svd.get('rank_tolerance')} sensitivity {svd.get('sensitivity')}",
        "- Δ_N is one vector in that nullspace. It is not a uniquely identified board-state coordinate, and the 388-dimensional nullspace is not a new board representation.",
        "",
        "## 5. Claims",
        "",
        f"### A. Continuous-probe insufficiency: **{claims['A_continuous_probe_insufficiency']}**",
        "",
        "Preserved from the original FIBER run after raw-array verification. Two activations with matching complete probe probabilities produce different F_x.",
        "",
        f"### B. Local causal sensitivity near natural activations: **{claims['B_natural_anchor_sensitivity']}**",
        "",
        "Primary scale is ε=1. Larger ε are reported separately and are not pooled into this verdict.",
        "",
        f"### C. Target-specific missing mathematical state: **{claims['C_target_specific_missing_state']}**",
        "",
        "Not inferred from A or B. Requires donor-derived null directions to beat random-null and wrong-donor controls on the intended legal set.",
        "",
        "## Limitations",
        "",
        "- Hybrid H01/H10 activations can be off-manifold; their large L2 is not evidence of a natural germ.",
        "- Recipient-game CIs on category A use 24 recipients and 2 targets.",
        "- History, prefix length, and policy features can sit in the residual even when the board probe is matched.",
        "- Wrong-donor JS was not stored.",
        "- Rank is a numerical truncation of W, not a unique causal basis.",
        "",
        f"## Verdicts",
        "",
        f"- Claim A: `{claims['A_continuous_probe_insufficiency']}`",
        f"- Claim B: `{claims['B_natural_anchor_sensitivity']}`",
        f"- Claim C: `{claims['C_target_specific_missing_state']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    fiber = Path(args.fiber_dir)
    out_dir = Path(args.out_dir) if args.out_dir else fiber
    if out_dir.resolve() != fiber.resolve():
        out_dir.mkdir(parents=True, exist_ok=True)
    tols = dict(DEFAULT_TOLS)
    counts = {name: {"exists": int((fiber / name).is_file()), "bytes": int((fiber / name).stat().st_size) if (fiber / name).is_file() else 0} for name in ORIGINAL}
    missing = [k for k, v in counts.items() if not v["exists"]]
    fact = pd.read_csv(fiber / "08r2_fiber_factorial_interventions.csv") if counts["08r2_fiber_factorial_interventions.csv"]["exists"] else pd.DataFrame()
    matched = pd.read_csv(fiber / "08r2_fiber_matched_pairs.csv") if counts["08r2_fiber_matched_pairs.csv"]["exists"] else pd.DataFrame()
    anchor = pd.read_csv(fiber / "08r2_fiber_natural_anchor.csv") if counts["08r2_fiber_natural_anchor.csv"]["exists"] else pd.DataFrame()
    controls = pd.read_csv(fiber / "08r2_fiber_controls.csv") if counts["08r2_fiber_controls.csv"]["exists"] else pd.DataFrame()
    svd = json.loads((fiber / "08r2_fiber_svd_report.json").read_text()) if counts["08r2_fiber_svd_report.json"]["exists"] else {}
    status = json.loads((fiber / "08r2_fiber_completion_status.json").read_text()) if counts["08r2_fiber_completion_status.json"]["exists"] else {}
    if "r2_type" in matched.columns:
        matched = matched[matched["r2_type"].isin(PRIMARY)].copy()
    if "r2_type" in fact.columns:
        fact = fact[fact["r2_type"].isin(PRIMARY)].copy()
    logit_map = load_npz_maps(fiber / "08r2_fiber_full_logits.npz")
    probe_map = load_npz_maps(fiber / "08r2_fiber_probe_probabilities.npz")
    fact_int = factorial_integrity(fact)
    logits = verify_logits(logit_map)
    probe = verify_probe_arrays(probe_map, float(tols["probe_proba_atol_f32"]))
    recomputed = recompute_from_logits(logit_map, matched) if len(matched) and "Y00" in matched.columns else pd.DataFrame()
    js_err_01 = float(np.nanmax((_num(recomputed["js_00_01_recomputed"]) - _num(recomputed["js_00_01_saved"])).abs())) if len(recomputed) else float("nan")
    js_err_11 = float(np.nanmax((_num(recomputed["js_10_11_recomputed"]) - _num(recomputed["js_10_11_saved"])).abs())) if len(recomputed) else float("nan")
    ident_max = float(np.nanmax(_num(recomputed["identity_err"]))) if len(recomputed) else float("nan")
    verif = {
        **fact_int,
        "n_pairs_four_cells": fact_int["n_pairs_with_four_cells"],
        "logits": logits,
        "js_00_01_max_abs_err": js_err_01,
        "js_10_11_max_abs_err": js_err_11,
        "identity_max_err": ident_max,
        "probe_invariance_max_01": probe.get("invariance_max_err_01"),
        "probe_invariance_max_10": probe.get("invariance_max_err_10"),
        "probe_invariance_ok": probe.get("invariance_ok"),
        "n_demo_saved": int(_num(matched.get("demonstrates_insufficiency", 0)).fillna(0).sum()) if len(matched) else 0,
        "token_level_Y": "NOT_IDENTIFIABLE_FROM_SAVED_DATA legal square sets were not stored in the fiber CSVs",
        "missing_original_files": missing,
    }
    anchor_sum = summarize_anchor(anchor, args.n_boot, args.boot_seed)
    wrong = summarize_wrong_donor(controls, matched, args.n_boot, args.boot_seed)
    claims = decide_claims(verif, anchor_sum, wrong, tols)
    if len(recomputed):
        recomputed.to_csv(out_dir / "08r2_fiber_specificity_pair_verification.csv", index=False)
    if len(anchor):
        keep = [c for c in anchor.columns if c in {
            "r2_type", "recip_game", "target_game", "family", "eps", "sign", "draw",
            "dY_vs_donor", "js_vs_donor", "l2_to_donor", "invariance_ok", "invariance_err",
            "became_legal_mass", "became_illegal_mass", "Y", "kl_to_base",
        }]
        anchor[keep].to_csv(out_dir / "08r2_fiber_specificity_anchor_extract.csv", index=False)
    payload = {
        "verification": verif,
        "anchor": anchor_sum,
        "wrong_donor": wrong,
        "claims": claims,
        "svd": {k: svd.get(k) for k in ("numerical_rank", "nullspace_dim", "rank_tolerance", "sensitivity")},
        "file_counts": counts,
        "row_counts": {
            "factorial": int(len(fact)),
            "matched": int(len(matched)),
            "anchor": int(len(anchor)),
            "controls": int(len(controls)),
            "recomputed_pairs": int(len(recomputed)),
        },
        "fiber_dir": str(fiber),
        "out_dir": str(out_dir),
        "overwrote_original": 0,
        "official_0A_letter": "C / PARTIAL",
        "section_5_3_status": "PARTIAL",
        "obstruction_ready": "NO",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "fiber_status_preserved": status.get("status"),
        "fiber_verdict_preserved": status.get("verdict"),
        "tolerances": tols,
    }
    write_json(out_dir / "08r2_fiber_specificity_summary.json", payload)
    write_report(out_dir / "08r2_fiber_specificity_scientific_report.md", payload)
    print("08R2-FIBER specificity audit written")
    print(f"  fiber_dir={fiber}")
    print(f"  out_dir={out_dir}")
    print(f"  files={ {k: v['exists'] for k, v in counts.items()} }")
    print(f"  rows={payload['row_counts']}")
    print(f"  four_cells={verif['n_pairs_four_cells']} identity_max_err={verif['identity_max_err']}")
    print(f"  probe_inv_max_01={verif['probe_invariance_max_01']} probe_inv_max_10={verif['probe_invariance_max_10']}")
    print(f"  js_recompute_max_err_00_01={verif['js_00_01_max_abs_err']}")
    print(f"  claim_A={claims['A_continuous_probe_insufficiency']}")
    print(f"  claim_B={claims['B_natural_anchor_sensitivity']}")
    print(f"  claim_C={claims['C_target_specific_missing_state']}")
    print("  official letter unchanged: C / PARTIAL")
    print("  08R2_VALIDATION_COMPLETE — NEGATIVE")
    print("  SECTION_5_3 = PARTIAL")
    print("  OBSTRUCTION_READY = NO")


if __name__ == "__main__":
    main()
