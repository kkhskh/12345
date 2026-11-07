"""Read-only 08R2-ANCHOR-INTERACTION-AUDIT of saved fresh-cohort outputs.

Does not rerun Othello-GPT or change Stage 5/6 verdicts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.othello_r2_state_semantics import paired_boot, predict_zhat
from src.othello_runutil import board_arr
from src.toy_othello import board_hamming


POST_DIRNAME = "postvalidation"
FRESH_TRIPLET_HASH = "3ae44e1eabf2684cb3e0b5025ccb9b76cabad6a1a6176454999ebc4b5ce70972"
PUBLISHED = {
    "A_T": 0.0284998707684192,
    "A_W": 0.01856145964513889,
    "C": 0.04706133041355809,
    "both_positive_frac": 0.40625,
    "n": 32,
}
LOG_EPS = 1e-12
Y_ATOL = 1e-6
MEAN_ATOL = 1e-9
N_BOOT = 5000
BOOT_SEED = 9582
STATUS_MISMATCH = "ANCHOR_INTERACTION_INTEGRITY_FAIL"
ANCHORS = ("N", "T", "W")
MODELS = ("FULL", "LEGAL_ONLY", "BOARD_ONLY", "STAGE_ONLY")
REQUIRED_FRESH = (
    "fresh_triplet_manifest.json",
    "fresh_cohort_prefixes.jsonl",
    "fresh_interventions.csv",
    "fresh_calibrations.csv",
    "fresh_full_logits.npz",
    "fresh_probe_probabilities.npz",
    "fresh_completion_status.json",
)
BANNED_OVERWRITE = {
    "fresh_interventions.csv",
    "fresh_calibrations.csv",
    "fresh_full_logits.npz",
    "fresh_probe_probabilities.npz",
    "fresh_triplet_manifest.json",
    "fresh_completion_status.json",
    "fresh_stage6_status.json",
}


def isolate_post_dir(fresh_dir: str | Path) -> dict:
    fresh = Path(fresh_dir)
    if fresh.name != "symmetric_target_fresh":
        raise ValueError(f"audit must attach to symmetric_target_fresh, got {fresh}")
    out = fresh / POST_DIRNAME
    if any(p == "postvalidation" and q != POST_DIRNAME for p, q in []):
        pass
    return {"fresh_dir": str(fresh), "results_dir": str(out)}


def mean_log_label_prob(probe_64x3: np.ndarray, rel_lab, eps: float = LOG_EPS) -> float:
    p = np.asarray(probe_64x3, dtype=np.float64).reshape(64, 3)
    lab = np.asarray(rel_lab, dtype=np.int64).reshape(-1)
    if lab.size != 64:
        raise ValueError("rel_lab must be 64")
    take = np.clip(p[np.arange(64), np.clip(lab, 0, 2)], float(eps), 1.0)
    return float(np.log(take).mean())


def switching_cell(a_t: float, a_w: float) -> str:
    tp, wp = a_t > 0, a_w > 0
    if tp and wp:
        return "both"
    if tp:
        return "t_only"
    if wp:
        return "w_only"
    return "neither"


def legal_stats(rec, tgt, wrg) -> dict:
    lr, lt, lw = set(rec["legal"]), set(tgt["legal"]), set(wrg["legal"])
    union_tw = lt | lw
    return {
        "n_legal_r": len(lr),
        "n_legal_t": len(lt),
        "n_legal_w": len(lw),
        "n_became_t": len(lt - lr),
        "n_became_w": len(lw - lr),
        "n_left_t": len(lr - lt),
        "n_left_w": len(lr - lw),
        "legal_jaccard_tw": float(len(lt & lw) / len(union_tw)) if union_tw else float("nan"),
        "n_diff_tw": len(lt ^ lw),
        "occupied_r": int(rec.get("occupied", -1)),
        "occupied_t": int(tgt.get("occupied", -1)),
        "occupied_w": int(wrg.get("occupied", -1)),
        "stage_r": rec.get("stage"),
        "hamming_rt": int(board_hamming(board_arr(rec), board_arr(tgt))),
        "hamming_rw": int(board_hamming(board_arr(rec), board_arr(wrg))),
        "hamming_tw": int(board_hamming(board_arr(tgt), board_arr(wrg))),
    }


def contrast_from_ys(y_t_ht, y_t_hw, y_w_hw, y_w_ht) -> dict:
    a_t = float(y_t_ht - y_t_hw)
    a_w = float(y_w_hw - y_w_ht)
    return {"A_T": a_t, "A_W": a_w, "C": a_t + a_w, "both_positive": int(a_t > 0 and a_w > 0), "cell": switching_cell(a_t, a_w)}


def zhat_target_diff(b_t: np.ndarray, b_w: np.ndarray, pack: dict) -> np.ndarray:
    return predict_zhat(b_t, pack) - predict_zhat(b_w, pack)


def pearson(x, y) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def published_match(recomputed: dict) -> dict:
    errs = {}
    ok = 1
    for k, exp in PUBLISHED.items():
        got = float(recomputed[k])
        err = abs(got - float(exp))
        errs[k] = {"got": got, "expected": float(exp), "abs_err": err}
        if err > MEAN_ATOL and k != "both_positive_frac":
            ok = 0
        if k == "both_positive_frac" and abs(got - float(exp)) > 1e-12:
            ok = 0
    return {"ok": int(ok), "errors": errs}
