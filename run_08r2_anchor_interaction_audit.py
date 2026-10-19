"""Read-only anchor-interaction audit of 08R2-SYMMETRIC-TARGET-FRESH.

Does not overwrite Stage 5/6 files. No Othello-GPT. No confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from run_08r2_symmetric_target import _load_packs, _resolve_sibling, merge_cfg
from src.othello_r2 import OFFICIAL_0A_LETTER
from src.othello_r2_anchor_interaction import (
    ANCHORS,
    BOOT_SEED,
    FRESH_TRIPLET_HASH,
    LOG_EPS,
    MODELS,
    N_BOOT,
    PUBLISHED,
    REQUIRED_FRESH,
    STATUS_MISMATCH,
    Y_ATOL,
    contrast_from_ys,
    isolate_post_dir,
    legal_stats,
    mean_log_label_prob,
    pearson,
    published_match,
    zhat_target_diff,
)
from src.othello_r2_fiber_cost import adopt_drive_defaults
from src.othello_r2_fiber_spec import y_components
from src.othello_r2_null_state import symbolic_state
from src.othello_r2_state_semantics import paired_boot
from src.othello_runutil import add_labels, write_json
from src.toy_othello import load_config, softmax_np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-ANCHOR-INTERACTION-AUDIT")
    p.add_argument("--config", default="configs/toy_othello_08r2_symmetric_target_fresh.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--fresh-dir", default=None)
    p.add_argument("--nst-dir", default=None)
    p.add_argument("--sem-dir", default=None)
    return p.parse_args()


def _fresh_dir(cfg, args, results_dir: Path) -> Path:
    cands = []
    if args.fresh_dir:
        cands.append(Path(args.fresh_dir))
    cands.append(results_dir / "results" / "08r2" / "symmetric_target_fresh")
    cands.append(Path(cfg["results_dir"]) / "results" / "08r2" / "symmetric_target_fresh")
    for path in cands:
        if path.is_dir() and all((path / f).is_file() for f in REQUIRED_FRESH):
            return path
    return cands[0]


def _rows_jsonl(path: Path) -> list[dict]:
    return add_labels([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _cal_map(cals: list[dict]) -> dict:
    out = {}
    for r in cals:
        out[(r.get("recip_game"), r.get("cell"))] = r
    return out


def main() -> None:
    args = parse_args()
    cfg = merge_cfg(load_config(args.parent_config), load_config(args.r2_config), load_config(args.config))
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    adopt_drive_defaults(cfg, args)
    fresh = _fresh_dir(cfg, args, Path(cfg["results_dir"]))
    out_dir = ensure_dir(isolate_post_dir(fresh)["results_dir"])
    for name in REQUIRED_FRESH:
        if not (fresh / name).is_file():
            write_json(out_dir / "anchor_interaction_integrity.json", {"ok": 0, "reason": f"missing:{name}"})
            raise SystemExit(STATUS_MISMATCH)
    man = json.loads((fresh / "fresh_triplet_manifest.json").read_text(encoding="utf-8"))
    prefixes = _rows_jsonl(fresh / "fresh_cohort_prefixes.jsonl")
    inter = list(csv.DictReader((fresh / "fresh_interventions.csv").open(encoding="utf-8")))
    cals = list(csv.DictReader((fresh / "fresh_calibrations.csv").open(encoding="utf-8")))
    logits = np.load(fresh / "fresh_full_logits.npz")
    probes = np.load(fresh / "fresh_probe_probabilities.npz")
    trips = man["triplets"]
    recip = {t["recip_game"] for t in trips}
    targets = {t["target_t_game"] for t in trips} | {t["target_w_game"] for t in trips}
    union = recip | targets
    hash_ok = str(man.get("triplet_sha256")) == FRESH_TRIPLET_HASH
    design_ok = int(man.get("n_a") == 16 and man.get("n_b") == 16 and len(recip) == 32 and len(targets) == 64 and len(union) == 96)
    nst = _resolve_sibling(cfg, args, fresh, "null_state_transfer", "nst_dir", ("null_state_pca_ridge.npz",))
    sem = _resolve_sibling(cfg, args, fresh, "state_semantics", "sem_dir", ("state_semantics_stage_only.npz",))
    packs = None
    if (nst / "null_state_pca_ridge.npz").is_file() and (sem / "state_semantics_stage_only.npz").is_file():
        packs = _load_packs(nst, sem)

    n_match = n_bad = 0
    by_trip = {t["recip_game"]: t for t in trips}
    for row in inter:
        if row.get("cell") == "triplet_contrast" or row.get("Y_T") in (None, ""):
            continue
        trip = by_trip.get(row.get("recip_game"))
        if not trip:
            continue
        rec, tgt = prefixes[int(trip["recip"])], prefixes[int(trip["target_t"])]
        lkey = f"{row['recip_game']}__{row['cell']}"
        if lkey not in logits.files:
            n_bad += 1
            continue
        recon = y_components(softmax_np(logits[lkey]), rec["legal"], tgt["legal"])
        try:
            ok = abs(float(row["Y_T"]) - float(recon["Y"])) <= Y_ATOL
        except (TypeError, ValueError):
            ok = False
        n_match += int(ok)
        n_bad += int(not ok)

    matrix, contrasts = [], []
    calmap = _cal_map(cals)
    ident_rows, scale_rows, vis_rows = [], [], []
    zhat_ok = 1
    max_zhat_pair_err = 0.0
    for trip in trips:
        rec, tgt, wrg = prefixes[int(trip["recip"])], prefixes[int(trip["target_t"])], prefixes[int(trip["target_w"])]
        meta = {"r2_type": trip["r2_type"], "recip_game": trip["recip_game"], "target_t_game": trip["target_t_game"], "target_w_game": trip["target_w_game"], **legal_stats(rec, tgt, wrg)}
        if packs is not None:
            b_t, b_w = symbolic_state(tgt), symbolic_state(wrg)
            d_full = zhat_target_diff(b_t, b_w, packs["FULL"])
            ident_rows.append({"recip_game": trip["recip_game"], "quantity": "zhat_T_minus_zhat_W_FULL", "l2": float(np.linalg.norm(d_full)), "anchor_independent": 1, "activations_saved": 0, "note": "Algebraic delta_T-delta_W equals this for every a. Raw a not saved."})
        for model in MODELS:
            for anc in ANCHORS:
                ht = next((r for r in inter if r.get("cell") == f"{model}_{anc}_t" and r.get("recip_game") == trip["recip_game"]), None)
                hw = next((r for r in inter if r.get("cell") == f"{model}_{anc}_w" and r.get("recip_game") == trip["recip_game"]), None)
                xo_row = next((r for r in inter if r.get("cell") == "triplet_contrast" and r.get("model") == model and r.get("anchor") == anc and r.get("recip_game") == trip["recip_game"]), None)
                if not ht or not hw:
                    continue
                y_t_ht, y_t_hw = float(ht["Y_T"]), float(hw["Y_T"])
                y_w_ht, y_w_hw = float(ht["Y_W"]), float(hw["Y_W"])
                xo = contrast_from_ys(y_t_ht, y_t_hw, y_w_hw, y_w_ht)
                if xo_row:
                    for k in ("A_T", "A_W", "C"):
                        if abs(float(xo_row[k]) - xo[k]) > Y_ATOL:
                            zhat_ok = 0
                            max_zhat_pair_err = max(max_zhat_pair_err, abs(float(xo_row[k]) - xo[k]))
                ct = calmap.get((trip["recip_game"], f"{model}_{anc}_t"), {})
                cw = calmap.get((trip["recip_game"], f"{model}_{anc}_w"), {})
                rec_m = {
                    **meta, "model": model, "anchor": anc,
                    "Y_T_ht": y_t_ht, "Y_T_hw": y_t_hw, "Y_W_ht": y_w_ht, "Y_W_hw": y_w_hw,
                    **xo,
                    "js_t": ct.get("js"), "js_w": cw.get("js"), "t_t": ct.get("t"), "t_w": cw.get("t"),
                    "l2_t": ct.get("l2"), "l2_w": cw.get("l2"),
                    "G_T_ht": ht.get("G_T"), "Q_T_ht": ht.get("Q_T"),
                    "Y_anchor_T": xo_row.get("Y_anchor_T") if xo_row else None,
                    "Y_anchor_W": xo_row.get("Y_anchor_W") if xo_row else None,
                    "board_exact_T": xo_row.get("board_exact_T") if xo_row else None,
                    "board_exact_W": xo_row.get("board_exact_W") if xo_row else None,
                }
                matrix.append(rec_m)
                if model == "FULL":
                    contrasts.append(rec_m)
                    scale_rows.append({
                        "recip_game": trip["recip_game"], "anchor": anc, "r2_type": trip["r2_type"],
                        "t_T": ct.get("t"), "t_W": cw.get("t"),
                        "l2_T": ct.get("l2"), "l2_W": cw.get("l2"),
                        "js_T": ct.get("js"), "js_W": cw.get("js"),
                        "t_ratio_T_over_W": (float(ct["t"]) / float(cw["t"])) if ct.get("t") not in (None, "") and cw.get("t") not in (None, "") and float(cw["t"]) != 0 else None,
                        "raw_direction_identifiable": 0,
                    })
        for anc in ANCHORS:
            pkey = f"{trip['recip_game']}__{anc}__anchor"
            if pkey not in probes.files:
                vis_rows.append({**meta, "anchor": anc, "identifiable": 0})
                continue
            k_t = mean_log_label_prob(probes[pkey], tgt["rel_lab"], LOG_EPS)
            k_w = mean_log_label_prob(probes[pkey], wrg["rel_lab"], LOG_EPS)
            pref = k_t - k_w
            full = next((r for r in contrasts if r["recip_game"] == trip["recip_game"] and r["anchor"] == anc), None)
            vis_rows.append({
                **meta, "anchor": anc, "identifiable": 1, "log_eps": LOG_EPS,
                "K_T": k_t, "K_W": k_w, "P": pref,
                "A_T": full["A_T"] if full else None, "A_W": full["A_W"] if full else None,
                "C": full["C"] if full else None, "both_positive": full["both_positive"] if full else None,
                "cell": full["cell"] if full else None,
                "Y_anchor_T": full.get("Y_anchor_T") if full else None,
                "Y_anchor_W": full.get("Y_anchor_W") if full else None,
            })

    full_n = [r for r in contrasts if r["anchor"] == "N"]
    if len(full_n) != 32:
        write_json(out_dir / "anchor_interaction_integrity.json", {"ok": 0, "reason": f"full_n={len(full_n)}"})
        raise SystemExit(STATUS_MISMATCH)
    recon_means = {
        "n": 32,
        "A_T": float(np.mean([r["A_T"] for r in full_n])),
        "A_W": float(np.mean([r["A_W"] for r in full_n])),
        "C": float(np.mean([r["C"] for r in full_n])),
        "both_positive_frac": float(np.mean([r["both_positive"] for r in full_n])),
    }
    pub = published_match(recon_means)
    counts = Counter(r["cell"] for r in full_n)
    integrity = {
        "ok": int(pub["ok"] and n_bad == 0 and n_match > 0 and hash_ok and design_ok and zhat_ok),
        "triplet_hash": man.get("triplet_sha256"),
        "triplet_hash_ok": int(hash_ok),
        "design_ok": design_ok,
        "n_a": man.get("n_a"),
        "n_b": man.get("n_b"),
        "n_union": len(union),
        "stage6_recompute_match": n_match,
        "stage6_recompute_mismatch": n_bad,
        "published_match": pub,
        "natural_counts": dict(counts),
        "contrast_csv_vs_recompute_max_err": max_zhat_pair_err,
        "raw_activations_saved": 0,
        "raw_directions_saved": 0,
        "used_confirmation": 0,
        "status_preserved": "SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED",
    }
    write_json(out_dir / "anchor_interaction_integrity.json", integrity)
    if not integrity["ok"]:
        print(STATUS_MISMATCH)
        print(integrity)
        raise SystemExit(STATUS_MISMATCH)

    write_csv(out_dir / "anchor_response_matrix.csv", matrix)
    conting = []
    for name, pred in (
        ("all", lambda r: True),
        ("A", lambda r: r["r2_type"] == "A_legal_diff"),
        ("B", lambda r: r["r2_type"] == "B_similar_board_legal_change"),
    ):
        g = [r for r in full_n if pred(r)]
        c = Counter(r["cell"] for r in g)
        conting.append({"slice": name, "n": len(g), "both": c.get("both", 0), "t_only": c.get("t_only", 0), "w_only": c.get("w_only", 0), "neither": c.get("neither", 0)})
        for r in g:
            conting.append({"slice": f"{name}_row", **{k: r[k] for k in ("recip_game", "r2_type", "cell", "A_T", "A_W", "C", "stage_r", "occupied_t", "occupied_w", "hamming_tw", "legal_jaccard_tw", "n_diff_tw", "n_legal_t", "n_legal_w", "l2_t", "l2_w", "js_t", "js_w")}})
    write_csv(out_dir / "anchor_switching_contingency.csv", conting)
    write_csv(out_dir / "anchor_target_contrast_identity.csv", ident_rows or [{"activations_saved": 0, "note": "Raw a and delta not saved. Algebra says delta_T-delta_W=zhat(T)-zhat(W)."}])
    write_csv(out_dir / "anchor_scaling_diagnostics.csv", scale_rows)
    write_csv(out_dir / "visible_preference_diagnostics.csv", vis_rows)

    boot_rows = []
    by_anc = {a: [r for r in contrasts if r["anchor"] == a] for a in ANCHORS}
    for anc, g in by_anc.items():
        if len(g) != 32:
            continue
        for key in ("A_T", "A_W", "C"):
            boot_rows.append({"slice": f"FULL_{anc}", "metric": key, **paired_boot([r[key] for r in g], N_BOOT, BOOT_SEED)})
        boot_rows.append({"slice": f"FULL_{anc}", "metric": "both_positive_frac", "n": 32, "point": float(np.mean([r["both_positive"] for r in g]))})
        for typ, pred in (("A", lambda r: r["r2_type"] == "A_legal_diff"), ("B", lambda r: r["r2_type"] == "B_similar_board_legal_change")):
            sg = [r for r in g if pred(r)]
            if not sg:
                continue
            for key in ("A_T", "A_W", "C"):
                boot_rows.append({"slice": f"FULL_{anc}_{typ}", "metric": key, **paired_boot([r[key] for r in sg], N_BOOT, BOOT_SEED + 1)})
    nat = {r["recip_game"]: r for r in by_anc.get("N", [])}
    at = {r["recip_game"]: r for r in by_anc.get("T", [])}
    aw = {r["recip_game"]: r for r in by_anc.get("W", [])}
    common = sorted(set(nat) & set(at) & set(aw))
    if len(common) == 32:
        d_t = [at[g]["A_T"] - aw[g]["A_T"] for g in common]
        d_w = [aw[g]["A_W"] - at[g]["A_W"] for g in common]
        boot_rows.append({"slice": "D_T", "metric": "A_T(T)-A_T(W)", **paired_boot(d_t, N_BOOT, BOOT_SEED + 2)})
        boot_rows.append({"slice": "D_W", "metric": "A_W(W)-A_W(T)", **paired_boot(d_w, N_BOOT, BOOT_SEED + 3)})
        for name, vals in (
            ("A_T(T)-A_T(N)", [at[g]["A_T"] - nat[g]["A_T"] for g in common]),
            ("A_T(W)-A_T(N)", [aw[g]["A_T"] - nat[g]["A_T"] for g in common]),
            ("A_W(W)-A_W(N)", [aw[g]["A_W"] - nat[g]["A_W"] for g in common]),
            ("A_W(T)-A_W(N)", [at[g]["A_W"] - nat[g]["A_W"] for g in common]),
        ):
            boot_rows.append({"slice": "vs_natural", "metric": name, **paired_boot(vals, N_BOOT, BOOT_SEED + 4)})
    vis_n = [r for r in vis_rows if r.get("anchor") == "N" and r.get("identifiable") == 1]
    if vis_n:
        boot_rows.append({"slice": "visible_N", "metric": "corr_P_AT", "point": pearson([r["P"] for r in vis_n], [r["A_T"] for r in vis_n])})
        boot_rows.append({"slice": "visible_N", "metric": "corr_P_AW", "point": pearson([r["P"] for r in vis_n], [r["A_W"] for r in vis_n])})
        boot_rows.append({"slice": "visible_N", "metric": "corr_P_C", "point": pearson([r["P"] for r in vis_n], [r["C"] for r in vis_n])})
        boot_rows.append({"slice": "visible_N", "metric": "corr_P_both", "point": pearson([r["P"] for r in vis_n], [r["both_positive"] for r in vis_n])})
        boot_rows.append({"slice": "visible_N", "metric": "corr_hammingTW_C", "point": pearson([r["hamming_tw"] for r in vis_n], [r["C"] for r in vis_n])})
        boot_rows.append({"slice": "visible_N", "metric": "corr_jaccard_C", "point": pearson([r["legal_jaccard_tw"] for r in vis_n], [r["C"] for r in vis_n])})
        boot_rows.append({"slice": "visible_N", "metric": "corr_anchorYpref_C", "point": pearson(
            [float(r["Y_anchor_T"]) - float(r["Y_anchor_W"]) if r.get("Y_anchor_T") not in (None, "") and r.get("Y_anchor_W") not in (None, "") else np.nan for r in vis_n],
            [r["C"] for r in vis_n],
        )})
    write_csv(out_dir / "anchor_effect_bootstrap.csv", boot_rows)

    feat_rows = []
    for model in MODELS:
        g = [r for r in matrix if r["model"] == model and r["anchor"] == "N"]
        if len(g) != 32:
            feat_rows.append({"model": model, "n": len(g), "coverage": len(g)})
            continue
        feat_rows.append({
            "model": model, "n": 32, "coverage": 32,
            "A_T": float(np.mean([r["A_T"] for r in g])),
            "A_W": float(np.mean([r["A_W"] for r in g])),
            "C": float(np.mean([r["C"] for r in g])),
            "both_positive_frac": float(np.mean([r["both_positive"] for r in g])),
            "A_T_ci95_low": paired_boot([r["A_T"] for r in g], N_BOOT, BOOT_SEED)["ci95_low"],
            "A_T_ci95_high": paired_boot([r["A_T"] for r in g], N_BOOT, BOOT_SEED)["ci95_high"],
            "A_W_ci95_low": paired_boot([r["A_W"] for r in g], N_BOOT, BOOT_SEED)["ci95_low"],
            "A_W_ci95_high": paired_boot([r["A_W"] for r in g], N_BOOT, BOOT_SEED)["ci95_high"],
            "mean_js_t": float(np.mean([float(r["js_t"]) for r in g if r.get("js_t") not in (None, "")])),
            "mean_l2_t": float(np.mean([float(r["l2_t"]) for r in g if r.get("l2_t") not in (None, "")])),
            "both_A": float(np.mean([r["both_positive"] for r in g if r["r2_type"] == "A_legal_diff"])),
            "both_B": float(np.mean([r["both_positive"] for r in g if r["r2_type"] == "B_similar_board_legal_change"])),
        })
    write_csv(out_dir / "feature_map_switching_comparison.csv", feat_rows)

    d_t_boot = next((r for r in boot_rows if r.get("slice") == "D_T"), {})
    d_w_boot = next((r for r in boot_rows if r.get("slice") == "D_W"), {})
    corr_p_c = next((r for r in boot_rows if r.get("metric") == "corr_P_C"), {})
    feat_full = next((r for r in feat_rows if r.get("model") == "FULL"), {})
    feat_legal = next((r for r in feat_rows if r.get("model") == "LEGAL_ONLY"), {})
    feat_board = next((r for r in feat_rows if r.get("model") == "BOARD_ONLY"), {})
    report = [
        "# 08R2-ANCHOR-INTERACTION-AUDIT",
        "",
        "Read-only. Stage 5 verdict unchanged: `SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED`.",
        "Exploratory. No new pass/fail gate. Triplets are the unit. 5000 paired bootstrap.",
        "Raw activations and unscaled directions were not saved (`NOT_IDENTIFIABLE_FROM_SAVED_DATA`).",
        f"Visible preference uses mean log class probability, clip `{LOG_EPS}`.",
        "",
        "## Integrity",
        "",
        f"- Manifest hash ok: {hash_ok} `{man.get('triplet_sha256')}`",
        f"- Design 16A+16B / 96 games: {design_ok}",
        f"- Logit Y recompute: match={n_match} mismatch={n_bad}",
        f"- Published means reproduced: {pub['ok']} {recon_means}",
        f"- Natural contingency: {dict(counts)}",
        "",
        "## A. What caused the 19/32 joint-switching failures?",
        "",
        f"Natural FULL cells: both={counts.get('both', 0)}, T-only={counts.get('t_only', 0)}, W-only={counts.get('w_only', 0)}, neither={counts.get('neither', 0)}.",
        "See `anchor_switching_contingency.csv` for stage, occupancy, Hamming, legal overlap, L2, and JS of each triplet.",
        "Failures are the complement of both-positive. They were not dropped.",
        "",
        "## B. Does the probe-visible anchor change target discrimination?",
        "",
        f"D_T = A_T(T)-A_T(W): {d_t_boot}",
        f"D_W = A_W(W)-A_W(T): {d_w_boot}",
        "These are within-triplet anchor contrasts, not independent confirmation.",
        "",
        "## C. Is the raw T-vs-W direction anchor-independent?",
        "",
        "Algebra: delta_T(a)-delta_W(a)=zhat(T)-zhat(W) for every a.",
        "Numeric check of that difference on saved activations: NOT_IDENTIFIABLE_FROM_SAVED_DATA.",
        "zhat(T)-zhat(W) from frozen FULL is saved in `anchor_target_contrast_identity.csv`.",
        "",
        "## D. JS calibration / scale",
        "",
        "Independent t and L2 per anchor and direction are in `anchor_scaling_diagnostics.csv`.",
        "Applied displacements t*delta are not recoverable without delta.",
        "",
        "## E. Visible-state agreement vs causal effect",
        "",
        f"Natural-anchor Pearson P vs C: {corr_p_c.get('point')}",
        "P(a)=K(a,T)-K(a,W). Exploratory only. Not a COR compatibility score.",
        "",
        "## F. Feature maps on the natural anchor",
        "",
        f"FULL: {feat_full}",
        f"LEGAL_ONLY: {feat_legal}",
        f"BOARD_ONLY: {feat_board}",
        "",
        "## G. Future prospective compatibility test?",
        "",
        "This audit does not implement one. A future test would need saved activations or a new preregistered design.",
        "",
        "Preserved: SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED, SYMMETRIC_TARGET_CENSUS_INFEASIBLE,",
        "LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY, DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED,",
        "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED, TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED,",
        "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE, 08R2_VALIDATION_COMPLETE — NEGATIVE,",
        "0A C / PARTIAL, SECTION_5_3 = PARTIAL, OBSTRUCTION_READY = NO.",
        "",
    ]
    (out_dir / "anchor_interaction_scientific_report.md").write_text("\n".join(report), encoding="utf-8")
    write_json(out_dir / "anchor_interaction_completion.json", {
        "status": "ANCHOR_INTERACTION_AUDIT_COMPLETE",
        "verdict_unchanged": "SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED",
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "used_confirmation": 0,
        "exploratory": 1,
        "counts": dict(counts),
        "recon_means": recon_means,
    })
    print("08R2-ANCHOR-INTERACTION-AUDIT")
    print(f"  integrity_ok=1 hash_ok={int(hash_ok)} y_match={n_match} y_bad={n_bad}")
    print(f"  published={recon_means}")
    print(f"  natural_cells={dict(counts)}")
    print(f"  D_T={d_t_boot.get('point')} D_W={d_w_boot.get('point')}")
    print(f"  corr_P_C={corr_p_c.get('point')}")
    print("  SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED preserved")


if __name__ == "__main__":
    main()
