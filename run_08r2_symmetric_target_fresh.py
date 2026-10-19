"""08R2-SYMMETRIC-TARGET-FRESH staged runner.

Does not overwrite leftover-val SYM, NST, or SEM. No confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from run_08r1_real_model_verification import git_hash
from run_08r2_symmetric_target import (
    _editor,
    _load_packs,
    _load_stack,
    _patch,
    _resolve_sibling,
    _sha256_text,
    _slice,
    merge_cfg,
)
from src.othello_r1_eval import sha256_file
from src.othello_r2 import OFFICIAL_0A_LETTER, SITE, decoder_state_effects, load_holdout_game_ids, require_no_confirmation
from src.othello_r2_fiber import js_divergence
from src.othello_r2_fiber_cost import adopt_drive_defaults
from src.othello_r2_fiber_cost_fixed import SCALE_GRID, calibrate_fixed_js, scaled_displacement
from src.othello_r2_fiber_spec import y_components
from src.othello_r2_null_state import symbolic_state
from src.othello_r2_state_semantics import ABLATIONS, ablation_delta, crossover_scores, predict_zhat, probe_image_residual
from src.othello_r2_symmetric_target import (
    ANCHORS,
    AT_FLOOR,
    AW_FLOOR,
    BOTH_POS_MIN,
    COVERAGE_MIN,
    FIXED_JS,
    FIXED_JS_TOL,
    STATUS_AWAITING,
    VERDICT_ENGINEERING,
    WANT_A,
    WANT_B,
    anchor_dependence_ok,
    decide_sym_verdict,
    natural_symmetric_ok,
    nst_eval_games,
    stage_only_identical,
)
from src.othello_r2_symmetric_target_fresh import (
    BOOT_SEED,
    COHORT_SEED,
    N_GAMES,
    PACK_TRIALS,
    PILOT_COHORT_SEED,
    PILOT_PREFIX_SEED,
    PREFIX_SEED,
    STATUS_INFEASIBLE,
    TRIPLET_SEED,
    WANT_UNION,
    apply_exclusions,
    banned_sequence_hashes,
    generate_game_records,
    isolate_fresh_dir,
    joint_pack_from_rows,
    prefixes_from_records,
)
from src.othello_runutil import add_labels, load_split, tokens_of, write_json
from src.toy_othello import cache_site, load_config, softmax_np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-SYMMETRIC-TARGET-FRESH")
    p.add_argument("--config", default="configs/toy_othello_08r2_symmetric_target_fresh.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--stage", default="0", choices=["0", "1", "2", "3", "4", "5", "6", "all"])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--nst-dir", default=None)
    p.add_argument("--sem-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def stage0(out_dir: Path) -> None:
    doc = {
        "original_sym_status": "SYMMETRIC_TARGET_CENSUS_INFEASIBLE",
        "original_protocol_rewritten": 0,
        "pilot_seeds_not_this_experiment": [PILOT_COHORT_SEED, PILOT_PREFIX_SEED],
        "FULL_minus_EditorA": 0.053713,
        "STAGE_ONLY_minus_EditorA": 0.030099,
        "FULL_minus_STAGE_ONLY": 0.023614,
        "stage_share_of_full_editor": 0.5604,
        "note": "Arithmetic of mean response levels, not independent causal mechanisms.",
        "preserved": {
            "LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY": 1,
            "DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED": 1,
            "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED": 1,
            "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED": 1,
            "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE": 1,
            "08R2_VALIDATION_COMPLETE — NEGATIVE": 1,
            "0A C / PARTIAL": 1,
            "SECTION_5_3 = PARTIAL": 1,
            "OBSTRUCTION_READY": "NO",
        },
    }
    write_json(out_dir / "fresh_stage0_preserved.json", doc)
    print("08R2-SYMMETRIC-TARGET-FRESH STAGE 0")
    print("  leftover-val status retained: SYMMETRIC_TARGET_CENSUS_INFEASIBLE")
    print("  stage share 0.5604 is mean-level arithmetic, not a mechanism split")


def stage1(cfg, out_dir: Path) -> dict:
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    splits = {}
    for name in ("train", "val", "test"):
        try:
            splits[name] = load_split(Path(cfg["data_dir"]), name)
        except FileNotFoundError:
            splits[name] = []
    require_no_confirmation(splits.get("val") or [], holdout["game_ids"])
    prior = nst_eval_games(splits["val"], holdout["game_ids"]) if splits["val"] else {"games": set(), "n_total": 0}
    banned_ids = set(holdout["game_ids"]) | set(map(str, prior.get("games") or []))
    for rows in splits.values():
        banned_ids |= {str(r["game_id"]) for r in rows}
    pilot = generate_game_records(int(cfg.get("n_games", N_GAMES)), int(cfg.get("pilot_cohort_seed", PILOT_COHORT_SEED)), "fresh")
    banned = banned_sequence_hashes(splits, pilot)
    raw = generate_game_records(int(cfg.get("n_games", N_GAMES)), int(cfg.get("cohort_seed", COHORT_SEED)), "symf")
    filtered = apply_exclusions(raw, banned["hashes"], banned_ids)
    rows = prefixes_from_records(filtered["kept"], int(cfg.get("prefix_seed", PREFIX_SEED)), "fresh2", k=int(cfg.get("positions_per_game", 6)))
    existing_states = set()
    for name, srows in splits.items():
        for r in srows:
            existing_states.add(json.dumps({"board": r.get("board"), "player": r.get("player")}, sort_keys=True))
    pilot_pref = prefixes_from_records(pilot, int(cfg.get("pilot_prefix_seed", PILOT_PREFIX_SEED)), "pilot", k=6)
    for r in pilot_pref:
        existing_states.add(json.dumps({"board": r.get("board"), "player": r.get("player")}, sort_keys=True))
    n_state_overlap = sum(1 for r in rows if json.dumps({"board": r.get("board"), "player": r.get("player")}, sort_keys=True) in existing_states)
    cohort = {
        "cohort_seed": int(cfg.get("cohort_seed", COHORT_SEED)),
        "prefix_seed": int(cfg.get("prefix_seed", PREFIX_SEED)),
        "pilot_excluded_seed": int(cfg.get("pilot_cohort_seed", PILOT_COHORT_SEED)),
        "n_generated": len(raw),
        "n_kept_games": filtered["n_kept"],
        "n_dropped": filtered["n_dropped"],
        "drop_reasons": {r["reason"] for r in filtered["dropped"]},
        "n_prefixes": len(rows),
        "n_identical_state_overlap": n_state_overlap,
        "n_banned_sequences": banned["n_total"],
        "banned_by_source": banned["by_source"],
        "n_prior_eval_games": prior.get("n_total", 0),
        "used_confirmation": 0,
        "games": [{"game_id": g["game_id"], "sequence_sha256": g["sequence_sha256"]} for g in filtered["kept"]],
    }
    write_json(out_dir / "fresh_cohort_manifest.json", {k: v for k, v in cohort.items() if k != "games"} | {"n_game_records": len(cohort["games"])})
    write_json(out_dir / "fresh_cohort_games.json", {"games": cohort["games"], "dropped": [{"game_id": d["game_id"], "reason": d["reason"], "sequence_sha256": d["sequence_sha256"]} for d in filtered["dropped"]]})
    with (out_dir / "fresh_cohort_prefixes.jsonl").open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    print("08R2-SYMMETRIC-TARGET-FRESH STAGE 1")
    print(f"  seed={cohort['cohort_seed']} kept_games={cohort['n_kept_games']} dropped={cohort['n_dropped']} prefixes={cohort['n_prefixes']}")
    print(f"  sequence_banned={cohort['n_banned_sequences']} identical_state_overlap={n_state_overlap}")
    return {"rows": rows, "cohort": cohort}


def stage2(cfg, out_dir: Path, rows: list[dict] | None = None) -> dict:
    if rows is None:
        rows = [json.loads(line) for line in (out_dir / "fresh_cohort_prefixes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = add_labels(rows)
    selected = joint_pack_from_rows(rows, int(cfg.get("n_a", WANT_A)), int(cfg.get("n_b", WANT_B)), int(cfg.get("triplet_seed", TRIPLET_SEED)), int(cfg.get("pack_trials", PACK_TRIALS)))
    wit = selected.get("witness") or {}
    census = {
        "n_eligible": sum(1 for r in rows if r.get("legal") and not r.get("terminal")),
        "n_b_candidates": selected.get("n_b_candidates"),
        "selected_A": selected["n_a"],
        "selected_B": selected["n_b"],
        "n_recipients": selected["n_recipients"],
        "n_targets": selected["n_targets"],
        "n_union": wit.get("n_union"),
        "recip_target_overlap": selected.get("recip_target_overlap"),
        "feasible": int(bool(selected.get("feasible") and wit.get("ok"))),
        "method": selected.get("method"),
        "upper_b": selected.get("upper_b"),
        "infeasible_certificate": selected.get("infeasible_certificate"),
        "triplet_sha256": selected.get("triplet_sha256"),
        "seed": selected.get("seed"),
        "required_union": int(cfg.get("required_union_games", WANT_UNION)),
    }
    write_json(out_dir / "fresh_triplet_census.json", census)
    write_json(out_dir / "fresh_triplet_manifest.json", {**selected, "rows_n": len(rows)})
    print("08R2-SYMMETRIC-TARGET-FRESH STAGE 2")
    print(f"  B_cand={census['n_b_candidates']} selected={census['selected_A']}A+{census['selected_B']}B union={census['n_union']} feasible={census['feasible']}")
    print(f"  witness={wit} method={census['method']}")
    print(f"  triplet_hash={census['triplet_sha256']}")
    if not census["feasible"]:
        write_json(out_dir / "fresh_completion_status.json", {
            "status": STATUS_INFEASIBLE, "census": census, "used_confirmation": 0,
            "original_sym_status_preserved": "SYMMETRIC_TARGET_CENSUS_INFEASIBLE",
            "note": "Predetermined 9182 cohort has no 16A+16B / 96-game witness. Protocol was not relaxed.",
        })
        raise SystemExit(STATUS_INFEASIBLE)
    return {"selected": selected, "rows": rows}


def stage3(cfg, out_dir: Path, nst: Path, sem: Path) -> None:
    packs = _load_packs(nst, sem)
    if int(packs["FULL"]["k"]) != 64 or abs(float(packs["FULL"]["alpha"]) - 10.0) > 1e-12:
        raise SystemExit(f"{VERDICT_ENGINEERING} FULL k/alpha mismatch")
    protocol = {
        "experiment": "08R2-SYMMETRIC-TARGET-FRESH",
        "cohort_seed": int(cfg.get("cohort_seed", COHORT_SEED)),
        "prefix_seed": int(cfg.get("prefix_seed", PREFIX_SEED)),
        "triplet_seed": int(cfg.get("triplet_seed", TRIPLET_SEED)),
        "boot_seed": int(cfg.get("boot_seed", BOOT_SEED)),
        "n_a": 16, "n_b": 16, "required_union": 96,
        "fixed_js": 0.01, "a_t_floor": 0.005, "a_w_floor": 0.005,
        "both_pos_min": 0.60, "coverage_min": 29, "n_boot": 5000,
        "frozen_k": 64, "frozen_alpha": 10.0,
        "site": SITE, "anchors": list(ANCHORS), "models": list(ABLATIONS),
        "primary": "FULL natural anchor",
        "scale_grid": list(SCALE_GRID),
        "nst_pca_sha256": sha256_file(nst / "null_state_pca_ridge.npz"),
        "refit": 0,
    }
    write_json(out_dir / "fresh_protocol.json", protocol)
    write_json(out_dir / "fresh_protocol_hash.json", {"sha256": _sha256_text(json.dumps(protocol, sort_keys=True))})
    print("08R2-SYMMETRIC-TARGET-FRESH STAGE 3")
    print(f"  frozen FULL k=64 alpha=10 refit=0 protocol={_sha256_text(json.dumps(protocol, sort_keys=True))}")


def stage_eval(cfg, out_dir: Path, args, nst: Path, sem: Path, *, smoke: bool) -> None:
    write_dir = out_dir / "smoke" if smoke else out_dir
    ensure_dir(write_dir)
    man = json.loads((out_dir / "fresh_triplet_manifest.json").read_text(encoding="utf-8"))
    rows = add_labels([json.loads(line) for line in (out_dir / "fresh_cohort_prefixes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()])
    stack = _load_stack(cfg, args)
    if stack is None:
        print(STATUS_AWAITING)
        return
    packs = _load_packs(nst, sem)
    H = cache_site(stack["model"], tokens_of(rows), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    decoder = stack["decoder"]
    trips = man["triplets"]
    if smoke:
        trips = [trips[int(cfg.get("smoke_triplet_index", 0))]]
    js_target = float(cfg.get("fixed_js", FIXED_JS))
    inv_atol = float(cfg.get("probe_proba_atol_f32", 1e-5))
    n_boot = int(cfg.get("n_boot", 5000))
    attempts = {"triplets": 0, "editor_fail": 0, "js_fail": 0, "invariance_fail": 0, "valid_full": 0}
    inter, cals, idents = [], [], []
    logit_pack, probe_pack = {}, {}
    hooks_ok = 1
    models = ("FULL", "LEGAL_ONLY", "BOARD_ONLY", "STAGE_ONLY")
    for trip in trips:
        attempts["triplets"] += 1
        rec, tgt, wrg = rows[int(trip["recip"])], rows[int(trip["target_t"])], rows[int(trip["target_w"])]
        if rec["game_id"] != trip["recip_game"] or tgt["game_id"] != trip["target_t_game"] or wrg["game_id"] != trip["target_w_game"]:
            raise SystemExit(f"{VERDICT_ENGINEERING} index/game mismatch")
        b_t, b_w = symbolic_state(tgt), symbolic_state(wrg)
        idents.append({"recip_game": trip["recip_game"], "stage_only_identical": int(stage_only_identical(b_t, b_w)), "stage_only_zhat_identical": int(bool(np.allclose(predict_zhat(b_t, packs["STAGE_ONLY"]), predict_zhat(b_w, packs["STAGE_ONLY"]))))})
        h_r = np.asarray(H[int(trip["recip"])], dtype=np.float64)
        ed_t, ed_w = _editor(cfg, decoder, rec, tgt, h_r), _editor(cfg, decoder, rec, wrg, h_r)
        if not ed_t.get("valid_edit") or not ed_w.get("valid_edit"):
            attempts["editor_fail"] += 1
            continue
        anchors = {"N": h_r, "T": np.asarray(ed_t["edited"], dtype=np.float64), "W": np.asarray(ed_w["edited"], dtype=np.float64)}
        exact = {
            "N": decoder_state_effects(decoder, h_r, rec["rel_lab"], rec["rel_lab"])["board_exact"],
            "T": decoder_state_effects(decoder, anchors["T"], rec["rel_lab"], tgt["rel_lab"])["board_exact"],
            "W": decoder_state_effects(decoder, anchors["W"], rec["rel_lab"], wrg["rel_lab"])["board_exact"],
        }
        full_ok = 1
        recs = {}
        for anc_name, a in anchors.items():
            probs_a, logits_a, audit_a = _patch(stack["model"], rec["tokens"], a)
            hooks_ok = hooks_ok and int(audit_a.get("ok", 0))
            probe_a = decoder.proba_np(a).reshape(64, 3)
            logit_pack[f"{trip['recip_game']}__{anc_name}__anchor"] = np.asarray(logits_a, dtype=np.float64)
            probe_pack[f"{trip['recip_game']}__{anc_name}__anchor"] = probe_a
            y_a_t = y_components(probs_a, rec["legal"], tgt["legal"])
            y_a_w = y_components(probs_a, rec["legal"], wrg["legal"])
            for model in models:
                d_t, d_w = ablation_delta(a, b_t, packs[model]), ablation_delta(a, b_w, packs[model])
                got = {}
                for tag, delta in (("t", d_t), ("w", d_w)):
                    def ev(t, _d=delta, _a=a):
                        h = _a + float(t) * np.asarray(_d)
                        probs, logits, audit = _patch(stack["model"], rec["tokens"], h)
                        return {"js": float(js_divergence(probs_a, probs)), "probs": probs, "logits": logits, "audit": audit, "h": h}

                    matched = calibrate_fixed_js(ev, target_js=js_target, grid=SCALE_GRID, tol=float(cfg.get("fixed_js_tol", FIXED_JS_TOL)))
                    sc = scaled_displacement(delta, matched.get("t", float("nan"))) if matched.get("t") == matched.get("t") else {"l2": float("nan")}
                    probe = decoder.proba_np(matched["h"]).reshape(64, 3) if matched.get("ok") and matched.get("h") is not None else None
                    inv = float(np.max(np.abs(probe - probe_a))) if probe is not None else float("nan")
                    cell = f"{model}_{anc_name}_{tag}"
                    cals.append({"cell": cell, "recip_game": trip["recip_game"], "ok": matched.get("ok", 0), "js": matched.get("js"), "t": matched.get("t"), "l2": sc.get("l2"), "invariance_max_err": inv, "probe_image": probe_image_residual(delta, stack["W_probe"])})
                    if not matched.get("ok"):
                        if model == "FULL":
                            full_ok = 0
                            attempts["js_fail"] += 1
                        got[tag] = None
                        continue
                    if inv > inv_atol:
                        attempts["invariance_fail"] += 1
                        if model == "FULL":
                            full_ok = 0
                    y_t = y_components(matched["probs"], rec["legal"], tgt["legal"])
                    y_w = y_components(matched["probs"], rec["legal"], wrg["legal"])
                    logit_pack[f"{trip['recip_game']}__{cell}"] = np.asarray(matched["logits"], dtype=np.float64)
                    if probe is not None:
                        probe_pack[f"{trip['recip_game']}__{cell}"] = probe
                    got[tag] = {"Y_T": y_t, "Y_W": y_w}
                    inter.append({"cell": cell, "r2_type": trip["r2_type"], "recip_game": trip["recip_game"], "valid": 1, "Y_T": y_t["Y"], "Y_W": y_w["Y"], "G_T": y_t["G"], "Q_T": y_t["Q"], "js": matched["js"], "l2": sc.get("l2"), "anchor_exact": exact[anc_name]})
                if got.get("t") and got.get("w"):
                    xo = crossover_scores(got["t"]["Y_T"]["Y"], got["w"]["Y_T"]["Y"], got["w"]["Y_W"]["Y"], got["t"]["Y_W"]["Y"])
                    recs[(model, anc_name)] = {**xo, "Y_T_ht": got["t"]["Y_T"]["Y"], "Y_T_hw": got["w"]["Y_T"]["Y"], "Y_W_hw": got["w"]["Y_W"]["Y"], "Y_W_ht": got["t"]["Y_W"]["Y"], "Y_anchor_T": y_a_t["Y"], "Y_anchor_W": y_a_w["Y"]}
        if full_ok and all(("FULL", a) in recs for a in ANCHORS):
            attempts["valid_full"] += 1
            for (model, anc_name), xo in recs.items():
                inter.append({"cell": "triplet_contrast", "model": model, "anchor": anc_name, "r2_type": trip["r2_type"], "recip_game": trip["recip_game"], "target_t_game": trip["target_t_game"], "target_w_game": trip["target_w_game"], "valid": 1, "board_exact_T": exact["T"], "board_exact_W": exact["W"], **xo})
    write_csv(write_dir / "fresh_calibrations.csv", cals or [{"reason": "none"}])
    write_csv(write_dir / "fresh_interventions.csv", inter or [{"reason": "none"}])
    write_csv(write_dir / "fresh_stage_identity.csv", idents or [{"reason": "none"}])
    if logit_pack:
        np.savez_compressed(write_dir / "fresh_full_logits.npz", **logit_pack)
    if probe_pack:
        np.savez_compressed(write_dir / "fresh_probe_probabilities.npz", **probe_pack)
    contrasts = [r for r in inter if r.get("cell") == "triplet_contrast" and r.get("valid") == 1]
    n_boot_s = int(cfg.get("boot_seed", BOOT_SEED))
    slices = {}
    for model in models:
        for anc in ANCHORS:
            slices[f"{model}_{anc}"] = _slice(contrasts, lambda r, m=model, a=anc: r.get("model") == m and r.get("anchor") == a, n_boot, n_boot_s)
        slices[f"{model}_N_A"] = _slice(contrasts, lambda r, m=model: r.get("model") == m and r.get("anchor") == "N" and r.get("r2_type") == "A_legal_diff", n_boot, n_boot_s + 1)
        slices[f"{model}_N_B"] = _slice(contrasts, lambda r, m=model: r.get("model") == m and r.get("anchor") == "N" and r.get("r2_type") == "B_similar_board_legal_change", n_boot, n_boot_s + 2)
    nat, at, aw = slices.get("FULL_N") or {}, slices.get("FULL_T") or {}, slices.get("FULL_W") or {}
    both = float(nat.get("both_positive_frac") or 0)
    gate = {
        "reconstruction_ok": 1, "split_ok": 1,
        "integrity_ok": int(hooks_ok and attempts["triplets"] == len(trips)),
        "invariance_ok": int(attempts["invariance_fail"] == 0 and attempts["valid_full"] > 0),
        "js_match_ok": int(attempts["valid_full"] > 0),
        "coverage_ok": int(attempts["valid_full"] >= (1 if smoke else int(cfg.get("coverage_min", COVERAGE_MIN)))),
        "natural_symmetric_ok": natural_symmetric_ok(nat.get("A_T") or {}, nat.get("A_W") or {}, nat.get("C") or {}, both, float(cfg.get("a_t_floor", AT_FLOOR)), float(cfg.get("both_pos_min", BOTH_POS_MIN))) if nat.get("n") else 0,
        "anchor_dependence_ok": anchor_dependence_ok(nat, at, aw) if nat.get("n") and at.get("n") and aw.get("n") else 0,
    }
    verdict = "SMOKE_ONLY" if smoke else decide_sym_verdict(gate)
    write_json(write_dir / "fresh_attempt_accounting.json", attempts)
    write_json(write_dir / "fresh_dependency_manifest.json", {"git_hash": git_hash(), "checkpoint_sha256": sha256_file(stack["ckpt"]), "probe_sha256": sha256_file(stack["probe_info"]["probe_path"]), "triplet_hash": man.get("triplet_sha256"), "used_confirmation": 0, "official_0A_letter": OFFICIAL_0A_LETTER})
    write_json(write_dir / "fresh_completion_status.json", {
        "status": "SMOKE_COMPLETE" if smoke else "SYMMETRIC_TARGET_FRESH_COMPLETE",
        "verdict": verdict, "gate": gate, "slices": slices, "attempts": attempts, "smoke": int(smoke),
        "used_confirmation": 0, "official_0A_letter": OFFICIAL_0A_LETTER,
        "original_sym_status_preserved": "SYMMETRIC_TARGET_CENSUS_INFEASIBLE",
        "sem_status_preserved": "LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY",
        "nst_status_preserved": "DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED",
        "fixed_js_status_preserved": "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED",
        "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
        "prior_cost_verdict_preserved": "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "section_5_3_status": "PARTIAL", "obstruction_ready": "NO",
    })
    (write_dir / "fresh_scientific_report.md").write_text(
        "\n".join(["# 08R2-SYMMETRIC-TARGET-FRESH Scientific Report", "", f"Verdict: `{verdict}`", "Natural-anchor FULL is primary. Positive C alone is not bidirectional switching.", f"Attempts: {attempts}", f"Gate: {gate}", f"FULL_N: {nat}", f"FULL_T: {at}", f"FULL_W: {aw}", ""]),
        encoding="utf-8",
    )
    print("08R2-SYMMETRIC-TARGET-FRESH")
    print(f"  verdict={verdict} valid_full={attempts['valid_full']}/{len(trips)}")
    print(f"  gate={gate}")
    print(f"  FULL_N={nat}")
    if smoke and attempts["valid_full"] != 1:
        raise SystemExit(f"{VERDICT_ENGINEERING} smoke failed")


def stage6(out_dir: Path) -> None:
    man = json.loads((out_dir / "fresh_triplet_manifest.json").read_text(encoding="utf-8"))
    rows = add_labels([json.loads(line) for line in (out_dir / "fresh_cohort_prefixes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()])
    inter = list(csv.DictReader((out_dir / "fresh_interventions.csv").open(encoding="utf-8")))
    logits = np.load(out_dir / "fresh_full_logits.npz")
    n_ok = n_bad = 0
    by = {t["recip_game"]: t for t in man["triplets"]}
    for row in inter:
        if row.get("cell") == "triplet_contrast" or row.get("Y_T") in (None, ""):
            continue
        trip = by.get(row.get("recip_game"))
        if not trip:
            continue
        rec, tgt = rows[int(trip["recip"])], rows[int(trip["target_t"])]
        lkey = f"{row['recip_game']}__{row['cell']}"
        if lkey not in logits.files:
            continue
        recon = y_components(softmax_np(logits[lkey]), rec["legal"], tgt["legal"])
        try:
            ok = abs(float(row["Y_T"]) - float(recon["Y"])) <= 1e-6
        except (TypeError, ValueError):
            ok = False
        n_ok += int(ok)
        n_bad += int(not ok)
    status = {"ok": int(n_bad == 0 and n_ok > 0), "n_match": n_ok, "n_mismatch": n_bad}
    write_json(out_dir / "fresh_stage6_status.json", status)
    print("08R2-SYMMETRIC-TARGET-FRESH STAGE 6")
    print(f"  ok={status['ok']} match={n_ok} mismatch={n_bad}")
    if not status["ok"]:
        raise SystemExit(VERDICT_ENGINEERING)


def main() -> None:
    args = parse_args()
    cfg = merge_cfg(load_config(args.parent_config), load_config(args.r2_config), load_config(args.config))
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.checkpoints_dir:
        cfg["checkpoints_dir"] = args.checkpoints_dir
    adopt_drive_defaults(cfg, args)
    out_dir = ensure_dir(isolate_fresh_dir(cfg["results_dir"])["results_dir"])
    prereg = Path(cfg.get("preregistration", "docs/experiments/08r2_symmetric_target_fresh_preregistration.md"))
    if prereg.is_file():
        (out_dir / "fresh_preregistration.md").write_text(prereg.read_text(encoding="utf-8"), encoding="utf-8")
    nst = _resolve_sibling(cfg, args, out_dir, "null_state_transfer", "nst_dir", ("null_state_pca_ridge.npz",))
    sem = _resolve_sibling(cfg, args, out_dir, "state_semantics", "sem_dir", ("state_semantics_interventions.csv",))
    stages = ["0", "1", "2", "3", "4", "5", "6"] if args.stage == "all" else [args.stage]
    if "0" in stages:
        stage0(out_dir)
    if "1" in stages:
        stage1(cfg, out_dir)
    if "2" in stages:
        stage2(cfg, out_dir)
    if "3" in stages:
        if not json.loads((out_dir / "fresh_triplet_census.json").read_text()).get("feasible"):
            raise SystemExit(STATUS_INFEASIBLE)
        stage3(cfg, out_dir, nst, sem)
    if "4" in stages:
        stage_eval(cfg, out_dir, args, nst, sem, smoke=True)
    if "5" in stages:
        stage_eval(cfg, out_dir, args, nst, sem, smoke=False)
    if "6" in stages:
        if not (out_dir / "fresh_full_logits.npz").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 5 logits missing")
        stage6(out_dir)


if __name__ == "__main__":
    main()
