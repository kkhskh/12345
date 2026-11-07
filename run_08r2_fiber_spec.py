"""08R2-FIBER-SPEC real-model target-specificity diagnostic.

Does not retrain the model or probe. Does not optimize a new editor.
Does not evaluate the sealed confirmation split.
Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from run_08r1_real_model_verification import git_hash, patch_and_audit, resolve_checkpoint
from src.othello_r1_eval import resolve_probe_path, sha256_file
from src.othello_r2 import (
    OFFICIAL_0A_LETTER,
    SITE,
    LinearBoardDecoder,
    decoder_state_effects,
    iterative_state_edit,
    load_holdout_game_ids,
    require_no_confirmation,
    torch_parity_report,
)
from src.othello_r2_fiber import (
    contrast_parity_report,
    independent_contrasts,
    js_divergence,
    reject_unexpected_preprocessing,
    softmax,
    svd_probe_geometry,
)
from src.othello_r2_fiber_spec import (
    BEHAVIOR_FLOOR,
    N_BOOT,
    N_RANDOM,
    N_WRONG,
    PAIR_SEED,
    RANDOM_SEEDS,
    REQUIRED_SPEC_FILES,
    STATUS_AWAITING,
    STATUS_COMPLETE,
    STATUS_INFEASIBLE,
    VERDICT_ENGINEERING,
    VERDICT_INCONCLUSIVE,
    WANT_A,
    WANT_B,
    WRONG_SEED,
    census_tables,
    decide_verdict,
    enumerate_candidates,
    evaluate_spec_completeness,
    fiber_excluded_games,
    isolate_spec_dir,
    match_l2,
    match_wrong_donors,
    pair_contrasts,
    select_disjoint_pairs,
    slice_boot,
    y_components,
)
from src.othello_runutil import add_labels, load_split, tokens_of, write_json
from src.toy_othello import (
    cache_site,
    load_config,
    load_othello_gpt,
    run_model_logits,
    softmax_np,
    verify_nanda_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-FIBER-SPEC L2-matched nullspace specificity.")
    p.add_argument("--config", default="configs/toy_othello_08r2_fiber_spec.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--census-only", action="store_true")
    return p.parse_args()


def merge_cfg(*parts: dict) -> dict:
    out = {}
    for p in parts:
        out.update(p)
    return out


def _require_real_stack():
    try:
        import torch
        import transformer_lens  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"real stack missing ({exc})")
    return torch


def _snapshot(model):
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def _unchanged(before, model):
    return all(np.allclose(a, b) for a, b in zip(before, _snapshot(model)))


def _patch(model, tokens, h):
    import torch

    vals = torch.tensor(np.asarray(h, dtype=np.float32).reshape(1, -1))
    logits, readback, audit = patch_and_audit(model, [tokens], SITE, vals, batch_size=1)
    return softmax_np(logits.numpy()[0]), logits.numpy()[0], audit


def _write_census(out_dir: Path, enum, selected, wrongs, cfg, extra=None):
    rows = []
    for p in selected["pairs"]:
        rows.append({k: p[k] for k in p if k != "census"})
    write_csv(out_dir / "fiber_spec_pair_census.csv", rows or [{"reason": "no_pairs", "n": 0}])
    write_json(
        out_dir / "fiber_spec_pair_manifest.json",
        {
            "n": len(selected["pairs"]),
            "n_a": selected["n_a"],
            "n_b": selected["n_b"],
            "feasible": selected["feasible"],
            "n_recipient_games": selected["n_recipient_games"],
            "n_target_games": selected["n_target_games"],
            "recip_target_overlap": selected["recip_target_overlap"],
            "pair_manifest_sha256": selected["pair_manifest_sha256"],
            "expected_sha256": cfg.get("expected_pair_manifest_sha256"),
            "hash_ok": int(selected["pair_manifest_sha256"] == cfg.get("expected_pair_manifest_sha256")),
            "seed": selected["seed"],
            "exclude_fiber_games": 1,
            "pairs": selected["pairs"],
            "wrong_donors": wrongs,
            "census": census_tables(enum, selected, wrongs),
        },
    )


def main() -> None:
    args = parse_args()
    parent = load_config(args.parent_config)
    r2cfg = load_config(args.r2_config)
    cfg = merge_cfg(parent, r2cfg, load_config(args.config))
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.checkpoints_dir:
        cfg["checkpoints_dir"] = args.checkpoints_dir
    out_dir = ensure_dir(isolate_spec_dir(cfg["results_dir"])["results_dir"])
    prereg_src = Path(cfg.get("preregistration", "docs/experiments/08r2_fiber_spec_preregistration.md"))
    if prereg_src.is_file():
        (out_dir / "fiber_spec_preregistration.md").write_text(prereg_src.read_text(encoding="utf-8"), encoding="utf-8")
        write_json(out_dir / "fiber_spec_preregistration.json", {k: cfg[k] for k in cfg if k in {
            "n_a", "n_b", "pair_seed", "wrong_seed", "random_seeds", "n_wrong", "n_random",
            "expected_pair_manifest_sha256", "behavior_floor", "l2_rtol", "primary_rank_tol",
            "probe_proba_atol_f32", "n_boot", "boot_seed", "editor_budget",
        }})

    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    if not holdout["hash_ok"]:
        raise SystemExit("FIBER_SPEC_ENGINEERING_FAILED confirmation manifest hash mismatch")
    data_dir = Path(cfg["data_dir"])
    val = add_labels(load_split(data_dir, "val"))
    require_no_confirmation(val, holdout["game_ids"])
    excl = fiber_excluded_games(val, holdout["game_ids"]) if cfg.get("exclude_fiber_games", True) else set()
    enum = enumerate_candidates(val, holdout["game_ids"], excl)
    selected = select_disjoint_pairs(enum["found"], int(cfg.get("n_a", WANT_A)), int(cfg.get("n_b", WANT_B)), int(cfg.get("pair_seed", PAIR_SEED)))
    if cfg.get("expected_pair_manifest_sha256") and selected["pair_manifest_sha256"] != cfg["expected_pair_manifest_sha256"]:
        raise SystemExit(f"FIBER_SPEC_ENGINEERING_FAILED pair hash {selected['pair_manifest_sha256']} != frozen {cfg['expected_pair_manifest_sha256']}")
    wrongs = [match_wrong_donors(enum["labeled"], p, holdout["game_ids"], int(cfg.get("n_wrong", N_WRONG)), int(cfg.get("wrong_seed", WRONG_SEED))) for p in selected["pairs"]]
    _write_census(out_dir, enum, selected, wrongs, cfg)
    census = census_tables(enum, selected, wrongs)
    print("08R2-FIBER-SPEC CENSUS")
    print(f"  A_cand={census['n_candidate_A']} B_cand={census['n_candidate_B']}")
    print(f"  selected_A={selected['n_a']} selected_B={selected['n_b']} feasible={selected['feasible']}")
    print(f"  recip={selected['n_recipient_games']} targets={selected['n_target_games']} overlap={selected['recip_target_overlap']}")
    print(f"  pair_hash={selected['pair_manifest_sha256']}")
    print(f"  wrong_ok={census['n_wrong_ok']}/{len(wrongs)}")
    if not selected["feasible"] or census["n_wrong_ok"] < len(selected["pairs"]):
        write_json(out_dir / "fiber_spec_completion_status.json", {
            "status": STATUS_INFEASIBLE,
            "verdict": VERDICT_INCONCLUSIVE,
            "census": census,
            "used_confirmation": 0,
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
        })
        raise SystemExit(STATUS_INFEASIBLE)

    if args.census_only:
        write_json(out_dir / "fiber_spec_completion_status.json", {
            "status": STATUS_AWAITING,
            "verdict": VERDICT_INCONCLUSIVE,
            "reason": "census_only",
            "census": census,
            "used_confirmation": 0,
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
        })
        print(STATUS_AWAITING)
        return

    try:
        _require_real_stack()
    except RuntimeError as exc:
        write_json(out_dir / "fiber_spec_completion_status.json", {
            "status": STATUS_AWAITING,
            "verdict": VERDICT_INCONCLUSIVE,
            "reason": str(exc),
            "census": census,
            "checkpoint_loaded": 0,
            "used_confirmation": 0,
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
        })
        print(STATUS_AWAITING)
        return

    probe_info = resolve_probe_path(args.probe_path, cfg["checkpoints_dir"])
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if not probe_info["exists"] or ckpt is None or not Path(ckpt).is_file():
        write_json(out_dir / "fiber_spec_completion_status.json", {
            "status": STATUS_AWAITING,
            "verdict": VERDICT_INCONCLUSIVE,
            "reason": "BLOCKED_MISSING_ARTIFACT",
            "census": census,
            "checkpoint_loaded": 0,
            "probe_loaded": 0,
            "used_confirmation": 0,
            "official_0A_letter": OFFICIAL_0A_LETTER,
        })
        print(STATUS_AWAITING)
        return

    with Path(probe_info["probe_path"]).open("rb") as handle:
        pack = pickle.load(handle)
    prep = reject_unexpected_preprocessing(pack)
    if not prep["ok"] or str(pack.get("hook")) != SITE:
        raise SystemExit(f"{VERDICT_ENGINEERING} probe identity {prep} {pack.get('hook')}")
    decoder = LinearBoardDecoder(pack)
    contrasts = independent_contrasts(decoder)
    dec_t = decoder.as_torch()
    model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=str(ckpt))
    ident = verify_nanda_checkpoint(model)
    if not ident.get("nanda_argmax_match"):
        raise SystemExit(f"{VERDICT_ENGINEERING} checkpoint {ident}")
    params0 = _snapshot(model)
    for p in model.parameters():
        p.requires_grad_(False)
    coef0 = decoder.coef.copy()

    H = cache_site(model, tokens_of(val), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    base_logits = run_model_logits(model, tokens_of(val), batch_size=int(cfg.get("batch_size", 64))).numpy()
    base_probs = np.stack([softmax_np(r) for r in base_logits])
    needed = sorted({int(p["recip"]) for p in selected["pairs"]} | {int(p["donor"]) for p in selected["pairs"]})
    for w in wrongs:
        needed.extend(int(d["index"]) for d in w["donors"])
    needed = sorted(set(needed))
    parity = contrast_parity_report(decoder, pack["models"], H[needed[:16]], contrasts, float(cfg.get("parity_proba_atol", 1e-6)))
    torch_p = torch_parity_report(decoder, pack["models"], H[needed[:16]])
    if parity["ok"] != 1:
        raise SystemExit(f"{VERDICT_ENGINEERING} contrast parity {parity}")
    svd = svd_probe_geometry(contrasts["W"], float(cfg.get("primary_rank_tol", 1e-8)), cfg.get("rank_tolerances", [1e-8]))
    if not svd.get("has_meaningful_nullspace"):
        raise SystemExit(f"{VERDICT_ENGINEERING} no nullspace")
    p_n = svd["P_N"]
    p_r = svd["P_R"]
    write_csv(out_dir / "fiber_spec_probe_parity.csv", [parity, {"backend": "torch", **{k: torch_p[k] for k in torch_p if k != "class_sets"}}])

    labeled = enum["labeled"]
    budget = float(cfg.get("editor_budget", 16.0))
    rows = []
    legal = {}
    logit_pack = {}
    probe_pack = {}
    null_pack = {}
    hooks_ok = 1
    attempts = {"pairs": 0, "editor_fail": 0, "invariance_fail": 0, "l2_fail": 0, "valid": 0}

    def record(pair, name, h, h_e, donor_game, extra=None):
        nonlocal hooks_ok
        rec = labeled[int(pair["recip"])]
        tgt = labeled[int(pair["donor"])]
        probs, logits, audit = _patch(model, rec["tokens"], h)
        hooks_ok = hooks_ok and int(audit.get("ok", 0))
        y = y_components(probs, rec["legal"], tgt["legal"])
        dec = decoder_state_effects(decoder, h, rec["rel_lab"], tgt["rel_lab"])
        probe = decoder.proba_np(h).reshape(64, 3)
        pe = decoder.proba_np(h_e).reshape(64, 3)
        inv = float(np.max(np.abs(probe - pe))) if name not in {"baseline", "editor_A", "full_donor", "correct_rowspace"} else float("nan")
        row = {
            "cell": name,
            "r2_type": pair["r2_type"],
            "recip": pair["recip"],
            "donor": pair["donor"],
            "recip_game": pair["recip_game"],
            "target_game": pair["target_game"],
            "donor_game": donor_game,
            "player": int(rec["player"]),
            "token_index": int(len(rec["tokens"]) - 1),
            "site": SITE,
            "Y": y["Y"],
            "G": y["G"],
            "Q": y["Q"],
            "stayed_legal_mass": y["stayed_legal_mass"],
            "stayed_illegal_mass": y["stayed_illegal_mass"],
            "board_exact": dec["board_exact"],
            "changed_acc": dec["changed_acc"],
            "unchanged_acc": dec["unchanged_acc"],
            "kl_to_base": kl_row_safe(base_probs[int(pair["recip"])], probs),
            "kl_to_editor": float("nan"),
            "l2_to_editor": float(np.linalg.norm(np.asarray(h) - np.asarray(h_e))),
            "invariance_max_err": inv,
            "hook_ok": int(audit.get("ok", 0)),
            "hamming": pair["hamming"],
        }
        if extra:
            row.update(extra)
        # fill kl_to_editor properly below after we have pe_editor
        rows.append(row)
        key = f"{pair['recip_game']}__{pair['target_game']}__{name}"
        logit_pack[key] = np.asarray(logits, dtype=np.float64)
        probe_pack[key] = probe
        return row, probs, probe, y

    for pair, wrep in zip(selected["pairs"], wrongs):
        attempts["pairs"] += 1
        rec = labeled[int(pair["recip"])]
        tgt = labeled[int(pair["donor"])]
        legal[f"{pair['recip_game']}|{pair['target_game']}"] = {
            "recip_legal": list(map(int, rec["legal"])),
            "target_legal": list(map(int, tgt["legal"])),
            "became_legal": sorted(set(tgt["legal"]) - set(rec["legal"])),
            "became_illegal": sorted(set(rec["legal"]) - set(tgt["legal"])),
            "recip_board": list(map(int, np.asarray(rec["board"]).reshape(-1))),
            "target_board": list(map(int, np.asarray(tgt["board"]).reshape(-1))),
        }
        h_rec = np.asarray(H[int(pair["recip"])], dtype=np.float64)
        h_d = np.asarray(H[int(pair["donor"])], dtype=np.float64)
        ed = iterative_state_edit(
            h_rec, tgt["rel_lab"], rec["rel_lab"], dec_t,
            steps=int(cfg.get("edit_steps", 40)), lr=float(cfg.get("edit_lr", 0.4)),
            budget=budget, lambda_norm=float(cfg["lambda_norm"]),
            eta_preserve=float(cfg["eta_preserve"]), mu_manifold=float(cfg.get("mu_manifold", 0.0)),
        )
        if not ed.get("valid_edit") or ed.get("edited") is None:
            attempts["editor_fail"] += 1
            rows.append({"cell": "editor_A", "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 0, "reason": ed.get("reason")})
            continue
        h_e = np.asarray(ed["edited"], dtype=np.float64)
        rec_e, probs_e, probe_e, _ = record(pair, "editor_A", h_e, h_e, pair["target_game"], extra={"valid": 1, "editor_l2": float(np.linalg.norm(h_e - h_rec)), "board_exact_editor": decoder_state_effects(decoder, h_e, rec["rel_lab"], tgt["rel_lab"])["board_exact"]})
        record(pair, "baseline", h_rec, h_e, pair["recip_game"], extra={"valid": 1})
        record(pair, "full_donor", h_d, h_e, pair["target_game"], extra={"valid": 1})
        delta = h_d - h_e
        d_n = p_n @ delta
        d_r = p_r @ delta
        target_l2 = float(np.linalg.norm(d_n))
        null_pack[f"{pair['recip_game']}__{pair['target_game']}__correct"] = d_n
        h_c = h_e + d_n
        rec_c, probs_c, probe_c, y_c = record(
            pair, "correct_null", h_c, h_e, pair["target_game"],
            extra={"valid": 1, "l2_target": target_l2, "proj_resid": float(np.linalg.norm(contrasts["W"] @ d_n))},
        )
        if cfg.get("include_rowspace_secondary", True):
            record(pair, "correct_rowspace", h_e + d_r, h_e, pair["target_game"], extra={"valid": 1, "secondary": 1})
        y_w = []
        g_w = []
        q_w = []
        l2_ok = 1
        inv_ok = 1
        for j, wd in enumerate(wrep["donors"]):
            h_wj = np.asarray(H[int(wd["index"])], dtype=np.float64)
            raw = p_n @ (h_wj - h_e)
            matched = match_l2(raw, target_l2)
            if not matched["ok"]:
                l2_ok = 0
                attempts["l2_fail"] += 1
                rows.append({"cell": f"wrong_null_{j}", "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 0, "reason": matched["reason"]})
                continue
            null_pack[f"{pair['recip_game']}__{pair['target_game']}__wrong_{j}"] = matched["vec"]
            rec_w, probs_w, probe_w, y_wj = record(
                pair, f"wrong_null_{j}", h_e + matched["vec"], h_e, wd["game_id"],
                extra={"valid": 1, "l2_rel_err": matched["rel_err"], **{k: wd[k] for k in wd if k != "index"}},
            )
            if rec_w["invariance_max_err"] > float(cfg.get("probe_proba_atol_f32", 1e-5)):
                inv_ok = 0
            y_w.append(y_wj["Y"])
            g_w.append(y_wj["G"])
            q_w.append(y_wj["Q"])
        y_r = []
        for j, seed in enumerate(cfg.get("random_seeds", RANDOM_SEEDS)):
            rng = np.random.default_rng(int(seed) + int(pair["recip"]))
            raw = p_n @ rng.normal(size=p_n.shape[0])
            matched = match_l2(raw, target_l2)
            if not matched["ok"]:
                l2_ok = 0
                attempts["l2_fail"] += 1
                continue
            null_pack[f"{pair['recip_game']}__{pair['target_game']}__random_{j}"] = matched["vec"]
            rec_r, _, _, y_rj = record(
                pair, f"random_null_{j}", h_e + matched["vec"], h_e, "random",
                extra={"valid": 1, "l2_rel_err": matched["rel_err"], "seed": int(seed)},
            )
            if rec_r["invariance_max_err"] > float(cfg.get("probe_proba_atol_f32", 1e-5)):
                inv_ok = 0
            y_r.append(y_rj["Y"])
        if rec_c["invariance_max_err"] > float(cfg.get("probe_proba_atol_f32", 1e-5)):
            inv_ok = 0
            attempts["invariance_fail"] += 1
        if inv_ok and l2_ok and len(y_w) == int(cfg.get("n_wrong", N_WRONG)) and len(y_r) == int(cfg.get("n_random", N_RANDOM)):
            attempts["valid"] += 1
            ctr = pair_contrasts(y_c["Y"], y_w, y_r)
            rec_c.update(ctr)
            rec_c["dG_wrong"] = float(y_c["G"] - np.mean(g_w))
            rec_c["dQ_wrong"] = float(y_c["Q"] - np.mean(q_w))
            rec_c["pair_valid"] = 1
            # store contrasts on a dedicated summary row
            rows.append({
                "cell": "pair_contrast",
                "r2_type": pair["r2_type"],
                "recip_game": pair["recip_game"],
                "target_game": pair["target_game"],
                "valid": 1,
                **ctr,
                "dG_wrong": rec_c["dG_wrong"],
                "dQ_wrong": rec_c["dQ_wrong"],
                "js_correct_vs_editor": js_divergence(probs_e, probs_c),
                "kl_correct_vs_base": rec_c["kl_to_base"],
            })

        # repair kl_to_editor for rows of this pair
        for row in rows:
            if row.get("recip_game") == pair["recip_game"] and row.get("target_game") == pair["target_game"] and "kl_to_editor" in row:
                key = f"{pair['recip_game']}__{pair['target_game']}__{row['cell']}"
                if key in logit_pack and f"{pair['recip_game']}__{pair['target_game']}__editor_A" in logit_pack:
                    row["kl_to_editor"] = kl_row_safe(softmax(logit_pack[f"{pair['recip_game']}__{pair['target_game']}__editor_A"]), softmax(logit_pack[key]))

    if not _unchanged(params0, model) or not np.allclose(coef0, decoder.coef):
        raise SystemExit(f"{VERDICT_ENGINEERING} parameters changed")

    contrasts_rows = [r for r in rows if r.get("cell") == "pair_contrast" and r.get("valid") == 1]
    boot_rows = []
    slices = {}
    for name, mask in (
        ("pooled", None),
        ("A", "A_legal_diff"),
        ("B", "B_similar_board_legal_change"),
    ):
        g = contrasts_rows if mask is None else [r for r in contrasts_rows if r.get("r2_type") == mask]
        if not g:
            slices[name] = {"n": 0}
            continue
        sl = {
            "n": len(g),
            "n_recipient_games": len({r["recip_game"] for r in g}),
            "n_target_games": len({r["target_game"] for r in g}),
            "mean_delta_specific": float(np.nanmean([r["delta_specific"] for r in g])),
            "median_delta_specific": float(np.nanmedian([r["delta_specific"] for r in g])),
            "mean_delta_random": float(np.nanmean([r["delta_random"] for r in g])),
            "mean_dG": float(np.nanmean([r["dG_wrong"] for r in g])),
            "mean_dQ": float(np.nanmean([r["dQ_wrong"] for r in g])),
            "frac_correct_wins_wrong": float(np.mean([r["delta_specific"] > 0 for r in g])),
            "frac_correct_wins_random": float(np.mean([r["delta_random"] > 0 for r in g])),
        }
        sl["delta_specific_ci"] = slice_boot([r["delta_specific"] for r in g], [r["recip_game"] for r in g], int(cfg.get("n_boot", N_BOOT)), int(cfg.get("boot_seed", 82)))
        sl["delta_random_ci"] = slice_boot([r["delta_random"] for r in g], [r["recip_game"] for r in g], int(cfg.get("n_boot", N_BOOT)), int(cfg.get("boot_seed", 82)) + 1)
        slices[name] = sl
        boot_rows.append({"slice": name, "metric": "delta_specific", **sl["delta_specific_ci"]})
        boot_rows.append({"slice": name, "metric": "delta_random", **sl["delta_random_ci"]})

    pooled = slices.get("pooled", {})
    ci = pooled.get("delta_specific_ci") or {}
    rci = pooled.get("delta_random_ci") or {}
    gate = {
        "invariance_ok": int(attempts["invariance_fail"] == 0 and attempts["valid"] > 0),
        "l2_ok": int(attempts["l2_fail"] == 0 and attempts["valid"] > 0),
        "mean_delta_specific_positive": int(pooled.get("mean_delta_specific", 0) > 0),
        "ci_lower_gt_0": int(ci.get("ci95_low", 0) > 0) if ci else 0,
        "mean_delta_ge_floor": int(pooled.get("mean_delta_specific", 0) >= float(cfg.get("behavior_floor", BEHAVIOR_FLOOR))),
        "both_components_correct_sign": int(pooled.get("mean_dG", 0) > 0 and pooled.get("mean_dQ", 0) < 0),
        "beats_random": int(pooled.get("mean_delta_random", 0) > 0 and (rci.get("ci95_low", 0) > 0 if rci else 0)),
        "integrity_ok": int(hooks_ok and parity["ok"] == 1 and attempts["valid"] == len(selected["pairs"])),
    }
    verdict = decide_verdict(gate)
    write_csv(out_dir / "fiber_spec_interventions.csv", rows)
    write_csv(out_dir / "fiber_spec_bootstrap.csv", boot_rows or [{"reason": "no_valid_contrasts", "n": 0}])
    write_json(out_dir / "fiber_spec_target_legal_sets.json", legal)
    write_json(out_dir / "fiber_spec_attempt_accounting.json", attempts)
    np.savez_compressed(out_dir / "fiber_spec_full_logits.npz", **logit_pack)
    np.savez_compressed(out_dir / "fiber_spec_full_probe_probabilities.npz", **probe_pack)
    np.savez_compressed(out_dir / "fiber_spec_nullspace_vectors.npz", **null_pack)
    write_json(
        out_dir / "fiber_spec_dependency_manifest.json",
        {
            "git_hash": git_hash(),
            "site": SITE,
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "probe_path": probe_info["probe_path"],
            "probe_sha256": sha256_file(probe_info["probe_path"]),
            "pair_manifest_sha256": selected["pair_manifest_sha256"],
            "nanda": ident,
            "svd_rank": svd["numerical_rank"],
            "nullspace_dim": svd["nullspace_dim"],
            "used_confirmation": 0,
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
            "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        },
    )
    write_report(out_dir / "fiber_spec_scientific_report.md", gate, verdict, slices, attempts, selected, svd)
    write_json(out_dir / "fiber_spec_completion_status.json", {
        "status": STATUS_COMPLETE,
        "verdict": verdict,
        "gate": gate,
        "slices": slices,
        "attempts": attempts,
        "checkpoint_loaded": 1,
        "probe_loaded": 1,
        "parity_ok": int(parity["ok"] == 1),
        "pairs_ok": 1,
        "controls_complete": int(attempts["valid"] == len(selected["pairs"])),
        "full_outputs_saved": int(len(logit_pack) > 0),
        "used_confirmation": 0,
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "section_5_3_status": "PARTIAL",
        "obstruction_ready": "NO",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "pair_counts": dict(Counter(p["r2_type"] for p in selected["pairs"])),
    })
    gate_files = evaluate_spec_completeness(out_dir, real_model=True)
    if not gate_files["ok"]:
        raise SystemExit(f"{VERDICT_ENGINEERING} completeness {gate_files['failures']}")
    print("08R2-FIBER-SPEC")
    print(f"  verdict={verdict}")
    print(f"  valid_pairs={attempts['valid']}/{len(selected['pairs'])}")
    print(f"  pooled={pooled}")
    print(f"  gate={gate}")
    print("  official letter unchanged: C / PARTIAL")
    print("  08R2_VALIDATION_COMPLETE — NEGATIVE")
    print("  SECTION_5_3 = PARTIAL")
    print("  OBSTRUCTION_READY = NO")


def kl_row_safe(p, q) -> float:
    from src.othello_r1 import kl_row

    if p is None or q is None or np.asarray(p).size == 0:
        return float("nan")
    return kl_row(p, q)


def write_report(path: Path, gate, verdict, slices, attempts, selected, svd) -> None:
    lines = [
        "# 08R2-FIBER-SPEC Scientific Report",
        "",
        "L2-matched correct-versus-wrong-donor probe-null interventions on a new disjoint validation pair set.",
        "",
        "- `08R2_VALIDATION_COMPLETE — NEGATIVE`",
        "- Original 0A: **C / PARTIAL**",
        "- `SECTION_5_3 = PARTIAL`",
        "- `OBSTRUCTION_READY = NO`",
        "",
        f"Verdict: `{verdict}`",
        f"Selected pairs: A={selected['n_a']} B={selected['n_b']} hash={selected['pair_manifest_sha256']}",
        f"Attempts: {attempts}",
        f"Gate: {gate}",
        f"SVD rank={svd.get('numerical_rank')} null={svd.get('nullspace_dim')}",
        f"Pooled: {slices.get('pooled')}",
        f"A: {slices.get('A')}",
        f"B: {slices.get('B')}",
        "",
        "A supported gate means the correct-donor null component beat L2-matched wrong donors and random null directions on the intended legal set.",
        "It does not identify a unique game-state coordinate and does not validate obstruction theory.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
