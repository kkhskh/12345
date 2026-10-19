"""08R2-NULL-STATE-TRANSFER staged donor-free null-state experiment.

Does not overwrite SPEC/COST/fixed-JS. No confirmation. Official letter C / PARTIAL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
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
)
from src.othello_r2_fiber import (
    independent_contrasts,
    js_divergence,
    reject_unexpected_preprocessing,
    softmax,
    svd_probe_geometry,
)
from src.othello_r2_fiber_cost import adopt_drive_defaults
from src.othello_r2_fiber_cost_fixed import SCALE_GRID, calibrate_fixed_js, scaled_displacement
from src.othello_r2_fiber_spec import enumerate_candidates, match_wrong_donors, select_disjoint_pairs, slice_boot, y_components
from src.othello_r2_null_state import (
    ALPHAS,
    BEHAVIOR_FLOOR,
    COVERAGE_MIN,
    CV_SEED,
    FEATURE_NAMES,
    FIXED_JS,
    FIXED_JS_TOL,
    KS,
    N_FOLDS,
    PAIR_SEED,
    PRIMARY_CELLS,
    RANDOM_SEED,
    SECONDARY_CELLS,
    STATUS_AWAITING,
    STATUS_INFEASIBLE,
    STATUS_STAGE0_FAIL,
    VERDICT_ENGINEERING,
    WANT_A,
    WANT_B,
    WRONG_SEED,
    Y_RECON_ATOL,
    audit_saved_y,
    decide_nst_verdict,
    donor_separation_audit,
    isolate_nst_dir,
    learned_null_delta,
    legal_record,
    mean_state_delta,
    oracle_null_delta,
    pack_model,
    parse_stage_mean_name,
    prior_eval_games,
    r2_score,
    rescore_y_from_logits,
    select_ridge_pca,
    split_leakage_audit,
    stage_key,
    stage_mean_name,
    symbolic_state,
    verify_simulator_legal,
)
from src.othello_runutil import add_labels, load_split, tokens_of, write_json
from src.toy_othello import cache_site, load_config, load_othello_gpt, softmax_np, square_to_token, verify_nanda_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-NULL-STATE-TRANSFER")
    p.add_argument("--config", default="configs/toy_othello_08r2_null_state_transfer.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--stage", default="0", choices=["0", "1", "2", "3", "4", "all"])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--fixed-js-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def merge_cfg(*parts):
    out = {}
    for p in parts:
        out.update(p)
    return out


def _by_game(rows):
    return {str(r["game_id"]): r for r in rows}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_fixed_js_dir(cfg, args, out_dir: Path) -> Path | None:
    candidates = []
    if args.fixed_js_dir:
        candidates.append(Path(args.fixed_js_dir))
    candidates.append(out_dir.parent / "fiber_cost_fixed_js")
    candidates.append(Path(cfg["results_dir"]) / "results" / "08r2" / "fiber_cost_fixed_js")
    if Path(cfg["results_dir"]).name == "08r2":
        candidates.append(Path(cfg["results_dir"]) / "fiber_cost_fixed_js")
    for path in candidates:
        if path.is_dir() and (path / "fixed_js_interventions.csv").is_file() and (path / "fixed_js_full_logits.npz").is_file():
            return path
    return candidates[0] if args.fixed_js_dir else candidates[1]


def _became(rec, tgt):
    became_l = sorted(set(tgt["legal"]) - set(rec["legal"]))
    became_i = sorted(set(rec["legal"]) - set(tgt["legal"]))
    return {
        "became_legal_squares": became_l,
        "became_illegal_squares": became_i,
        "became_legal_tokens": [square_to_token(s) for s in became_l if square_to_token(s) is not None],
        "became_illegal_tokens": [square_to_token(s) for s in became_i if square_to_token(s) is not None],
    }


def stage0(cfg, out_dir: Path, args) -> dict:
    fixed = _resolve_fixed_js_dir(cfg, args, out_dir)
    if fixed is None or not Path(fixed).is_dir() or not (Path(fixed) / "fixed_js_interventions.csv").is_file():
        write_json(out_dir / "null_state_stage0_status.json", {"ok": 0, "reason": f"missing_fixed_js_dir:{fixed}", "secondary_js_runs_present": 0})
        raise SystemExit(f"{VERDICT_ENGINEERING} Stage 0 missing fixed-JS archive at {fixed}")
    fixed = Path(fixed)
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    require_no_confirmation(val, holdout["game_ids"])
    by = _by_game(val)
    inter = list(csv.DictReader((fixed / "fixed_js_interventions.csv").open(encoding="utf-8")))
    cals = list(csv.DictReader((fixed / "fixed_js_calibrations.csv").open(encoding="utf-8"))) if (fixed / "fixed_js_calibrations.csv").is_file() else []
    logits = np.load(fixed / "fixed_js_full_logits.npz")
    js_vals = []
    for r in cals + inter:
        v = r.get("target_js") or r.get("js_target")
        if v not in (None, ""):
            js_vals.append(float(v))
    secondary_present = int(any(abs(v - 0.005) < 1e-9 or abs(v - 0.02) < 1e-9 for v in js_vals))
    legal_doc = {}
    score_rows = []
    n_ok = 0
    n_bad = 0
    n_legal_bad = 0
    n_missing_logits = 0
    for row in inter:
        rg, tg, cell = row.get("recip_game"), row.get("target_game"), row.get("cell")
        if cell in {"pair_contrast"} or rg not in by or tg not in by:
            continue
        rec, tgt = by[rg], by[tg]
        vrec = verify_simulator_legal(rec)
        vtgt = verify_simulator_legal(tgt)
        if not vrec["ok"] or not vtgt["ok"]:
            n_legal_bad += 1
        key = f"{rg}|{tg}"
        if key not in legal_doc:
            became = _became(rec, tgt)
            legal_doc[key] = {
                "recipient": legal_record(rec),
                "target": legal_record(tgt),
                **became,
                "sim_ok": int(vrec["ok"] and vtgt["ok"]),
            }
        lkey = f"{rg}__{tg}__{cell}"
        if lkey not in logits.files:
            n_missing_logits += 1
            continue
        recon = rescore_y_from_logits(logits[lkey], rec["legal"], tgt["legal"])
        chk = audit_saved_y(row, recon, float(cfg.get("y_recon_atol", Y_RECON_ATOL)))
        score_rows.append({
            "cell": cell, "recip_game": rg, "target_game": tg, "ok": chk["ok"],
            "saved_Y": row.get("Y"), "recomputed_Y": recon["Y"],
            "saved_G": row.get("G"), "recomputed_G": recon["G"],
            "saved_Q": row.get("Q"), "recomputed_Q": recon["Q"],
            "mismatches": ",".join(chk["mismatches"]),
        })
        n_ok += int(chk["ok"] == 1)
        n_bad += int(chk["ok"] == 0)
    status = {
        "ok": int(n_bad == 0 and n_legal_bad == 0 and n_ok > 0),
        "n_rescored": n_ok + n_bad,
        "n_match": n_ok,
        "n_mismatch": n_bad,
        "n_legal_mismatch": n_legal_bad,
        "n_missing_logits": n_missing_logits,
        "secondary_js_runs_present": secondary_present,
        "secondary_js_runs_note": "JS=0.005 and JS=0.02 were not in the uploaded primary archive" if not secondary_present else "secondary rows found",
        "fixed_js_dir": str(fixed),
        "y_recon_atol": float(cfg.get("y_recon_atol", Y_RECON_ATOL)),
    }
    write_csv(out_dir / "null_state_stage0_y_rescore.csv", score_rows or [{"reason": "no_rows"}])
    write_json(out_dir / "null_state_stage0_legal_sets.json", legal_doc)
    write_json(out_dir / "null_state_stage0_status.json", status)
    print("08R2-NULL-STATE STAGE 0")
    print(f"  ok={status['ok']} rescored={status['n_rescored']} mismatch={n_bad} legal_bad={n_legal_bad} missing_logits={n_missing_logits}")
    print(f"  secondary_js_runs_present={secondary_present} ({status['secondary_js_runs_note']})")
    if not status["ok"]:
        write_json(out_dir / "null_state_completion_status.json", {"status": STATUS_STAGE0_FAIL, "stage0": status, "used_confirmation": 0})
        raise SystemExit(STATUS_STAGE0_FAIL)
    return status


def stage1(cfg, out_dir: Path) -> dict:
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    require_no_confirmation(val, holdout["game_ids"])
    prior = prior_eval_games(val, holdout["game_ids"])
    enum = enumerate_candidates(val, holdout["game_ids"], prior["games"])
    selected = select_disjoint_pairs(enum["found"], int(cfg.get("n_a", WANT_A)), int(cfg.get("n_b", WANT_B)), int(cfg.get("pair_seed", PAIR_SEED)))
    used = {p["recip_game"] for p in selected["pairs"]} | {p["target_game"] for p in selected["pairs"]}
    blocked = holdout["game_ids"] | prior["games"] | used
    wrongs = [match_wrong_donors(enum["labeled"], p, blocked, 1, int(cfg.get("wrong_seed", WRONG_SEED))) for p in selected["pairs"]]
    recip = [p["recip_game"] for p in selected["pairs"]]
    target = [p["target_game"] for p in selected["pairs"]]
    wrong_games = [d["game_id"] for w in wrongs for d in (w.get("donors") or [])]
    leak = split_leakage_audit([], [], recip, target, wrong_games, prior["games"], holdout["game_ids"])
    expected_hash = cfg.get("expected_pair_manifest_sha256")
    hash_ok = int(not expected_hash or selected["pair_manifest_sha256"] == expected_hash)
    if expected_hash and not hash_ok:
        write_json(out_dir / "null_state_completion_status.json", {
            "status": STATUS_INFEASIBLE,
            "reason": "pair_hash_mismatch",
            "expected": expected_hash,
            "got": selected["pair_manifest_sha256"],
            "used_confirmation": 0,
        })
        raise SystemExit(f"{STATUS_INFEASIBLE} pair hash {selected['pair_manifest_sha256']} != frozen {expected_hash}")
    census = {
        "n_prior_excluded_games": prior["n_total"],
        "n_candidate_A": len(enum["found"]["A_legal_diff"]),
        "n_candidate_B": len(enum["found"]["B_similar_board_legal_change"]),
        "selected_A": selected["n_a"],
        "selected_B": selected["n_b"],
        "n_recipient_games": selected["n_recipient_games"],
        "n_target_games": selected["n_target_games"],
        "recip_target_overlap": selected["recip_target_overlap"],
        "feasible": int(bool(selected.get("feasible")) and selected["n_a"] == int(cfg.get("n_a", WANT_A)) and selected["n_b"] == int(cfg.get("n_b", WANT_B)) and all(w.get("ok") for w in wrongs) and leak["ok"]),
        "pair_manifest_sha256": selected["pair_manifest_sha256"],
        "expected_pair_manifest_sha256": expected_hash,
        "hash_ok": hash_ok,
        "seed": selected["seed"],
        "prior_spec_hash": prior["spec_hash"],
        "split_leakage_ok": leak["ok"],
    }
    write_json(out_dir / "null_state_census.json", {**census, "prior": {"n_fiber": prior["n_fiber"], "n_spec_pairs": prior["n_spec_pairs"], "n_total": prior["n_total"]}})
    write_json(out_dir / "null_state_split_games.json", {
        "excluded_prior": sorted(prior["games"]),
        "n_excluded": prior["n_total"],
        "recipients": recip,
        "targets": target,
        "wrong_targets": wrong_games,
    })
    write_json(out_dir / "null_state_split_leakage.json", leak)
    write_json(
        out_dir / "null_state_pair_manifest.json",
        {
            "n": len(selected["pairs"]),
            "n_a": selected["n_a"],
            "n_b": selected["n_b"],
            "feasible": selected.get("feasible"),
            "pair_manifest_sha256": selected["pair_manifest_sha256"],
            "pairs": selected["pairs"],
            "wrong_targets": wrongs,
        },
    )
    print("08R2-NULL-STATE STAGE 1 CENSUS")
    print(f"  excluded={prior['n_total']} A_cand={census['n_candidate_A']} B_cand={census['n_candidate_B']}")
    print(f"  selected_A={selected['n_a']} selected_B={selected['n_b']} feasible={census['feasible']}")
    print(f"  pair_hash={selected['pair_manifest_sha256']}")
    if not census["feasible"]:
        write_json(out_dir / "null_state_completion_status.json", {"status": STATUS_INFEASIBLE, "census": census, "used_confirmation": 0})
        raise SystemExit(STATUS_INFEASIBLE)
    return {"selected": selected, "wrongs": wrongs, "census": census, "prior": prior, "val": val}


def _load_model_stack(cfg, args):
    probe_info = resolve_probe_path(args.probe_path, cfg["checkpoints_dir"])
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if not probe_info["exists"] or ckpt is None or not Path(ckpt).is_file():
        return None
    with Path(probe_info["probe_path"]).open("rb") as handle:
        pack = pickle.load(handle)
    prep = reject_unexpected_preprocessing(pack)
    if not prep["ok"] or str(pack.get("hook")) != SITE:
        raise SystemExit(f"{VERDICT_ENGINEERING} probe {prep}")
    decoder = LinearBoardDecoder(pack)
    contrasts = independent_contrasts(decoder)
    model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=str(ckpt))
    ident = verify_nanda_checkpoint(model)
    if not ident.get("nanda_argmax_match"):
        raise SystemExit(f"{VERDICT_ENGINEERING} checkpoint {ident}")
    for p in model.parameters():
        p.requires_grad_(False)
    svd = svd_probe_geometry(contrasts["W"], float(cfg.get("primary_rank_tol", 1e-8)), [1e-8])
    return {"probe_info": probe_info, "ckpt": ckpt, "pack": pack, "decoder": decoder, "contrasts": contrasts, "model": model, "ident": ident, "svd": svd, "P_N": svd["P_N"]}


def _natural_rows(rows):
    return [r for r in rows if (not r.get("terminal")) and r.get("legal")]


def stage2(cfg, out_dir: Path, args) -> dict:
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    train = add_labels(load_split(Path(cfg["data_dir"]), "train"))
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    require_no_confirmation(train, holdout["game_ids"])
    require_no_confirmation(val, holdout["game_ids"])
    stack = _load_model_stack(cfg, args)
    if stack is None:
        write_json(out_dir / "null_state_completion_status.json", {"status": STATUS_AWAITING, "reason": "BLOCKED_MISSING_ARTIFACT", "stage": 2})
        print(STATUS_AWAITING)
        return {}
    man = json.loads((out_dir / "null_state_pair_manifest.json").read_text(encoding="utf-8"))
    prior = json.loads((out_dir / "null_state_split_games.json").read_text(encoding="utf-8"))
    eval_games = set(prior.get("recipients") or []) | set(prior.get("targets") or []) | set(prior.get("wrong_targets") or [])
    excluded_prior = set(prior.get("excluded_prior") or [])
    train_nat = _natural_rows(train)
    H = cache_site(stack["model"], tokens_of(train_nat), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    P_N = stack["P_N"]
    Z, B, games = [], [], []
    stage_acc = {}
    for i, row in enumerate(train_nat):
        z = P_N @ np.asarray(H[i], dtype=np.float64)
        b = symbolic_state(row)
        Z.append(z)
        B.append(b)
        games.append(str(row["game_id"]))
        stage_acc.setdefault(stage_key(row), []).append(z)
    Z = np.asarray(Z)
    B = np.asarray(B)
    stage_means = {k: np.mean(v, axis=0) for k, v in stage_acc.items()}
    sel = select_ridge_pca(Z, B, games, tuple(cfg.get("ks", KS)), tuple(cfg.get("alphas", ALPHAS)), int(cfg.get("n_folds", N_FOLDS)), int(cfg.get("cv_seed", CV_SEED)))
    model = pack_model(sel["pca"], sel["ridge"], P_N, stage_means)
    unused_val = [r for r in _natural_rows(val) if str(r["game_id"]) not in eval_games and str(r["game_id"]) not in excluded_prior and str(r["game_id"]) not in holdout["game_ids"]]
    held = {"n_rows": 0, "n_games": 0, "r2": float("nan")}
    if unused_val:
        Hv = cache_site(stack["model"], tokens_of(unused_val), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
        Zv = np.asarray([P_N @ np.asarray(Hv[i], dtype=np.float64) for i in range(len(unused_val))])
        Bv = np.asarray([symbolic_state(r) for r in unused_val])
        pred = sel["pca"]["mu"] + (Bv @ model["W"] + model["c"]) @ model["V"].T
        pred = (P_N @ pred.T).T
        held = {"n_rows": int(Zv.shape[0]), "n_games": len({str(r["game_id"]) for r in unused_val}), "r2": r2_score(Zv, pred)}
    np.savez_compressed(out_dir / "null_state_pca_ridge.npz", mu=model["mu"], V=model["V"], W=model["W"], c=model["c"], P_N=P_N, k=model["k"], alpha=model["alpha"])
    write_csv(out_dir / "null_state_cv_table.csv", sel["table"])
    write_json(out_dir / "null_state_feature_names.json", {"names": FEATURE_NAMES})
    write_json(out_dir / "null_state_train_game_ids.json", {"games": sorted(set(games)), "n_games": len(set(games)), "n_rows": int(Z.shape[0])})
    write_json(
        out_dir / "null_state_train_diagnostics.json",
        {
            "n_rows": int(Z.shape[0]),
            "n_games": len(set(games)),
            "best": sel["best"],
            "train_r2": sel["train_r2"],
            "heldout_unused_val_residual": held,
            "explained_variance_ratio": sel["pca"]["explained_variance_ratio"],
            "reconstructed_rank": int(np.linalg.matrix_rank(model["V"])),
            "nullspace_dim": int(stack["svd"]["nullspace_dim"]),
            "probe_rank": int(stack["svd"].get("numerical_rank", -1)),
            "uses_donor_activation": 0,
            "site": SITE,
        },
    )
    write_json(out_dir / "null_state_heldout_residual.json", held)
    write_json(out_dir / "null_state_stage_means.json", {stage_mean_name(k): v.tolist() for k, v in stage_means.items()})
    write_json(out_dir / "null_state_donor_separation.json", donor_separation_audit())
    print("08R2-NULL-STATE STAGE 2")
    print(f"  n_rows={Z.shape[0]} n_games={len(set(games))} best={sel['best']} train_r2={sel['train_r2']:.4f}")
    print(f"  unused_val_r2={held['r2']} n_unused={held['n_rows']}")
    return {"model": model, "stack": stack}


def _y_grad(model, tokens, h_e, became_legal_toks, became_illegal_toks):
    import torch

    device = next(model.parameters()).device
    h = torch.tensor(np.asarray(h_e, dtype=np.float32), device=device, requires_grad=True)
    t_idx = len(tokens) - 1

    def hook_fn(act, hook):
        x = act.clone()
        x[0, t_idx] = h
        return x

    ids = torch.tensor([tokens], device=device)
    logits = model.run_with_hooks(ids, fwd_hooks=[(SITE, hook_fn)])
    last = logits[0, t_idx].float()
    probs = torch.softmax(last, dim=-1)
    g = probs[became_legal_toks].sum() if became_legal_toks else last.new_zeros(())
    q = probs[became_illegal_toks].sum() if became_illegal_toks else last.new_zeros(())
    (g - q).backward()
    if h.grad is None:
        return None
    return h.grad.detach().cpu().numpy().astype(np.float64)


def _patch(model, tokens, h):
    vals = __import__("torch").tensor(np.asarray(h, dtype=np.float32).reshape(1, -1))
    logits, readback, audit = patch_and_audit(model, [tokens], SITE, vals, batch_size=1)
    return softmax_np(logits.numpy()[0]), logits.numpy()[0], audit


def _slice_stats(rows, boot_seed: int, n_boot: int) -> dict:
    if not rows:
        return {"n": 0}
    sl = {
        "n": len(rows),
        "mean_delta_wrong": float(np.nanmean([r["delta_learned_wrong"] for r in rows])),
        "mean_delta_random": float(np.nanmean([r["delta_learned_random"] for r in rows])),
        "mean_delta_mean": float(np.nanmean([r["delta_learned_mean"] for r in rows])),
        "mean_delta_editor": float(np.nanmean([r["delta_learned_editor"] for r in rows])),
        "mean_dG_editor": float(np.nanmean([r["dG_editor"] for r in rows])),
        "mean_dQ_editor": float(np.nanmean([r["dQ_editor"] for r in rows])),
        "mean_Y_learned": float(np.nanmean([r["Y_learned"] for r in rows])),
        "mean_Y_editor": float(np.nanmean([r["Y_editor"] for r in rows])),
        "mean_hamming": float(np.nanmean([float(r.get("hamming", np.nan)) for r in rows])),
        "mean_n_target_legal": float(np.nanmean([float(r.get("n_target_legal", np.nan)) for r in rows])),
        "mean_occupied": float(np.nanmean([float(r.get("occupied", np.nan)) for r in rows])),
    }
    sl["delta_wrong_ci"] = slice_boot([r["delta_learned_wrong"] for r in rows], [r["recip_game"] for r in rows], n_boot, boot_seed)
    sl["delta_random_ci"] = slice_boot([r["delta_learned_random"] for r in rows], [r["recip_game"] for r in rows], n_boot, boot_seed + 1)
    sl["delta_mean_ci"] = slice_boot([r["delta_learned_mean"] for r in rows], [r["recip_game"] for r in rows], n_boot, boot_seed + 2)
    sl["delta_editor_ci"] = slice_boot([r["delta_learned_editor"] for r in rows], [r["recip_game"] for r in rows], n_boot, boot_seed + 3)
    return sl


def stage_eval(cfg, out_dir: Path, args, *, smoke: bool) -> None:
    write_dir = out_dir / "smoke" if smoke else out_dir
    ensure_dir(write_dir)
    man = json.loads((out_dir / "null_state_pair_manifest.json").read_text(encoding="utf-8"))
    pack = np.load(out_dir / "null_state_pca_ridge.npz")
    means = {parse_stage_mean_name(k): np.asarray(v, dtype=np.float64) for k, v in json.loads((out_dir / "null_state_stage_means.json").read_text(encoding="utf-8")).items()}
    model = {
        "mu": pack["mu"], "V": pack["V"], "W": pack["W"], "c": pack["c"],
        "P_N": pack["P_N"], "k": int(pack["k"]), "alpha": float(pack["alpha"]),
        "stage_means": {stage_mean_name(k): v for k, v in means.items()},
    }
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    require_no_confirmation(val, holdout["game_ids"])
    by = _by_game(val)
    stack = _load_model_stack(cfg, args)
    if stack is None:
        print(STATUS_AWAITING)
        return
    H = cache_site(stack["model"], tokens_of(val), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    decoder = stack["decoder"]
    dec_t = decoder.as_torch()
    P_N = stack["P_N"]
    pairs = man["pairs"]
    wrongs = man["wrong_targets"]
    if smoke:
        i = int(cfg.get("smoke_pair_index", 0))
        pairs, wrongs = [pairs[i]], [wrongs[i]]
    sep = donor_separation_audit()
    if not sep["ok"]:
        raise SystemExit(f"{VERDICT_ENGINEERING} learned direction signature admits a donor")

    prepared = []
    learned_frozen = {}
    legal_doc = {}
    attempts = {"pairs": 0, "editor_fail": 0, "js_fail": 0, "invariance_fail": 0, "valid": 0, "gradient_unavailable": 0, "primary_unavailable": 0}
    for pair, wrep in zip(pairs, wrongs):
        attempts["pairs"] += 1
        rec = val[int(pair["recip"])]
        tgt = val[int(pair["donor"])]
        if rec["game_id"] != pair["recip_game"] or tgt["game_id"] != pair["target_game"]:
            raise SystemExit(f"{VERDICT_ENGINEERING} pair index/game mismatch")
        h_rec = np.asarray(H[int(pair["recip"])], dtype=np.float64)
        ed = iterative_state_edit(h_rec, tgt["rel_lab"], rec["rel_lab"], dec_t, steps=int(cfg.get("edit_steps", 40)), lr=float(cfg.get("edit_lr", 0.4)), budget=float(cfg.get("editor_budget", 16.0)), lambda_norm=float(cfg["lambda_norm"]), eta_preserve=float(cfg["eta_preserve"]), mu_manifold=float(cfg.get("mu_manifold", 0.0)))
        if not ed.get("valid_edit") or ed.get("edited") is None:
            attempts["editor_fail"] += 1
            continue
        h_e = np.asarray(ed["edited"], dtype=np.float64)
        b_t = symbolic_state(tgt)
        wd = (wrep.get("donors") or [None])[0]
        b_w = symbolic_state(val[int(wd["index"])]) if wd else None
        primary = {
            "learned": learned_null_delta(h_e, b_t, model),
            "wrong_target": learned_null_delta(h_e, b_w, model) if b_w is not None else None,
            "mean_state": mean_state_delta(h_e, stage_key(tgt), model),
            "random": P_N @ np.random.default_rng(int(cfg.get("random_seed", RANDOM_SEED)) + int(pair["recip"])).normal(size=P_N.shape[0]),
        }
        if any(primary[k] is None for k in PRIMARY_CELLS):
            attempts["primary_unavailable"] += 1
            continue
        learned_frozen[f"{pair['recip_game']}|{pair['target_game']}"] = np.asarray(primary["learned"], dtype=np.float64)
        became = _became(rec, tgt)
        legal_doc[f"{pair['recip_game']}|{pair['target_game']}"] = {
            "recipient": legal_record(rec),
            "target": legal_record(tgt),
            "wrong_target": legal_record(val[int(wd["index"])]) if wd else None,
            **became,
        }
        prepared.append({
            "pair": pair,
            "rec": rec,
            "tgt": tgt,
            "h_rec": h_rec,
            "h_e": h_e,
            "primary": primary,
            "became": became,
            "board_exact_editor": decoder_state_effects(decoder, h_e, rec["rel_lab"], tgt["rel_lab"])["board_exact"],
        })

    write_json(write_dir / "null_state_donor_separation.json", {**sep, "n_learned_frozen_before_oracle": len(learned_frozen), "oracle_computed_after_freeze": 1})
    if learned_frozen:
        np.savez_compressed(write_dir / "null_state_learned_directions.npz", **{k.replace("|", "__"): v for k, v in learned_frozen.items()})
    write_json(write_dir / "null_state_learned_directions.json", {"uses_donor_activation": 0, "n": len(learned_frozen), "frozen_before_oracle": 1})
    write_json(write_dir / "null_state_target_legal_sets.json", legal_doc)

    rows = []
    cals = []
    doses = []
    logit_pack = {}
    probe_pack = {}
    direction_pack = {}
    hooks_ok = 1
    inv_atol = float(cfg.get("probe_proba_atol_f32", 1e-5))
    js_target = float(cfg.get("fixed_js", FIXED_JS))
    n_boot = int(cfg.get("n_boot", 2000))
    boot_seed = int(cfg.get("boot_seed", 82))

    for item in prepared:
        pair, rec, tgt = item["pair"], item["rec"], item["tgt"]
        h_e, h_rec = item["h_e"], item["h_rec"]
        probs_u, logits_u, audit_u = _patch(stack["model"], rec["tokens"], h_rec)
        probs_e, logits_e, audit_e = _patch(stack["model"], rec["tokens"], h_e)
        hooks_ok = hooks_ok and int(audit_e.get("ok", 0)) and int(audit_u.get("ok", 0))
        y_e = y_components(probs_e, rec["legal"], tgt["legal"])
        probe_e = decoder.proba_np(h_e).reshape(64, 3)
        logit_pack[f"{pair['recip_game']}__{pair['target_game']}__unedited"] = np.asarray(logits_u, dtype=np.float64)
        logit_pack[f"{pair['recip_game']}__{pair['target_game']}__editor_A"] = np.asarray(logits_e, dtype=np.float64)
        probe_pack[f"{pair['recip_game']}__{pair['target_game']}__unedited"] = decoder.proba_np(h_rec).reshape(64, 3)
        probe_pack[f"{pair['recip_game']}__{pair['target_game']}__editor_A"] = probe_e
        dirs = dict(item["primary"])
        h_d = np.asarray(H[int(pair["donor"])], dtype=np.float64)
        dirs["oracle"] = oracle_null_delta(h_e, h_d, P_N)
        try:
            g = _y_grad(stack["model"], rec["tokens"], h_e, item["became"]["became_legal_tokens"], item["became"]["became_illegal_tokens"])
            dirs["gradient"] = (P_N @ g) if g is not None else None
        except Exception:
            dirs["gradient"] = None
            attempts["gradient_unavailable"] += 1
        if dirs["gradient"] is None and "gradient" not in [c.get("cell") for c in cals if c.get("recip_game") == pair["recip_game"]]:
            if not any(c.get("cell") == "gradient" and c.get("recip_game") == pair["recip_game"] for c in cals):
                pass
        y_by = {"editor_A": y_e}
        inv_ok = 1
        primary_ok = 1
        for name, delta in dirs.items():
            if delta is None:
                if name in PRIMARY_CELLS:
                    primary_ok = 0
                    attempts["primary_unavailable"] += 1
                else:
                    attempts["gradient_unavailable"] += int(name == "gradient")
                cals.append({"cell": name, "recip_game": pair["recip_game"], "target_game": pair["target_game"], "ok": 0, "reason": "unavailable", "target_js": js_target, "used_donor": int(name == "oracle")})
                continue
            direction_pack[f"{pair['recip_game']}__{pair['target_game']}__{name}"] = np.asarray(delta, dtype=np.float64)

            def ev(t, _d=delta):
                h = h_e + float(t) * np.asarray(_d)
                probs, logits, audit = _patch(stack["model"], rec["tokens"], h)
                return {"js": float(js_divergence(probs_e, probs)), "probs": probs, "logits": logits, "audit": audit, "h": h}

            matched = calibrate_fixed_js(ev, target_js=js_target, grid=SCALE_GRID, tol=float(cfg.get("fixed_js_tol", FIXED_JS_TOL)))
            for tr in matched.get("trace") or []:
                doses.append({"cell": name, "recip_game": pair["recip_game"], "target_game": pair["target_game"], "t": tr.get("t"), "js": tr.get("js"), "target_js": js_target})
            sc = scaled_displacement(delta, matched.get("t", float("nan"))) if matched.get("t") == matched.get("t") else {"l2": float("nan")}
            probe = decoder.proba_np(matched["h"]).reshape(64, 3) if matched.get("ok") and matched.get("h") is not None else None
            inv = float(np.max(np.abs(probe - probe_e))) if probe is not None else float("nan")
            cals.append({
                "cell": name, "recip_game": pair["recip_game"], "target_game": pair["target_game"],
                "ok": matched.get("ok", 0), "reason": matched.get("reason"), "target_js": js_target,
                "js": matched.get("js"), "t": matched.get("t"), "l2": sc.get("l2"),
                "invariance_max_err": inv, "used_donor": int(name == "oracle"),
            })
            if name in PRIMARY_CELLS and not matched.get("ok"):
                primary_ok = 0
                attempts["js_fail"] += 1
            if name in PRIMARY_CELLS and matched.get("ok") and inv > inv_atol:
                inv_ok = 0
                attempts["invariance_fail"] += 1
            if matched.get("ok"):
                y = y_components(matched["probs"], rec["legal"], tgt["legal"])
                y_by[name] = y
                logit_pack[f"{pair['recip_game']}__{pair['target_game']}__{name}"] = np.asarray(matched["logits"], dtype=np.float64)
                if probe is not None:
                    probe_pack[f"{pair['recip_game']}__{pair['target_game']}__{name}"] = probe
                rows.append({
                    "cell": name, "r2_type": pair["r2_type"], "recip_game": pair["recip_game"], "target_game": pair["target_game"],
                    "valid": 1, "Y": y["Y"], "G": y["G"], "Q": y["Q"], "js": matched["js"], "l2": sc.get("l2"),
                    "board_exact_editor": item["board_exact_editor"], "hamming": pair.get("hamming"),
                    "occupied": rec.get("occupied"), "n_target_legal": len(tgt["legal"]),
                })
        rows.append({
            "cell": "editor_A", "r2_type": pair["r2_type"], "recip_game": pair["recip_game"], "target_game": pair["target_game"],
            "valid": 1, **{k: y_e[k] for k in ("Y", "G", "Q")}, "board_exact_editor": item["board_exact_editor"],
            "hamming": pair.get("hamming"), "occupied": rec.get("occupied"), "n_target_legal": len(tgt["legal"]),
        })
        if inv_ok and primary_ok and all(c in y_by for c in PRIMARY_CELLS):
            attempts["valid"] += 1
            yl, yw, ym, yr, ye = y_by["learned"]["Y"], y_by["wrong_target"]["Y"], y_by["mean_state"]["Y"], y_by["random"]["Y"], y_e["Y"]
            rows.append({
                "cell": "pair_contrast", "r2_type": pair["r2_type"], "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 1,
                "delta_learned_wrong": float(yl - yw), "delta_learned_random": float(yl - yr), "delta_learned_mean": float(yl - ym),
                "delta_learned_editor": float(yl - ye), "Y_learned": yl, "Y_wrong": yw, "Y_random": yr, "Y_mean": ym, "Y_editor": ye,
                "Y_oracle": y_by.get("oracle", {}).get("Y") if isinstance(y_by.get("oracle"), dict) else float("nan"),
                "Y_gradient": y_by.get("gradient", {}).get("Y") if isinstance(y_by.get("gradient"), dict) else float("nan"),
                "G_learned": y_by["learned"]["G"], "Q_learned": y_by["learned"]["Q"], "G_editor": y_e["G"], "Q_editor": y_e["Q"],
                "dG_editor": float(y_by["learned"]["G"] - y_e["G"]), "dQ_editor": float(y_by["learned"]["Q"] - y_e["Q"]),
                "board_exact_editor": item["board_exact_editor"], "hamming": pair.get("hamming"),
                "occupied": rec.get("occupied"), "n_target_legal": len(tgt["legal"]),
            })

    write_csv(write_dir / "null_state_calibrations.csv", cals or [{"reason": "no_calibrations"}])
    write_csv(write_dir / "null_state_interventions.csv", rows or [{"reason": "no_rows"}])
    write_csv(write_dir / "null_state_dose_search.csv", doses or [{"reason": "no_search"}])
    if logit_pack:
        np.savez_compressed(write_dir / "null_state_full_logits.npz", **logit_pack)
    if probe_pack:
        np.savez_compressed(write_dir / "null_state_probe_probabilities.npz", **probe_pack)
    if direction_pack:
        np.savez_compressed(write_dir / "null_state_intervention_directions.npz", **direction_pack)
    contrasts = [r for r in rows if r.get("cell") == "pair_contrast" and r.get("valid") == 1]
    slices = {
        "pooled": _slice_stats(contrasts, boot_seed, n_boot),
        "A": _slice_stats([r for r in contrasts if r.get("r2_type") == "A_legal_diff"], boot_seed + 10, n_boot),
        "B": _slice_stats([r for r in contrasts if r.get("r2_type") == "B_similar_board_legal_change"], boot_seed + 20, n_boot),
        "exact": _slice_stats([r for r in contrasts if r.get("board_exact_editor") in {1, True, "1", 1.0}], boot_seed + 30, n_boot),
        "nonexact": _slice_stats([r for r in contrasts if r.get("board_exact_editor") not in {1, True, "1", 1.0}], boot_seed + 40, n_boot),
    }
    boot = []
    for name, sl in slices.items():
        if sl.get("n", 0) == 0:
            continue
        for met, key in (("wrong", "delta_wrong_ci"), ("random", "delta_random_ci"), ("mean", "delta_mean_ci"), ("editor", "delta_editor_ci")):
            boot.append({"slice": name, "metric": met, **sl[key]})
    pooled = slices.get("pooled") or {}
    wci, rci, mci, eci = pooled.get("delta_wrong_ci") or {}, pooled.get("delta_random_ci") or {}, pooled.get("delta_mean_ci") or {}, pooled.get("delta_editor_ci") or {}
    prior_split = json.loads((out_dir / "null_state_split_leakage.json").read_text(encoding="utf-8")) if (out_dir / "null_state_split_leakage.json").is_file() else {"ok": 0}
    train_ids = json.loads((out_dir / "null_state_train_game_ids.json").read_text(encoding="utf-8")) if (out_dir / "null_state_train_game_ids.json").is_file() else {"games": []}
    leak = split_leakage_audit(
        train_ids.get("games") or [],
        train_ids.get("games") or [],
        [p["recip_game"] for p in man["pairs"]],
        [p["target_game"] for p in man["pairs"]],
        [d["game_id"] for w in man.get("wrong_targets") or [] for d in (w.get("donors") or [])],
        json.loads((out_dir / "null_state_split_games.json").read_text(encoding="utf-8")).get("excluded_prior") or [],
        holdout["game_ids"],
    )
    write_json(write_dir / "null_state_split_leakage.json", leak)
    n_pairs = len(pairs)
    gate = {
        "split_ok": int(leak["ok"] and prior_split.get("ok", 0)),
        "no_donor_leak": int(sep["ok"] == 1),
        "invariance_ok": int(attempts["invariance_fail"] == 0 and attempts["valid"] > 0),
        "js_match_ok": int(attempts["valid"] > 0),
        "coverage_ok": int(attempts["valid"] >= (1 if smoke else int(cfg.get("coverage_min", COVERAGE_MIN)))),
        "mean_vs_wrong_ge_floor": int(pooled.get("mean_delta_wrong", 0) >= float(cfg.get("behavior_floor", BEHAVIOR_FLOOR))) if pooled.get("n", 0) else 0,
        "ci_wrong_gt_0": int(wci.get("ci95_low", 0) > 0) if wci else 0,
        "beats_random": int(pooled.get("mean_delta_random", 0) > 0 and (rci.get("ci95_low", 0) > 0 if rci else 0)),
        "beats_mean_state": int(pooled.get("mean_delta_mean", 0) > 0 and (mci.get("ci95_low", 0) > 0 if mci else 0)),
        "beats_editor": int(pooled.get("mean_delta_editor", 0) > 0 and (eci.get("ci95_low", 0) > 0 if eci else 0)),
        "components_vs_editor_ok": int(pooled.get("mean_dG_editor", 0) > 0 and pooled.get("mean_dQ_editor", 0) < 0) if pooled.get("n", 0) else 0,
        "integrity_ok": int(hooks_ok and attempts["pairs"] == n_pairs),
    }
    verdict = decide_nst_verdict(gate)
    write_csv(write_dir / "null_state_bootstrap.csv", boot or [{"n": 0}])
    write_json(write_dir / "null_state_attempt_accounting.json", attempts)
    write_json(write_dir / "null_state_dependency_manifest.json", {
        "git_hash": git_hash(),
        "checkpoint_sha256": sha256_file(stack["ckpt"]),
        "probe_sha256": sha256_file(stack["probe_info"]["probe_path"]),
        "pair_hash": man.get("pair_manifest_sha256"),
        "used_confirmation": 0,
        "official_0A_letter": OFFICIAL_0A_LETTER,
    })
    write_json(write_dir / "null_state_completion_status.json", {
        "status": "SMOKE_COMPLETE" if smoke else "NULL_STATE_TRANSFER_COMPLETE",
        "verdict": "SMOKE_ONLY" if smoke else verdict,
        "gate": gate, "slices": slices, "attempts": attempts, "smoke": int(smoke),
        "used_confirmation": 0, "official_0A_letter": OFFICIAL_0A_LETTER,
        "fixed_js_status_preserved": "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED",
        "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
        "prior_cost_verdict_preserved": "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "section_5_3_status": "PARTIAL", "obstruction_ready": "NO",
        "secondary_cells": list(SECONDARY_CELLS),
        "primary_cells": list(PRIMARY_CELLS),
    })
    (write_dir / "null_state_scientific_report.md").write_text(
        "\n".join([
            "# 08R2-NULL-STATE-TRANSFER Scientific Report",
            "",
            f"Verdict: `{'SMOKE_ONLY' if smoke else verdict}`",
            "",
            "This is a donor-free transfer test, not a new decision germ.",
            "High residual R² is not a causal verdict. Oracle uses the donor and is not donor-free.",
            "Gradient steering is a baseline, not a learned representation.",
            "",
            f"Attempts: {attempts}",
            f"Gate: {gate}",
            f"Pooled: {pooled}",
            f"A: {slices.get('A')}",
            f"B: {slices.get('B')}",
            f"Exact: {slices.get('exact')}",
            f"Nonexact: {slices.get('nonexact')}",
            "",
            "Official letters unchanged. No obstruction. No confirmation. No 08R3.",
            "- `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`",
            "- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`",
            "- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`",
            "- `08R2_VALIDATION_COMPLETE — NEGATIVE`",
            "- `0A C / PARTIAL`",
            "- `SECTION_5_3 = PARTIAL`",
            "- `OBSTRUCTION_READY = NO`",
            "",
        ]),
        encoding="utf-8",
    )
    print("08R2-NULL-STATE-TRANSFER")
    print(f"  verdict={'SMOKE_ONLY' if smoke else verdict}")
    print(f"  valid_pairs={attempts['valid']}/{n_pairs}")
    print(f"  attempts={attempts}")
    print(f"  gate={gate}")
    print(f"  pooled={pooled}")
    print("  FIXED_JS_TARGET_SPECIFICITY_SUPPORTED preserved")
    print("  TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED preserved")
    print("  TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE preserved")
    print("  official letter unchanged: C / PARTIAL")
    if smoke and attempts["valid"] != 1:
        raise SystemExit(f"{VERDICT_ENGINEERING} smoke failed to complete the primary pair")


def _write_protocol(cfg, out_dir: Path) -> None:
    prereg = Path(cfg.get("preregistration", "docs/experiments/08r2_null_state_transfer_preregistration.md"))
    if prereg.is_file():
        text = prereg.read_text(encoding="utf-8")
        (out_dir / "null_state_preregistration.md").write_text(text, encoding="utf-8")
        protocol = {
            "experiment": "08R2-NULL-STATE-TRANSFER",
            "ks": list(cfg.get("ks", KS)),
            "alphas": list(cfg.get("alphas", ALPHAS)),
            "pair_seed": int(cfg.get("pair_seed", PAIR_SEED)),
            "wrong_seed": int(cfg.get("wrong_seed", WRONG_SEED)),
            "random_seed": int(cfg.get("random_seed", RANDOM_SEED)),
            "cv_seed": int(cfg.get("cv_seed", CV_SEED)),
            "fixed_js": float(cfg.get("fixed_js", FIXED_JS)),
            "fixed_js_tol": float(cfg.get("fixed_js_tol", FIXED_JS_TOL)),
            "behavior_floor": float(cfg.get("behavior_floor", BEHAVIOR_FLOOR)),
            "coverage_min": int(cfg.get("coverage_min", COVERAGE_MIN)),
            "n_a": int(cfg.get("n_a", WANT_A)),
            "n_b": int(cfg.get("n_b", WANT_B)),
            "primary_cells": list(PRIMARY_CELLS),
            "secondary_cells": list(SECONDARY_CELLS),
            "feature_names": list(FEATURE_NAMES),
            "scale_grid": list(SCALE_GRID),
            "preregistration_sha256": _sha256_text(text),
        }
        write_json(out_dir / "null_state_protocol.json", protocol)
        write_json(out_dir / "null_state_protocol_hash.json", {"sha256": _sha256_text(json.dumps(protocol, sort_keys=True)), "preregistration_sha256": protocol["preregistration_sha256"]})


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
    out_dir = ensure_dir(isolate_nst_dir(cfg["results_dir"])["results_dir"])
    _write_protocol(cfg, out_dir)
    stages = ["0", "1", "2", "3", "4"] if args.stage == "all" else [args.stage]
    if "0" in stages:
        stage0(cfg, out_dir, args)
    if "1" in stages:
        stage1(cfg, out_dir)
    if "2" in stages:
        if not (out_dir / "null_state_pair_manifest.json").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 1 manifest missing")
        stage2(cfg, out_dir, args)
    if "3" in stages:
        if not (out_dir / "null_state_pca_ridge.npz").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 2 model missing")
        stage_eval(cfg, out_dir, args, smoke=True)
    if "4" in stages:
        if not (out_dir / "null_state_pca_ridge.npz").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 2 model missing")
        stage_eval(cfg, out_dir, args, smoke=False)


if __name__ == "__main__":
    main()
