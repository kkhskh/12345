"""08R2-STATE-SEMANTICS exploratory ablation of the frozen donor-free map.

Does not overwrite NST/SPEC/COST/fixed-JS. No confirmation. Letter C / PARTIAL.
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
from src.othello_r2_fiber import independent_contrasts, js_divergence, reject_unexpected_preprocessing, softmax, svd_probe_geometry
from src.othello_r2_fiber_cost import adopt_drive_defaults
from src.othello_r2_fiber_cost_fixed import SCALE_GRID, calibrate_fixed_js, scaled_displacement
from src.othello_r2_fiber_spec import y_components
from src.othello_r2_null_state import (
    FEATURE_NAMES,
    audit_saved_y,
    grouped_folds,
    legal_record,
    pca_coords,
    predict_z,
    r2_score,
    rescore_y_from_logits,
    resolve_pair_rows,
    saved_y_present,
    symbolic_state,
    verify_simulator_legal,
)
from src.othello_r2_state_semantics import (
    ABLATIONS,
    BEHAVIOR_FLOOR,
    COVERAGE_MIN,
    CV_SEED,
    DIR_ATOL,
    FIXED_JS,
    FIXED_JS_TOL,
    FROZEN_ALPHA,
    FROZEN_K,
    JS_METHODS,
    N_BOOT,
    N_FOLDS,
    NST_PAIR_HASH,
    PRIMARY_JS,
    STATUS_AWAITING,
    STATUS_STAGE0_FAIL,
    STATUS_STAGE1_FAIL,
    VERDICT_ENGINEERING,
    Y_RECON_ATOL,
    ablation_delta,
    crossover_scores,
    decide_semantics_verdict,
    decompose_full_stage,
    direction_parity,
    feature_names_for,
    fit_frozen_ridge,
    increment_ok,
    isolate_sem_dir,
    match_controls,
    pack_ablation,
    pack_full_from_nst,
    paired_boot,
    predict_zhat,
    probe_image_residual,
)
from src.othello_runutil import add_labels, load_split, tokens_of, write_json
from src.toy_othello import cache_site, load_config, load_othello_gpt, softmax_np, square_to_token, verify_nanda_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-STATE-SEMANTICS")
    p.add_argument("--config", default="configs/toy_othello_08r2_state_semantics.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--stage", default="0", choices=["0", "1", "2", "3", "4", "5", "6", "all"])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--nst-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def merge_cfg(*parts):
    out = {}
    for p in parts:
        out.update(p)
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_nst_dir(cfg, args, out_dir: Path) -> Path:
    cands = []
    if args.nst_dir:
        cands.append(Path(args.nst_dir))
    cands.append(out_dir.parent / "null_state_transfer")
    cands.append(Path(cfg["results_dir"]) / "results" / "08r2" / "null_state_transfer")
    for path in cands:
        if path.is_dir() and (path / "null_state_pca_ridge.npz").is_file() and (path / "null_state_pair_manifest.json").is_file():
            return path
    return cands[0] if args.nst_dir else cands[1]


def _load_stack(cfg, args):
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
    return {"probe_info": probe_info, "ckpt": ckpt, "pack": pack, "decoder": decoder, "contrasts": contrasts, "model": model, "ident": ident, "svd": svd, "P_N": svd["P_N"], "W_probe": contrasts["W"]}


def _patch(model, tokens, h):
    vals = __import__("torch").tensor(np.asarray(h, dtype=np.float32).reshape(1, -1))
    logits, readback, audit = patch_and_audit(model, [tokens], SITE, vals, batch_size=1)
    return softmax_np(logits.numpy()[0]), logits.numpy()[0], audit


def _editor(cfg, decoder, rec, tgt, h_rec):
    return iterative_state_edit(
        h_rec, tgt["rel_lab"], rec["rel_lab"], decoder.as_torch(),
        steps=int(cfg.get("edit_steps", 40)), lr=float(cfg.get("edit_lr", 0.4)),
        budget=float(cfg.get("editor_budget", 16.0)), lambda_norm=float(cfg["lambda_norm"]),
        eta_preserve=float(cfg["eta_preserve"]), mu_manifold=float(cfg.get("mu_manifold", 0.0)),
    )


def stage0(cfg, out_dir: Path, nst: Path) -> dict:
    man = json.loads((nst / "null_state_pair_manifest.json").read_text(encoding="utf-8"))
    expected = cfg.get("expected_nst_pair_hash", NST_PAIR_HASH)
    if man.get("pair_manifest_sha256") != expected:
        write_json(out_dir / "state_semantics_stage0_status.json", {"ok": 0, "reason": "pair_hash_mismatch", "got": man.get("pair_manifest_sha256"), "expected": expected})
        raise SystemExit(STATUS_STAGE0_FAIL)
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    require_no_confirmation(val, holdout["game_ids"])
    pair_by = {(p["recip_game"], p["target_game"]): p for p in man["pairs"]}
    inter = list(csv.DictReader((nst / "null_state_interventions.csv").open(encoding="utf-8")))
    logits = np.load(nst / "null_state_full_logits.npz")
    n_ok = n_bad = n_legal = 0
    rows = []
    for row in inter:
        rg, tg, cell = row.get("recip_game"), row.get("target_game"), row.get("cell")
        if cell in {"pair_contrast"} or (rg, tg) not in pair_by:
            continue
        resolved = resolve_pair_rows(val, pair_by[(rg, tg)])
        rec, tgt = resolved["rec"], resolved["tgt"]
        if not verify_simulator_legal(rec)["ok"] or not verify_simulator_legal(tgt)["ok"]:
            n_legal += 1
        lkey = f"{rg}__{tg}__{cell}"
        if lkey not in logits.files:
            if cell == "learned":
                n_bad += 1
            continue
        recon = rescore_y_from_logits(logits[lkey], rec["legal"], tgt["legal"], probs=softmax_np(logits[lkey]) if cell != "editor_A" or saved_y_present(row) else None)
        if not saved_y_present(row):
            continue
        chk = audit_saved_y(row, recon, float(cfg.get("y_recon_atol", Y_RECON_ATOL)))
        rows.append({"cell": cell, "recip_game": rg, "target_game": tg, "ok": chk["ok"], "abs_Y": chk.get("abs_Y")})
        n_ok += int(chk["ok"] == 1)
        n_bad += int(chk["ok"] == 0)
    status = {
        "ok": int(n_bad == 0 and n_legal == 0 and n_ok > 0),
        "n_match": n_ok, "n_mismatch": n_bad, "n_legal_mismatch": n_legal,
        "pair_hash": expected, "nst_dir": str(nst),
        "exploratory": 1, "not_independent_confirmation": 1,
    }
    write_csv(out_dir / "state_semantics_stage0_y_rescore.csv", rows or [{"reason": "no_rows"}])
    write_json(out_dir / "state_semantics_stage0_status.json", status)
    print("08R2-STATE-SEMANTICS STAGE 0")
    print(f"  ok={status['ok']} match={n_ok} mismatch={n_bad} legal_bad={n_legal}")
    if not status["ok"]:
        raise SystemExit(STATUS_STAGE0_FAIL)
    return {"man": man, "val": val, "status": status}


def stage1(cfg, out_dir: Path, args, nst: Path, man: dict, val: list) -> dict:
    stack = _load_stack(cfg, args)
    if stack is None:
        write_json(out_dir / "state_semantics_completion_status.json", {"status": STATUS_AWAITING, "reason": "BLOCKED_MISSING_ARTIFACT", "stage": 1})
        print(STATUS_AWAITING)
        return {}
    nst_pack = np.load(nst / "null_state_pca_ridge.npz")
    if int(nst_pack["k"]) != int(cfg.get("frozen_k", FROZEN_K)) or abs(float(nst_pack["alpha"]) - float(cfg.get("frozen_alpha", FROZEN_ALPHA))) > 1e-12:
        raise SystemExit(f"{STATUS_STAGE1_FAIL} frozen k/alpha mismatch")
    full = pack_full_from_nst(nst_pack)
    saved = np.load(nst / "null_state_learned_directions.npz")
    H = cache_site(stack["model"], tokens_of(val), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    rows = []
    h_pack = {}
    n_ok = n_bad = 0
    max_err = 0.0
    for pair in man["pairs"]:
        rec, tgt = resolve_pair_rows(val, pair)["rec"], resolve_pair_rows(val, pair)["tgt"]
        ed = _editor(cfg, stack["decoder"], rec, tgt, np.asarray(H[int(pair["recip"])], dtype=np.float64))
        if not ed.get("valid_edit") or ed.get("edited") is None:
            n_bad += 1
            rows.append({"recip_game": pair["recip_game"], "target_game": pair["target_game"], "ok": 0, "reason": "editor_fail"})
            continue
        h_e = np.asarray(ed["edited"], dtype=np.float64)
        key = f"{pair['recip_game']}__{pair['target_game']}"
        h_pack[key] = h_e
        d = ablation_delta(h_e, symbolic_state(tgt), full)
        skey = key
        if skey not in saved.files:
            skey = f"{pair['recip_game']}|{pair['target_game']}"
        if skey not in saved.files:
            n_bad += 1
            rows.append({"recip_game": pair["recip_game"], "target_game": pair["target_game"], "ok": 0, "reason": "missing_saved_dir"})
            continue
        chk = direction_parity(d, saved[skey], float(cfg.get("dir_atol", DIR_ATOL)))
        max_err = max(max_err, chk["l2_err"])
        n_ok += int(chk["ok"] == 1)
        n_bad += int(chk["ok"] == 0)
        rows.append({"recip_game": pair["recip_game"], "target_game": pair["target_game"], "ok": chk["ok"], "l2_err": chk["l2_err"]})
    np.savez_compressed(out_dir / "state_semantics_editor_h.npz", **h_pack)
    status = {"ok": int(n_bad == 0 and n_ok == len(man["pairs"])), "n_match": n_ok, "n_mismatch": n_bad, "max_l2_err": max_err, "n_editor": len(h_pack)}
    write_csv(out_dir / "state_semantics_direction_parity.csv", rows)
    write_json(out_dir / "state_semantics_stage1_status.json", status)
    print("08R2-STATE-SEMANTICS STAGE 1")
    print(f"  ok={status['ok']} match={n_ok}/{len(man['pairs'])} max_l2_err={max_err:.3e}")
    if not status["ok"]:
        raise SystemExit(STATUS_STAGE1_FAIL)
    return {"full": full, "stack": stack, "H": H}


def stage2(cfg, out_dir: Path, args, nst: Path) -> dict:
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    train = add_labels(load_split(Path(cfg["data_dir"]), "train"))
    require_no_confirmation(train, holdout["game_ids"])
    stack = _load_stack(cfg, args)
    if stack is None:
        print(STATUS_AWAITING)
        return {}
    nst_pack = np.load(nst / "null_state_pca_ridge.npz")
    pca = {"mu": nst_pack["mu"], "V": nst_pack["V"], "k": int(nst_pack["k"])}
    P_N = nst_pack["P_N"]
    train_nat = [r for r in train if (not r.get("terminal")) and r.get("legal")]
    H = cache_site(stack["model"], tokens_of(train_nat), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    Z, B, games = [], [], []
    for i, row in enumerate(train_nat):
        Z.append(P_N @ np.asarray(H[i], dtype=np.float64))
        B.append(symbolic_state(row))
        games.append(str(row["game_id"]))
    Z, B = np.asarray(Z), np.asarray(B)
    table = []
    packs = {"FULL": pack_full_from_nst(nst_pack)}
    folds = grouped_folds(games, int(cfg.get("n_folds", N_FOLDS)), int(cfg.get("cv_seed", CV_SEED)))
    full_scores = []
    for held in folds:
        held = set(map(str, held))
        te = np.array([g in held for g in games])
        if te.sum() < 4:
            continue
        pred = predict_z(B[te], pca, {"W": nst_pack["W"], "c": nst_pack["c"]})
        full_scores.append(r2_score(Z[te], pred))
    table.append({"name": "FULL", "cv_r2": float(np.nanmean(full_scores)) if full_scores else float("nan"), "train_r2": r2_score(Z, predict_z(B, pca, {"W": nst_pack["W"], "c": nst_pack["c"]})), "n_features": 258, "refit": 0})
    for name in ("BOARD_ONLY", "LEGAL_ONLY", "STAGE_ONLY"):
        fit = fit_frozen_ridge(Z, B, games, pca, name, float(cfg.get("frozen_alpha", FROZEN_ALPHA)), int(cfg.get("n_folds", N_FOLDS)), int(cfg.get("cv_seed", CV_SEED)))
        packs[name] = pack_ablation(fit, pca, P_N)
        table.append({"name": name, "cv_r2": fit["cv_r2"], "train_r2": fit["train_r2"], "n_features": fit["n_features"], "refit": 1})
        np.savez_compressed(out_dir / f"state_semantics_{name.lower()}.npz", mu=pca["mu"], V=pca["V"], W=fit["ridge"]["W"], c=fit["ridge"]["c"], idx=fit["idx"], P_N=P_N, k=pca["k"], alpha=fit["alpha"])
        write_json(out_dir / f"state_semantics_{name.lower()}_features.json", {"names": feature_names_for(name)})
    write_csv(out_dir / "state_semantics_cv_table.csv", table)
    write_json(out_dir / "state_semantics_train_provenance.json", {"n_rows": int(Z.shape[0]), "n_games": len(set(games)), "games": sorted(set(games)), "pca_refit": 0, "k": int(pca["k"]), "alpha": float(cfg.get("frozen_alpha", FROZEN_ALPHA))})
    print("08R2-STATE-SEMANTICS STAGE 2")
    for row in table:
        print(f"  {row['name']} cv_r2={row['cv_r2']:.4f} train_r2={row['train_r2']:.4f} n_features={row['n_features']}")
    return {"packs": packs, "stack": stack}


def stage3(cfg, out_dir: Path, nst: Path) -> None:
    protocol = {
        "experiment": "08R2-STATE-SEMANTICS",
        "exploratory": 1,
        "not_independent_confirmation": 1,
        "pair_hash": cfg.get("expected_nst_pair_hash", NST_PAIR_HASH),
        "frozen_k": int(cfg.get("frozen_k", FROZEN_K)),
        "frozen_alpha": float(cfg.get("frozen_alpha", FROZEN_ALPHA)),
        "fixed_js": float(cfg.get("fixed_js", FIXED_JS)),
        "behavior_floor": float(cfg.get("behavior_floor", BEHAVIOR_FLOOR)),
        "coverage_min": int(cfg.get("coverage_min", COVERAGE_MIN)),
        "n_boot": int(cfg.get("n_boot", N_BOOT)),
        "primary_js": list(PRIMARY_JS),
        "js_methods": list(JS_METHODS),
        "ablations": list(ABLATIONS),
        "estimands": ["FULL-LEGAL_ONLY", "FULL-BOARD_ONLY", "BOARD_ONLY-STAGE_ONLY", "LEGAL_ONLY-STAGE_ONLY", "A_T", "A_W", "C", "I_Y_t1"],
        "scale_grid": list(SCALE_GRID),
    }
    write_json(out_dir / "state_semantics_protocol.json", protocol)
    write_json(out_dir / "state_semantics_protocol_hash.json", {"sha256": _sha256_text(json.dumps(protocol, sort_keys=True))})
    print("08R2-STATE-SEMANTICS STAGE 3 FROZEN")
    print(f"  protocol_sha256={_sha256_text(json.dumps(protocol, sort_keys=True))}")


def _load_packs(out_dir: Path, nst: Path) -> dict:
    nst_pack = np.load(nst / "null_state_pca_ridge.npz")
    packs = {"FULL": pack_full_from_nst(nst_pack)}
    for name in ("BOARD_ONLY", "LEGAL_ONLY", "STAGE_ONLY"):
        z = np.load(out_dir / f"state_semantics_{name.lower()}.npz")
        packs[name] = {
            "name": name, "mu": z["mu"], "V": z["V"], "W": z["W"], "c": z["c"],
            "idx": z["idx"], "P_N": z["P_N"], "k": int(z["k"]), "alpha": float(z["alpha"]),
            "feature_names": feature_names_for(name),
        }
    return packs


def _slice(rows, pred, n_boot, seed):
    g = [r for r in rows if pred(r)]
    if not g:
        return {"n": 0}
    out = {"n": len(g)}
    for key in ("d_board_inc", "d_legal_inc", "d_board_vs_stage", "d_legal_vs_stage", "d_full_editor", "A_T", "A_W", "C"):
        if key not in g[0]:
            continue
        boot = paired_boot([r[key] for r in g], n_boot, seed)
        out[key] = boot
    return out


def stage_eval(cfg, out_dir: Path, args, nst: Path, *, smoke: bool) -> None:
    write_dir = out_dir / "smoke" if smoke else out_dir
    ensure_dir(write_dir)
    man = json.loads((nst / "null_state_pair_manifest.json").read_text(encoding="utf-8"))
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    require_no_confirmation(val, holdout["game_ids"])
    stack = _load_stack(cfg, args)
    if stack is None:
        print(STATUS_AWAITING)
        return
    packs = _load_packs(out_dir, nst)
    H = cache_site(stack["model"], tokens_of(val), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    decoder = stack["decoder"]
    W_probe = stack["W_probe"]
    pairs = man["pairs"]
    wrongs = man["wrong_targets"]
    if smoke:
        i = int(cfg.get("smoke_pair_index", 0))
        pairs, wrongs = [pairs[i]], [wrongs[i]]
    h_cache = {}
    if (out_dir / "state_semantics_editor_h.npz").is_file():
        h_cache = dict(np.load(out_dir / "state_semantics_editor_h.npz"))
    js_target = float(cfg.get("fixed_js", FIXED_JS))
    inv_atol = float(cfg.get("probe_proba_atol_f32", 1e-5))
    n_boot = int(cfg.get("n_boot", N_BOOT))
    attempts = {"pairs": 0, "editor_fail": 0, "js_fail": 0, "invariance_fail": 0, "valid": 0, "crossover_complete": 0}
    rows, cals, doses, matches = [], [], [], []
    logit_pack, probe_pack, dir_pack = {}, {}, {}
    hooks_ok = 1

    for pair, wrep in zip(pairs, wrongs):
        attempts["pairs"] += 1
        rec = resolve_pair_rows(val, pair)["rec"]
        tgt = resolve_pair_rows(val, pair)["tgt"]
        wd = (wrep.get("donors") or [None])[0]
        if wd is None:
            attempts["editor_fail"] += 1
            continue
        wrong = val[int(wd["index"])]
        key = f"{pair['recip_game']}__{pair['target_game']}"
        if key in h_cache:
            h_e = np.asarray(h_cache[key], dtype=np.float64)
        else:
            ed = _editor(cfg, decoder, rec, tgt, np.asarray(H[int(pair["recip"])], dtype=np.float64))
            if not ed.get("valid_edit") or ed.get("edited") is None:
                attempts["editor_fail"] += 1
                continue
            h_e = np.asarray(ed["edited"], dtype=np.float64)
        probs_e, logits_e, audit_e = _patch(stack["model"], rec["tokens"], h_e)
        hooks_ok = hooks_ok and int(audit_e.get("ok", 0))
        probe_e = decoder.proba_np(h_e).reshape(64, 3)
        y_e_t = y_components(probs_e, rec["legal"], tgt["legal"])
        y_e_w = y_components(probs_e, rec["legal"], wrong["legal"])
        logit_pack[f"{key}__editor_A"] = np.asarray(logits_e, dtype=np.float64)
        probe_pack[f"{key}__editor_A"] = probe_e
        b_t, b_w = symbolic_state(tgt), symbolic_state(wrong)
        dec = decompose_full_stage(h_e, b_t, packs["FULL"], packs["STAGE_ONLY"])
        dirs_t = {name: ablation_delta(h_e, b_t, packs[name]) for name in ABLATIONS}
        dirs_t["TARGET_COMP"] = dec["delta_target"]
        dirs_w = {name: ablation_delta(h_e, b_w, packs[name]) for name in ABLATIONS}
        exact = decoder_state_effects(decoder, h_e, rec["rel_lab"], tgt["rel_lab"])["board_exact"]
        match = match_controls(rec, tgt, wrong)
        match.update({"recip_game": pair["recip_game"], "target_game": pair["target_game"], "wrong_game": wrong["game_id"], "r2_type": pair["r2_type"]})
        matches.append(match)

        y_js = {"editor_A": y_e_t}
        inv_ok = 1
        primary_ok = 1
        y_cross = {}

        def run_dir(name, delta, score_rec, score_tgt, tag):
            nonlocal inv_ok, primary_ok, hooks_ok
            dir_pack[f"{key}__{tag}"] = np.asarray(delta, dtype=np.float64)
            resid = probe_image_residual(delta, W_probe)

            def ev(t, _d=delta):
                h = h_e + float(t) * np.asarray(_d)
                probs, logits, audit = _patch(stack["model"], rec["tokens"], h)
                return {"js": float(js_divergence(probs_e, probs)), "probs": probs, "logits": logits, "audit": audit, "h": h}

            matched = calibrate_fixed_js(ev, target_js=js_target, grid=SCALE_GRID, tol=float(cfg.get("fixed_js_tol", FIXED_JS_TOL)))
            for tr in matched.get("trace") or []:
                doses.append({"cell": tag, "recip_game": pair["recip_game"], "target_game": pair["target_game"], "t": tr.get("t"), "js": tr.get("js")})
            sc = scaled_displacement(delta, matched.get("t", float("nan"))) if matched.get("t") == matched.get("t") else {"l2": float("nan")}
            probe = decoder.proba_np(matched["h"]).reshape(64, 3) if matched.get("ok") and matched.get("h") is not None else None
            inv = float(np.max(np.abs(probe - probe_e))) if probe is not None else float("nan")
            cals.append({"cell": tag, "recip_game": pair["recip_game"], "target_game": pair["target_game"], "ok": matched.get("ok", 0), "reason": matched.get("reason"), "js": matched.get("js"), "t": matched.get("t"), "l2": sc.get("l2"), "probe_image": resid, "invariance_max_err": inv})
            if not matched.get("ok"):
                if name in PRIMARY_JS and tag.endswith("_T"):
                    primary_ok = 0
                    attempts["js_fail"] += 1
                return None
            if inv > inv_atol:
                inv_ok = 0
                attempts["invariance_fail"] += 1
            hooks_ok = hooks_ok and int(matched.get("audit", {}).get("ok", 0))
            y = y_components(matched["probs"], score_rec["legal"], score_tgt["legal"])
            logit_pack[f"{key}__{tag}"] = np.asarray(matched["logits"], dtype=np.float64)
            if probe is not None:
                probe_pack[f"{key}__{tag}"] = probe
            rows.append({"cell": tag, "r2_type": pair["r2_type"], "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 1, "Y": y["Y"], "G": y["G"], "Q": y["Q"], "js": matched["js"], "l2": sc.get("l2"), "board_exact_editor": exact, "probe_image": resid})
            return y

        for name, delta in dirs_t.items():
            y = run_dir(name, delta, rec, tgt, f"{name}_T")
            if y is not None:
                y_js[name] = y
                y_cross[f"{name}_T_on_T"] = y
                y_cross[f"{name}_T_on_W"] = y_components(softmax_np(logit_pack[f"{key}__{name}_T"]), rec["legal"], wrong["legal"]) if f"{key}__{name}_T" in logit_pack else None
        for name, delta in dirs_w.items():
            y = run_dir(name, delta, rec, tgt, f"{name}_W")
            if y is not None and f"{key}__{name}_W" in logit_pack:
                y_cross[f"{name}_W_on_T"] = y
                y_cross[f"{name}_W_on_W"] = y_components(softmax_np(logit_pack[f"{key}__{name}_W"]), rec["legal"], wrong["legal"])

        t1_y = {}
        for tag, delta in (("FULL_t1", dec["delta_full"]), ("GENERIC_t1", dec["delta_generic"]), ("TARGET_t1", dec["delta_target"])):
            h = h_e + delta
            probs, logits, audit = _patch(stack["model"], rec["tokens"], h)
            hooks_ok = hooks_ok and int(audit.get("ok", 0))
            y = y_components(probs, rec["legal"], tgt["legal"])
            t1_y[tag] = y
            logit_pack[f"{key}__{tag}"] = np.asarray(logits, dtype=np.float64)
            rows.append({"cell": tag, "r2_type": pair["r2_type"], "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 1, "Y": y["Y"], "G": y["G"], "Q": y["Q"], "t": 1.0, "l2": float(np.linalg.norm(delta)), "board_exact_editor": exact})

        if inv_ok and primary_ok and all(n in y_js for n in PRIMARY_JS):
            attempts["valid"] += 1
            rec_row = {
                "cell": "pair_contrast", "r2_type": pair["r2_type"], "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 1,
                "board_exact_editor": exact, "hamming": pair.get("hamming"), "occupied": rec.get("occupied"),
                "Y_FULL": y_js["FULL"]["Y"], "Y_BOARD_ONLY": y_js["BOARD_ONLY"]["Y"], "Y_LEGAL_ONLY": y_js["LEGAL_ONLY"]["Y"], "Y_STAGE_ONLY": y_js["STAGE_ONLY"]["Y"],
                "Y_editor": y_e_t["Y"], "G_FULL": y_js["FULL"]["G"], "Q_FULL": y_js["FULL"]["Q"],
                "d_board_inc": float(y_js["FULL"]["Y"] - y_js["LEGAL_ONLY"]["Y"]),
                "d_legal_inc": float(y_js["FULL"]["Y"] - y_js["BOARD_ONLY"]["Y"]),
                "d_board_vs_stage": float(y_js["BOARD_ONLY"]["Y"] - y_js["STAGE_ONLY"]["Y"]),
                "d_legal_vs_stage": float(y_js["LEGAL_ONLY"]["Y"] - y_js["STAGE_ONLY"]["Y"]),
                "d_full_editor": float(y_js["FULL"]["Y"] - y_e_t["Y"]),
                "dY_full_t1": float(t1_y["FULL_t1"]["Y"] - y_e_t["Y"]),
                "dY_generic_t1": float(t1_y["GENERIC_t1"]["Y"] - y_e_t["Y"]),
                "dY_target_t1": float(t1_y["TARGET_t1"]["Y"] - y_e_t["Y"]),
                "I_Y": float((t1_y["FULL_t1"]["Y"] - y_e_t["Y"]) - (t1_y["GENERIC_t1"]["Y"] - y_e_t["Y"]) - (t1_y["TARGET_t1"]["Y"] - y_e_t["Y"])),
                "sum_err": dec["sum_err"],
            }
            if all(y_cross.get(k) is not None for k in ("FULL_T_on_T", "FULL_W_on_T", "FULL_W_on_W", "FULL_T_on_W")):
                xo = crossover_scores(y_cross["FULL_T_on_T"]["Y"], y_cross["FULL_W_on_T"]["Y"], y_cross["FULL_W_on_W"]["Y"], y_cross["FULL_T_on_W"]["Y"])
                rec_row.update(xo)
                attempts["crossover_complete"] += 1
            rows.append(rec_row)

    write_csv(write_dir / "state_semantics_calibrations.csv", cals or [{"reason": "none"}])
    write_csv(write_dir / "state_semantics_interventions.csv", rows or [{"reason": "none"}])
    write_csv(write_dir / "state_semantics_dose_search.csv", doses or [{"reason": "none"}])
    write_csv(write_dir / "state_semantics_match_controls.csv", matches or [{"reason": "none"}])
    if logit_pack:
        np.savez_compressed(write_dir / "state_semantics_full_logits.npz", **logit_pack)
    if probe_pack:
        np.savez_compressed(write_dir / "state_semantics_probe_probabilities.npz", **probe_pack)
    if dir_pack:
        np.savez_compressed(write_dir / "state_semantics_directions.npz", **dir_pack)
    contrasts = [r for r in rows if r.get("cell") == "pair_contrast" and r.get("valid") == 1]
    slices = {
        "pooled": _slice(contrasts, lambda r: True, n_boot, int(cfg.get("boot_seed", 82))),
        "A": _slice(contrasts, lambda r: r.get("r2_type") == "A_legal_diff", n_boot, 83),
        "B": _slice(contrasts, lambda r: r.get("r2_type") == "B_similar_board_legal_change", n_boot, 84),
        "exact": _slice(contrasts, lambda r: r.get("board_exact_editor") in {1, True, "1", 1.0}, n_boot, 85),
        "nonexact": _slice(contrasts, lambda r: r.get("board_exact_editor") not in {1, True, "1", 1.0}, n_boot, 86),
    }
    pooled = slices.get("pooled") or {}
    board_boot = pooled.get("d_board_inc") or {}
    legal_boot = pooled.get("d_legal_inc") or {}
    gate = {
        "reconstruction_ok": int((out_dir / "state_semantics_stage1_status.json").is_file() and json.loads((out_dir / "state_semantics_stage1_status.json").read_text())["ok"] == 1),
        "split_ok": 1,
        "integrity_ok": int(hooks_ok and attempts["pairs"] == len(pairs)),
        "invariance_ok": int(attempts["invariance_fail"] == 0 and attempts["valid"] > 0),
        "js_match_ok": int(attempts["valid"] > 0),
        "coverage_ok": int(attempts["valid"] >= (1 if smoke else int(cfg.get("coverage_min", COVERAGE_MIN)))),
        "board_increment_ok": increment_ok(board_boot, float(cfg.get("behavior_floor", BEHAVIOR_FLOOR))) if board_boot else 0,
        "legal_increment_ok": increment_ok(legal_boot, float(cfg.get("behavior_floor", BEHAVIOR_FLOOR))) if legal_boot else 0,
        "legal_matches_full": int(board_boot.get("n", 0) > 0 and (abs(board_boot.get("point", 99)) < float(cfg.get("behavior_floor", BEHAVIOR_FLOOR)) or board_boot.get("ci95_low", 1) <= 0)),
    }
    verdict = decide_semantics_verdict(gate)
    boot_rows = []
    for name, sl in slices.items():
        for met, val in sl.items():
            if isinstance(val, dict) and "ci95_low" in val:
                boot_rows.append({"slice": name, "metric": met, **val})
    write_csv(write_dir / "state_semantics_bootstrap.csv", boot_rows or [{"n": 0}])
    write_json(write_dir / "state_semantics_attempt_accounting.json", attempts)
    write_json(write_dir / "state_semantics_dependency_manifest.json", {
        "git_hash": git_hash(), "checkpoint_sha256": sha256_file(stack["ckpt"]),
        "probe_sha256": sha256_file(stack["probe_info"]["probe_path"]),
        "pair_hash": man.get("pair_manifest_sha256"), "used_confirmation": 0,
        "official_0A_letter": OFFICIAL_0A_LETTER, "exploratory": 1,
    })
    write_json(write_dir / "state_semantics_completion_status.json", {
        "status": "SMOKE_COMPLETE" if smoke else "STATE_SEMANTICS_COMPLETE",
        "verdict": "SMOKE_ONLY" if smoke else verdict,
        "gate": gate, "slices": slices, "attempts": attempts, "smoke": int(smoke),
        "used_confirmation": 0, "exploratory": 1, "not_independent_confirmation": 1,
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "nst_status_preserved": "DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED",
        "fixed_js_status_preserved": "FIXED_JS_TARGET_SPECIFICITY_SUPPORTED",
        "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
        "prior_cost_verdict_preserved": "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "section_5_3_status": "PARTIAL", "obstruction_ready": "NO",
    })
    (write_dir / "state_semantics_scientific_report.md").write_text(
        "\n".join([
            "# 08R2-STATE-SEMANTICS Scientific Report",
            "",
            f"Verdict: `{'SMOKE_ONLY' if smoke else verdict}`",
            "Exploratory ablation. Not independent confirmation.",
            f"Attempts: {attempts}",
            f"Gate: {gate}",
            f"Pooled: {pooled}",
            f"A: {slices.get('A')}",
            f"B: {slices.get('B')}",
            f"Exact: {slices.get('exact')}",
            f"Nonexact: {slices.get('nonexact')}",
            "",
            "Residual R² is not a causal verdict. Official NST letter is unchanged.",
            "",
        ]),
        encoding="utf-8",
    )
    print("08R2-STATE-SEMANTICS")
    print(f"  verdict={'SMOKE_ONLY' if smoke else verdict}")
    print(f"  valid_pairs={attempts['valid']}/{len(pairs)} crossover={attempts['crossover_complete']}")
    print(f"  attempts={attempts}")
    print(f"  gate={gate}")
    print(f"  pooled={pooled}")
    print("  DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED preserved")
    print("  official letter unchanged: C / PARTIAL")
    if smoke and attempts["valid"] != 1:
        raise SystemExit(f"{VERDICT_ENGINEERING} smoke failed")


def stage6(cfg, out_dir: Path, nst: Path) -> None:
    man = json.loads((nst / "null_state_pair_manifest.json").read_text(encoding="utf-8"))
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    pair_by = {(p["recip_game"], p["target_game"]): p for p in man["pairs"]}
    inter = list(csv.DictReader((out_dir / "state_semantics_interventions.csv").open(encoding="utf-8")))
    logits = np.load(out_dir / "state_semantics_full_logits.npz")
    n_ok = n_bad = 0
    rows = []
    for row in inter:
        if row.get("cell") in {"pair_contrast"} or not saved_y_present(row):
            continue
        rg, tg, cell = row["recip_game"], row["target_game"], row["cell"]
        if (rg, tg) not in pair_by:
            continue
        rec = resolve_pair_rows(val, pair_by[(rg, tg)])["rec"]
        tgt = resolve_pair_rows(val, pair_by[(rg, tg)])["tgt"]
        lkey = f"{rg}__{tg}__{cell}"
        if lkey not in logits.files:
            continue
        recon = rescore_y_from_logits(logits[lkey], rec["legal"], tgt["legal"], probs=softmax_np(logits[lkey]))
        chk = audit_saved_y(row, recon, float(cfg.get("y_recon_atol", Y_RECON_ATOL)))
        rows.append({"cell": cell, "ok": chk["ok"], "abs_Y": chk.get("abs_Y")})
        n_ok += int(chk["ok"] == 1)
        n_bad += int(chk["ok"] == 0)
    status = {"ok": int(n_bad == 0 and n_ok > 0), "n_match": n_ok, "n_mismatch": n_bad}
    write_csv(out_dir / "state_semantics_stage6_rescore.csv", rows or [{"reason": "none"}])
    write_json(out_dir / "state_semantics_stage6_status.json", status)
    print("08R2-STATE-SEMANTICS STAGE 6")
    print(f"  ok={status['ok']} match={n_ok} mismatch={n_bad}")
    if not status["ok"]:
        raise SystemExit(STATUS_STAGE0_FAIL)


def _write_protocol_copy(cfg, out_dir: Path) -> None:
    prereg = Path(cfg.get("preregistration", "docs/experiments/08r2_state_semantics_preregistration.md"))
    if prereg.is_file():
        (out_dir / "state_semantics_preregistration.md").write_text(prereg.read_text(encoding="utf-8"), encoding="utf-8")


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
    out_dir = ensure_dir(isolate_sem_dir(cfg["results_dir"])["results_dir"])
    _write_protocol_copy(cfg, out_dir)
    nst = _resolve_nst_dir(cfg, args, out_dir)
    if not nst.is_dir():
        raise SystemExit(f"{VERDICT_ENGINEERING} missing NST archive at {nst}")
    stages = ["0", "1", "2", "3", "4", "5", "6"] if args.stage == "all" else [args.stage]
    man = val = None
    if "0" in stages:
        s0 = stage0(cfg, out_dir, nst)
        man, val = s0["man"], s0["val"]
    if man is None:
        man = json.loads((nst / "null_state_pair_manifest.json").read_text(encoding="utf-8"))
        val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    if "1" in stages:
        stage1(cfg, out_dir, args, nst, man, val)
    if "2" in stages:
        if not (out_dir / "state_semantics_stage1_status.json").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 1 missing")
        if not json.loads((out_dir / "state_semantics_stage1_status.json").read_text())["ok"]:
            raise SystemExit(STATUS_STAGE1_FAIL)
        stage2(cfg, out_dir, args, nst)
    if "3" in stages:
        stage3(cfg, out_dir, nst)
    if "4" in stages:
        if not (out_dir / "state_semantics_board_only.npz").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 2 models missing")
        stage_eval(cfg, out_dir, args, nst, smoke=True)
    if "5" in stages:
        if not (out_dir / "state_semantics_board_only.npz").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 2 models missing")
        stage_eval(cfg, out_dir, args, nst, smoke=False)
    if "6" in stages:
        if not (out_dir / "state_semantics_full_logits.npz").is_file():
            raise SystemExit(f"{VERDICT_ENGINEERING} stage 5 logits missing")
        stage6(cfg, out_dir, nst)


if __name__ == "__main__":
    main()
