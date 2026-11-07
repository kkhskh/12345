"""08R2-FIBER real-model diagnostic.

Does not train a model or probe. Does not optimize a new editor.
Does not evaluate the sealed confirmation split.
Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from run_08r1_real_model_verification import git_hash, patch_and_audit, resolve_checkpoint
from src.othello_r1_eval import resolve_probe_path, sha256_file
from src.othello_r2 import (
    OFFICIAL_0A_LETTER,
    PROTOCOL_ABD,
    SITE,
    LinearBoardDecoder,
    causal_scores,
    decoder_state_effects,
    isolate_r2_results_dir,
    iterative_state_edit,
    load_holdout_game_ids,
    r2_pair_categories,
    require_no_confirmation,
    torch_parity_report,
    train_activation_stats,
)
from src.othello_r2_fiber import (
    DEFAULT_TOLS,
    PRIMARY,
    REQUIRED_FIBER_FILES,
    STATUS_COMPLETE,
    STATUS_ENGINEERING,
    STATUS_REAL_REQUIRED,
    STATUS_VALIDATION_FAILED,
    VERDICT_ENGINEERING,
    VERDICT_INCONCLUSIVE,
    causal_from_base,
    contrast_parity_report,
    decide_verdict,
    evaluate_fiber_completeness,
    factorial_activations,
    factorial_effects,
    historical_editor_a_row,
    independent_contrasts,
    isolate_fiber_results_dir,
    js_divergence,
    load_validation_primary_pairs,
    pair_demonstrates_insufficiency,
    probe_invariance,
    random_nullspace_direction,
    reject_unexpected_preprocessing,
    reproduction_ok,
    require_no_confirm_pairs,
    slice_boot,
    softmax,
    svd_probe_geometry,
    training_distance,
    unit_null,
    verify_pairs_against_sampler,
    write_scientific_report,
    y_functional,
)
from src.othello_runutil import add_labels, load_split, tokens_of, write_json
from src.toy_othello import (
    cache_site,
    load_config,
    load_othello_gpt,
    run_model_logits,
    softmax_np,
    square_to_token,
    verify_nanda_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-FIBER probe fiber diagnostic.")
    p.add_argument("--config", default="configs/toy_othello_08r2_fiber.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--validation-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--n-boot", type=int, default=None)
    return p.parse_args()


def _require_real_stack():
    try:
        import torch
        import transformer_lens  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            f"{STATUS_REAL_REQUIRED}\nReal stack missing ({exc}). Fiber does not fall back to a fake model."
        )
    return torch


def merge_cfg(*parts: dict) -> dict:
    out = {}
    for p in parts:
        out.update(p)
    return out


def _snapshot_params(model) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def _params_unchanged(before, model) -> bool:
    after = _snapshot_params(model)
    return all(np.allclose(a, b) for a, b in zip(before, after))


def _tols(cfg: dict) -> dict:
    out = dict(DEFAULT_TOLS)
    for k in out:
        if k in cfg:
            out[k] = cfg[k]
    out["rank_tolerances"] = [float(x) for x in cfg.get("rank_tolerances", out["rank_tolerances"])]
    return out


def _resolve_validation_dir(cfg: dict, explicit: str | None) -> Path:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    cands.append(Path("/content/drive/MyDrive/ml_research_artifacts/toy_othello/results/08r2/validation"))
    cands.append(Path(cfg["results_dir"]) / "08r2" / "validation")
    isolated = isolate_r2_results_dir(cfg["results_dir"], validation=True)
    cands.append(Path(isolated["results_dir"]))
    for p in cands:
        if (p / "08r2_validation_pairs.json").is_file():
            return p
    return cands[0]


def behavior_gradient(model, tokens, site, h, rec_legal, tgt_legal):
    """Mechanistic sensitivity diagnostic. Not a donor-free state editor."""
    import torch
    from src.toy_othello import pad_batch

    device = next(model.parameters()).device
    h0 = torch.as_tensor(h, dtype=torch.float32, device=device).clone().requires_grad_(True)
    toks, lengths = pad_batch([tokens], device)
    last = int(lengths[0]) - 1

    def patch_fn(act, hook, val=h0, pos=last):
        out = act.clone()
        out[0, pos, :] = val.to(dtype=out.dtype)
        return out

    logits = model.run_with_hooks(toks, fwd_hooks=[(site, patch_fn)])[0, last]
    probs = torch.softmax(logits, dim=-1)
    became = [square_to_token(s) for s in set(tgt_legal) - set(rec_legal) if square_to_token(s) is not None]
    left = [square_to_token(s) for s in set(rec_legal) - set(tgt_legal) if square_to_token(s) is not None]
    y = probs.new_zeros(())
    if became:
        y = y + probs[became].sum()
    if left:
        y = y - probs[left].sum()
    y.backward()
    if h0.grad is None or not torch.isfinite(h0.grad).all():
        return None, float("nan")
    return h0.grad.detach().cpu().numpy().astype(np.float64), float(y.detach().cpu())


def finite_difference_y(model, tokens, site, h, rec_legal, tgt_legal, direction, eps: float):
    import torch
    from src.toy_othello import pad_batch

    device = next(model.parameters()).device
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    toks, lengths = pad_batch([tokens], device)
    last = int(lengths[0]) - 1
    ys = []
    for sign in (1.0, -1.0):
        val = torch.as_tensor(h + sign * eps * direction, dtype=torch.float32, device=device)

        def patch_fn(act, hook, v=val, pos=last):
            out = act.clone()
            out[0, pos, :] = v.to(dtype=out.dtype)
            return out

        with torch.no_grad():
            logits = model.run_with_hooks(toks, fwd_hooks=[(site, patch_fn)])[0, last].detach().cpu().numpy()
        ys.append(y_functional(softmax(logits), rec_legal, tgt_legal)["Y"])
    return float((ys[0] - ys[1]) / (2.0 * eps))


def _patch(model, tokens, h):
    import torch

    vals = torch.tensor(np.asarray(h, dtype=np.float32).reshape(1, -1))
    logits, readback, audit = patch_and_audit(model, [tokens], SITE, vals, batch_size=1)
    return softmax_np(logits.numpy()[0]), logits.numpy()[0], audit, readback


def _cell_row(name, pair, rec, tgt, h, h_e, h_d, decoder, base_probs, audit, logits, stats, tols, extra=None):
    probs = softmax(logits)
    causal = causal_from_base(base_probs, probs, rec["legal"], tgt["legal"])
    dec = decoder_state_effects(decoder, h, rec["rel_lab"], tgt["rel_lab"])
    proba = decoder.proba_np(h)
    dist = training_distance(h, stats["mean"], stats["pca_basis"]) if stats else {}
    row = {
        "cell": name,
        "method": name,
        "r2_type": pair["r2_type"],
        "recip": pair["recip"],
        "donor": pair["donor"],
        "recip_game": pair["recip_game"],
        "target_game": pair["target_game"],
        "player": int(rec["player"]),
        "token_index": int(len(rec["tokens"]) - 1),
        "site": SITE,
        "prefix_len_recip": int(len(rec["tokens"])),
        "prefix_len_donor": int(len(tgt["tokens"])),
        "occupied_recip": int(rec.get("occupied", -1)),
        "occupied_donor": int(tgt.get("occupied", -1)),
        "hamming": pair.get("hamming"),
        "n_became_legal": pair.get("n_became_legal"),
        "n_became_illegal": pair.get("n_became_illegal"),
        "board_exact": dec["board_exact"],
        "changed_acc": dec["changed_acc"],
        "unchanged_acc": dec["unchanged_acc"],
        "square_acc": dec["square_acc"],
        "Y": causal["Y"],
        "became_legal_mass": causal["became_legal_mass"],
        "became_illegal_mass": causal["became_illegal_mass"],
        "stayed_legal_mass": causal["stayed_legal_mass"],
        "stayed_illegal_mass": causal["stayed_illegal_mass"],
        "dP_gain": causal["became_legal_d_mass"],
        "dP_loss": causal["became_illegal_d_mass"],
        "kl_to_base": causal["kl_to_base"],
        "l2_to_editor_A": float(np.linalg.norm(np.asarray(h) - np.asarray(h_e))),
        "l2_to_donor": float(np.linalg.norm(np.asarray(h) - np.asarray(h_d))),
        "hook_ok": int(audit.get("ok", 0)),
        "valid_edit": 1,
        **dist,
    }
    if extra:
        row.update(extra)
    return row, proba.reshape(64, 3), np.asarray(logits, dtype=np.float64)


def _summarize(rows: list[dict], r2: str | None, n_boot: int, seed: int) -> dict:
    g = [r for r in rows if r.get("valid_fiber") and (r2 is None or r.get("r2_type") == r2)]
    name = "pooled" if r2 is None else ("A" if r2 == "A_legal_diff" else "B")
    if not g:
        return {"slice": name, "n_valid": 0, "n_demo": 0, "n_target_games": 0}
    def col(key):
        return [r[key] for r in g if r.get(key) == r.get(key)]
    clusters = [r["recip_game"] for r in g]
    out = {
        "slice": name,
        "n_valid": int(len(g)),
        "n_demo": int(sum(bool(r.get("demonstrates_insufficiency")) for r in g)),
        "n_recipient_games": int(len({r["recip_game"] for r in g})),
        "n_target_games": int(len({r["target_game"] for r in g})),
        "mean_E_R": float(np.nanmean(col("E_R"))),
        "median_E_R": float(np.nanmedian(col("E_R"))),
        "mean_E_N": float(np.nanmean(col("E_N"))),
        "median_E_N": float(np.nanmedian(col("E_N"))),
        "mean_abs_E_N": float(np.nanmean(np.abs(col("E_N")))) if col("E_N") else float("nan"),
        "mean_I": float(np.nanmean(col("I"))),
        "median_I": float(np.nanmedian(col("I"))),
        "mean_E_R_given_N": float(np.nanmean(col("E_R_given_N"))),
        "mean_E_N_given_R": float(np.nanmean(col("E_N_given_R"))),
        "mean_js_00_01": float(np.nanmean(col("js_00_01"))),
        "mean_js_10_11": float(np.nanmean(col("js_10_11"))),
        "mean_js_00_10": float(np.nanmean(col("js_00_10"))),
        "mean_js_00_11": float(np.nanmean(col("js_00_11"))),
        "mean_l2_N": float(np.nanmean(col("l2_N"))),
        "mean_l2_R": float(np.nanmean(col("l2_R"))),
        "mean_invariance_err": float(np.nanmean(col("invariance_max_err"))),
    }
    if col("E_R"):
        out["E_R_ci"] = slice_boot(col("E_R"), clusters, n_boot, seed)
        out["E_N_ci"] = slice_boot(col("E_N"), clusters, n_boot, seed + 1)
        out["I_ci"] = slice_boot(col("I"), clusters, n_boot, seed + 2)
        out["js_00_01_ci"] = slice_boot(col("js_00_01"), clusters, n_boot, seed + 3)
    return out


def write_engineering_status(out_dir: Path, reason: str, extra: dict | None = None) -> None:
    payload = {
        "status": STATUS_REAL_REQUIRED,
        "verdict": VERDICT_ENGINEERING,
        "reason": reason,
        "checkpoint_loaded": 0,
        "probe_loaded": 0,
        "synthetic_scientific_rows": 0,
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "section_5_3_status": "PARTIAL",
        "obstruction_ready": "NO",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "used_confirmation": 0,
    }
    if extra:
        payload.update(extra)
    write_json(out_dir / "08r2_fiber_completion_status.json", payload)
    print(STATUS_REAL_REQUIRED)
    print(reason)


def main() -> None:
    args = parse_args()
    parent = load_config(args.parent_config)
    r2cfg = load_config(args.r2_config)
    child = load_config(args.config)
    cfg = merge_cfg(parent, r2cfg, child)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.checkpoints_dir:
        cfg["checkpoints_dir"] = args.checkpoints_dir
    tols = _tols(cfg)
    isolated = isolate_fiber_results_dir(cfg["results_dir"])
    out_dir = ensure_dir(isolated["results_dir"])

    try:
        torch = _require_real_stack()
    except SystemExit as exc:
        write_engineering_status(out_dir, str(exc))
        raise

    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    if not holdout["hash_ok"]:
        raise SystemExit("FIBER_VALIDATION_FAILED confirmation manifest payload hash mismatch")

    probe_info = resolve_probe_path(args.probe_path, cfg["checkpoints_dir"])
    probe_path = Path(probe_info["probe_path"])
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if not probe_info["exists"] or ckpt is None or not Path(ckpt).is_file():
        write_engineering_status(
            out_dir,
            "BLOCKED_MISSING_ARTIFACT checkpoint or probe",
            {"probe_exists": int(probe_info["exists"]), "checkpoint_exists": int(ckpt is not None and Path(ckpt).is_file())},
        )
        raise SystemExit(STATUS_REAL_REQUIRED)

    val_dir = _resolve_validation_dir(cfg, args.validation_dir)
    pairs_path = val_dir / "08r2_validation_pairs.json"
    if not pairs_path.is_file():
        write_engineering_status(out_dir, f"missing frozen pairs {pairs_path}")
        raise SystemExit(STATUS_REAL_REQUIRED)

    data_dir = Path(cfg["data_dir"])
    train = add_labels(load_split(data_dir, "train"))
    val = add_labels(load_split(data_dir, "val"))
    require_no_confirmation(train, holdout["game_ids"])
    require_no_confirmation(val, holdout["game_ids"])

    saved_pairs, pair_flags = load_validation_primary_pairs(pairs_path)
    require_no_confirm_pairs(saved_pairs, holdout["game_ids"])
    recomputed = r2_pair_categories(
        val,
        32,
        int(cfg["pair_seed"]),
        holdout["game_ids"],
        n_a=int(PROTOCOL_ABD["n_a"]),
        n_b=int(PROTOCOL_ABD["n_b"]),
        include_c=False,
    )
    pair_check = verify_pairs_against_sampler(saved_pairs, recomputed)
    if not pair_check["ok"]:
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} pair hash mismatch {pair_check}")
    if pair_flags["n_a"] != 24 or pair_flags["n_b"] != 8:
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} expected 24A/8B, got {pair_flags}")

    causal_hist = []
    hist_csv = val_dir / "08r2_validation_causal_effects.csv"
    if hist_csv.is_file():
        causal_hist = pd.read_csv(hist_csv).to_dict("records")

    with probe_path.open("rb") as handle:
        pack = pickle.load(handle)
    prep = reject_unexpected_preprocessing(pack)
    if not prep["ok"]:
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} unexpected probe preprocessing {prep}")
    if str(pack.get("hook")) != SITE:
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} probe site {pack.get('hook')} != {SITE}")
    decoder = LinearBoardDecoder(pack)
    contrasts = independent_contrasts(decoder)
    dec_t = decoder.as_torch()

    model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=str(ckpt))
    ident = verify_nanda_checkpoint(model)
    if not ident.get("nanda_argmax_match"):
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} checkpoint identity {ident}")
    params_before = _snapshot_params(model)
    for p in model.parameters():
        p.requires_grad_(False)
    probe_coef_before = decoder.coef.copy()

    needed = sorted({int(p["recip"]) for p in saved_pairs} | {int(p["donor"]) for p in saved_pairs})
    val_tokens = tokens_of(val)
    H = cache_site(model, val_tokens, SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    base_logits = run_model_logits(model, val_tokens, batch_size=int(cfg.get("batch_size", 64))).numpy()
    base_probs = np.stack([softmax_np(row) for row in base_logits], axis=0)

    parity_idx = needed[: min(16, len(needed))]
    torch_parity = torch_parity_report(decoder, pack["models"], H[parity_idx])
    contrast_parity = contrast_parity_report(decoder, pack["models"], H[parity_idx], contrasts, float(tols["parity_proba_atol"]))
    if torch_parity["max_abs_proba"] > float(tols["parity_proba_atol"]) or contrast_parity["ok"] != 1:
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} probe reconstruction parity {contrast_parity} {torch_parity}")

    svd = svd_probe_geometry(contrasts["W"], float(tols["primary_rank_tol"]), tols["rank_tolerances"])
    if not svd.get("has_meaningful_nullspace"):
        write_json(out_dir / "08r2_fiber_svd_report.json", {k: svd[k] for k in svd if k not in {"V_r", "P_R", "P_N"}})
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} probe has no meaningful numerical nullspace; rank={svd['numerical_rank']}")

    p_r = svd["P_R"]
    p_n = svd["P_N"]
    train_sub = train[: int(cfg.get("n_train_stat_rows", 256))]
    Xtr = cache_site(model, tokens_of(train_sub), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    stats = train_activation_stats(Xtr, rank=int(cfg.get("pca_rank", 32)))

    budget = float(cfg.get("editor_budget", 16.0))
    steps = int(cfg.get("edit_steps", 40))
    lr = float(cfg.get("edit_lr", 0.4))
    n_boot = int(args.n_boot or cfg.get("n_boot", 2000))
    if n_boot < 2000:
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} n_boot>=2000 required, got {n_boot}")

    matched_rows = []
    factorial_rows = []
    control_rows = []
    anchor_rows = []
    boot_rows = []
    logit_pack = {}
    proba_pack = {}
    hooks_ok = 1
    n_invalid = 0
    n_demo = 0
    effects_rows = []
    grad_rows = []

    wrong_pool = list(saved_pairs)
    for pair in saved_pairs:
        rec = val[int(pair["recip"])]
        tgt = val[int(pair["donor"])]
        h_rec = H[int(pair["recip"])]
        h_d = np.asarray(H[int(pair["donor"])], dtype=np.float64)
        pb = base_probs[int(pair["recip"])]
        hist = historical_editor_a_row(causal_hist, pair, budget)
        ed = iterative_state_edit(
            h_rec,
            tgt["rel_lab"],
            rec["rel_lab"],
            dec_t,
            steps=steps,
            lr=lr,
            budget=budget,
            lambda_norm=float(cfg["lambda_norm"]),
            eta_preserve=float(cfg["eta_preserve"]),
            mu_manifold=float(cfg.get("mu_manifold", 0.0)),
        )
        if not ed.get("valid_edit") or ed.get("edited") is None:
            n_invalid += 1
            matched_rows.append(
                {
                    "recip_game": pair["recip_game"],
                    "target_game": pair["target_game"],
                    "r2_type": pair["r2_type"],
                    "valid_fiber": 0,
                    "reason": ed.get("reason", "editor_A_reproduction_failed"),
                }
            )
            continue
        h_e = np.asarray(ed["edited"], dtype=np.float64)
        dec_e = decoder_state_effects(decoder, h_e, rec["rel_lab"], tgt["rel_lab"])
        dec_d = decoder_state_effects(decoder, h_d, rec["rel_lab"], tgt["rel_lab"])
        pe_e, log_e, aud_e, _ = _patch(model, rec["tokens"], h_e)
        hooks_ok = hooks_ok and int(aud_e.get("ok", 0))
        scores_e = causal_scores(pb, pe_e, rec["legal"], tgt["legal"])
        repro = reproduction_ok(
            {
                "board_exact": dec_e["board_exact"],
                "dP_gain": scores_e["dP_gain"],
                "dP_loss": scores_e["dP_loss"],
                "delta_norm": float(np.linalg.norm(h_e - h_rec)),
            },
            hist,
            tols,
        )
        if dec_e["board_exact"] != 1.0 or dec_d["board_exact"] != 1.0:
            n_invalid += 1
            matched_rows.append(
                {
                    "recip_game": pair["recip_game"],
                    "target_game": pair["target_game"],
                    "r2_type": pair["r2_type"],
                    "valid_fiber": 0,
                    "reason": "decoded_board_not_exact_target",
                    "board_exact_E": dec_e["board_exact"],
                    "board_exact_D": dec_d["board_exact"],
                    **repro,
                }
            )
            continue
        if hist is not None and not repro["ok"]:
            n_invalid += 1
            matched_rows.append(
                {
                    "recip_game": pair["recip_game"],
                    "target_game": pair["target_game"],
                    "r2_type": pair["r2_type"],
                    "valid_fiber": 0,
                    "reason": "reproduction_outside_tolerance",
                    **repro,
                }
            )
            continue

        fact = factorial_activations(h_e, h_d, p_r, p_n)
        inv64_01 = probe_invariance(decoder, fact["cells"]["H01"], fact["cells"]["H00"], float(tols["probe_proba_atol_f64"]))
        inv64_10 = probe_invariance(decoder, fact["cells"]["H10"], fact["cells"]["H11"], float(tols["probe_proba_atol_f64"]))
        inv32_01 = probe_invariance(
            decoder,
            np.asarray(fact["cells"]["H01"], dtype=np.float32),
            np.asarray(fact["cells"]["H00"], dtype=np.float32),
            float(tols["probe_proba_atol_f32"]),
        )
        inv32_10 = probe_invariance(
            decoder,
            np.asarray(fact["cells"]["H10"], dtype=np.float32),
            np.asarray(fact["cells"]["H11"], dtype=np.float32),
            float(tols["probe_proba_atol_f32"]),
        )
        inv_ok = bool(inv64_01["ok"] and inv64_10["ok"] and inv32_01["ok"] and inv32_10["ok"])
        inv_err = max(inv64_01["max_abs_proba"], inv64_10["max_abs_proba"], inv32_01["max_abs_proba"], inv32_10["max_abs_proba"])
        if not inv_ok:
            n_invalid += 1

        cell_probs = {}
        cell_logits = {}
        cell_y = {}
        for name, h in fact["cells"].items():
            probs, logits, audit, _ = _patch(model, rec["tokens"], h)
            hooks_ok = hooks_ok and int(audit.get("ok", 0))
            row, proba, logit = _cell_row(name, pair, rec, tgt, h, h_e, h_d, decoder, pb, audit, logits, stats, tols)
            row["valid_fiber"] = int(inv_ok)
            row["invariance_ok"] = int(inv_ok)
            row["invariance_max_err"] = inv_err
            factorial_rows.append(row)
            cell_probs[name] = probs
            cell_logits[name] = logit
            cell_y[name] = row["Y"]
            key = f"{pair['recip_game']}|{pair['target_game']}|{name}"
            logit_pack[key] = logit
            proba_pack[key] = proba

        fx = factorial_effects(cell_y["H00"], cell_y["H10"], cell_y["H01"], cell_y["H11"])
        js = {
            "js_00_01": js_divergence(cell_probs["H00"], cell_probs["H01"]),
            "js_10_11": js_divergence(cell_probs["H10"], cell_probs["H11"]),
            "js_00_10": js_divergence(cell_probs["H00"], cell_probs["H10"]),
            "js_00_11": js_divergence(cell_probs["H00"], cell_probs["H11"]),
            "kl_00_01": kl_safe(cell_probs["H00"], cell_probs["H01"]),
            "kl_10_11": kl_safe(cell_probs["H10"], cell_probs["H11"]),
        }
        demo = int(
            inv_ok
            and pair_demonstrates_insufficiency(
                js["js_00_01"],
                js["js_10_11"],
                fx["E_N"],
                float(tols["behavior_floor"]),
                float(tols["js_meaningful"]),
            )
        )
        n_demo += demo
        effect = {
            "r2_type": pair["r2_type"],
            "recip_game": pair["recip_game"],
            "target_game": pair["target_game"],
            "recip": pair["recip"],
            "donor": pair["donor"],
            "valid_fiber": int(inv_ok),
            "demonstrates_insufficiency": demo,
            "invariance_max_err": inv_err,
            "l2_R": fact["l2_R"],
            "l2_N": fact["l2_N"],
            "l2_delta": fact["l2"],
            "h11_minus_donor": fact["h11_minus_donor"],
            "reproduction_ok": repro["ok"],
            "board_exact_E": dec_e["board_exact"],
            "board_exact_D": dec_d["board_exact"],
            "prefix_len_diff": int(len(tgt["tokens"]) - len(rec["tokens"])),
            "occupied_diff": int(tgt.get("occupied", 0) - rec.get("occupied", 0)),
            "hamming": pair.get("hamming"),
            **fx,
            **js,
            **repro,
        }
        effects_rows.append(effect)
        matched_rows.append(effect)

        # No-op and full donor already implied; still record explicit no-op on recipient activation.
        p_noop, log_noop, aud_noop, _ = _patch(model, rec["tokens"], h_rec)
        hooks_ok = hooks_ok and int(aud_noop.get("ok", 0))
        noop_row, _, _ = _cell_row("noop", pair, rec, tgt, h_rec, h_e, h_d, decoder, pb, aud_noop, log_noop, stats, tols)
        noop_row["js_to_base"] = js_divergence(pb, p_noop)
        control_rows.append(noop_row)

        # Random nullspace matched to ||Δ_N||, added to h_E.
        nvec = unit_null(fact["delta_N"], float(tols["null_component_min_norm"]))
        if nvec is not None and fact["l2_N"] >= float(tols["null_component_min_norm"]):
            rng_dir = random_nullspace_direction(p_n, int(cfg["random_seeds"][0]) + int(pair["recip"]))
            h_rand = h_e + fact["l2_N"] * rng_dir
            p_rnull, log_rnull, aud_r, _ = _patch(model, rec["tokens"], h_rand)
            hooks_ok = hooks_ok and int(aud_r.get("ok", 0))
            inv_r = probe_invariance(decoder, h_rand, h_e, float(tols["probe_proba_atol_f32"]))
            rr, _, _ = _cell_row(
                "random_nullspace",
                pair,
                rec,
                tgt,
                h_rand,
                h_e,
                h_d,
                decoder,
                pb,
                aud_r,
                log_rnull,
                stats,
                tols,
                extra={"invariance_ok": inv_r["ok"], "matched_l2": fact["l2_N"], "dY_vs_H00": y_functional(p_rnull, rec["legal"], tgt["legal"])["Y"] - cell_y["H00"]},
            )
            control_rows.append(rr)

            # Rowspace random of comparable ||Δ_R||
            rng_r = np.random.default_rng(int(cfg["random_seeds"][1]) + int(pair["recip"]))
            raw_r = p_r @ rng_r.normal(size=p_r.shape[0])
            nrm_r = float(np.linalg.norm(raw_r))
            if nrm_r > 1e-15 and fact["l2_R"] > 0:
                h_row = h_e + fact["l2_R"] * (raw_r / nrm_r)
                p_row, log_row, aud_row, _ = _patch(model, rec["tokens"], h_row)
                hooks_ok = hooks_ok and int(aud_row.get("ok", 0))
                rrow, _, _ = _cell_row("random_rowspace", pair, rec, tgt, h_row, h_e, h_d, decoder, pb, aud_row, log_row, stats, tols)
                control_rows.append(rrow)

            # Wrong-donor nullspace, stage-matched.
            others = [
                q
                for q in wrong_pool
                if q["target_game"] != pair["target_game"] and q["recip_game"] != pair["recip_game"]
            ]
            if others:
                others = sorted(others, key=lambda q: abs(int(val[int(q["donor"])].get("occupied", 0)) - int(tgt.get("occupied", 0))))
                q = others[0]
                h_wrong = np.asarray(H[int(q["donor"])], dtype=np.float64)
                wparts = factorial_activations(h_e, h_wrong, p_r, p_n)
                h_wn = h_e + wparts["delta_N"]
                p_w, log_w, aud_w, _ = _patch(model, rec["tokens"], h_wn)
                hooks_ok = hooks_ok and int(aud_w.get("ok", 0))
                inv_w = probe_invariance(decoder, h_wn, h_e, float(tols["probe_proba_atol_f32"]))
                wr, _, _ = _cell_row(
                    "wrong_donor_nullspace",
                    pair,
                    rec,
                    tgt,
                    h_wn,
                    h_e,
                    h_d,
                    decoder,
                    pb,
                    aud_w,
                    log_w,
                    stats,
                    tols,
                    extra={
                        "wrong_target_game": q["target_game"],
                        "invariance_ok": inv_w["ok"],
                        "l2_applied": wparts["l2_N"],
                        "dY_vs_H00": y_functional(p_w, rec["legal"], tgt["legal"])["Y"] - cell_y["H00"],
                    },
                )
                control_rows.append(wr)

            # Natural-anchor perturbations around h_D.
            for eps in [float(e) for e in cfg.get("anchor_eps", [1.0, 2.0, 4.0])]:
                for sign, tag in ((1.0, "plus"), (-1.0, "minus")):
                    h_a = h_d + sign * eps * nvec
                    p_a, log_a, aud_a, _ = _patch(model, rec["tokens"], h_a)
                    hooks_ok = hooks_ok and int(aud_a.get("ok", 0))
                    inv_a = probe_invariance(decoder, h_a, h_d, float(tols["probe_proba_atol_f32"]))
                    ar, _, _ = _cell_row(
                        f"natural_anchor_{tag}",
                        pair,
                        rec,
                        tgt,
                        h_a,
                        h_e,
                        h_d,
                        decoder,
                        pb,
                        aud_a,
                        log_a,
                        stats,
                        tols,
                        extra={
                            "eps": eps,
                            "sign": sign,
                            "family": "donor_null_direction",
                            "invariance_ok": inv_a["ok"],
                            "invariance_err": inv_a["max_abs_proba"],
                            "dY_vs_donor": y_functional(p_a, rec["legal"], tgt["legal"])["Y"] - cell_y["H11"],
                            "js_vs_donor": js_divergence(cell_probs["H11"], p_a),
                        },
                    )
                    anchor_rows.append(ar)
                for draw, seed in enumerate(cfg.get("random_seeds", [1082, 2082, 3082])):
                    rnd = random_nullspace_direction(p_n, int(seed) + 17 * int(pair["recip"]) + draw)
                    for sign, tag in ((1.0, "plus"), (-1.0, "minus")):
                        h_ra = h_d + sign * eps * rnd
                        p_ra, log_ra, aud_ra, _ = _patch(model, rec["tokens"], h_ra)
                        hooks_ok = hooks_ok and int(aud_ra.get("ok", 0))
                        inv_ra = probe_invariance(decoder, h_ra, h_d, float(tols["probe_proba_atol_f32"]))
                        rar, _, _ = _cell_row(
                            f"random_anchor_{tag}",
                            pair,
                            rec,
                            tgt,
                            h_ra,
                            h_e,
                            h_d,
                            decoder,
                            pb,
                            aud_ra,
                            log_ra,
                            stats,
                            tols,
                            extra={
                                "eps": eps,
                                "sign": sign,
                                "draw": draw,
                                "family": "random_nullspace",
                                "invariance_ok": inv_ra["ok"],
                                "invariance_err": inv_ra["max_abs_proba"],
                                "dY_vs_donor": y_functional(p_ra, rec["legal"], tgt["legal"])["Y"] - cell_y["H11"],
                                "js_vs_donor": js_divergence(cell_probs["H11"], p_ra),
                            },
                        )
                        anchor_rows.append(rar)

        if bool(cfg.get("do_gradient_diagnostic", True)) and inv_ok:
            g, y_d = behavior_gradient(model, rec["tokens"], SITE, h_d, rec["legal"], tgt["legal"])
            if g is not None:
                gn = float(np.linalg.norm(g))
                g_n = p_n @ g
                frac = float(np.linalg.norm(g_n) ** 2 / (gn ** 2)) if gn > 0 else float("nan")
                grad_rows.append(
                    {
                        "recip_game": pair["recip_game"],
                        "target_game": pair["target_game"],
                        "r2_type": pair["r2_type"],
                        "Y_at_donor": y_d,
                        "grad_norm": gn,
                        "nullspace_grad_energy_frac": frac,
                        "note": "mechanistic_sensitivity_not_an_editor",
                    }
                )

    if not _params_unchanged(params_before, model) or not np.allclose(probe_coef_before, decoder.coef):
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} model or probe parameters changed")

    valid_effects = [r for r in effects_rows if r.get("valid_fiber")]
    slices = {
        "A": _summarize(valid_effects, "A_legal_diff", n_boot, int(cfg["boot_seed"])),
        "B": _summarize(valid_effects, "B_similar_board_legal_change", n_boot, int(cfg["boot_seed"])),
        "pooled": _summarize(valid_effects, None, n_boot, int(cfg["boot_seed"])),
    }
    for sl in slices.values():
        if sl.get("E_R_ci"):
            boot_rows.append({"slice": sl["slice"], "metric": "E_R", **sl["E_R_ci"]})
            boot_rows.append({"slice": sl["slice"], "metric": "E_N", **sl["E_N_ci"]})
            boot_rows.append({"slice": sl["slice"], "metric": "I", **sl["I_ci"]})
            boot_rows.append({"slice": sl["slice"], "metric": "js_00_01", **sl["js_00_01_ci"]})

    anchor_eps1 = [r["dY_vs_donor"] for r in anchor_rows if r.get("family") == "donor_null_direction" and r.get("eps") == 1.0]
    rand_eps1 = [r["dY_vs_donor"] for r in anchor_rows if r.get("family") == "random_nullspace" and r.get("eps") == 1.0]
    wrong = [r.get("dY_vs_H00") for r in control_rows if r.get("cell") == "wrong_donor_nullspace"]
    n_anchor_fail = int(sum(r.get("invariance_ok") == 0 for r in anchor_rows))

    n_valid = int(len(valid_effects))
    verdict = decide_verdict(n_valid, n_demo, len(saved_pairs), int(svd["nullspace_dim"]), contrast_parity["ok"] == 1, bool(hooks_ok))
    status = STATUS_COMPLETE if n_valid and hooks_ok and contrast_parity["ok"] == 1 else STATUS_VALIDATION_FAILED
    if n_valid == 0:
        status = STATUS_VALIDATION_FAILED
        verdict = VERDICT_INCONCLUSIVE if contrast_parity["ok"] == 1 else VERDICT_ENGINEERING

    def _csv(name: str, rows: list[dict], stub: str) -> None:
        write_csv(out_dir / name, rows or [{"reason": stub, "n": 0}])

    _csv("08r2_fiber_probe_contrast_parity.csv", [contrast_parity, {"backend": "torch", **{k: torch_parity[k] for k in torch_parity if k != "class_sets"}}], "parity_missing")
    write_json(out_dir / "08r2_fiber_svd_report.json", {k: svd[k] for k in svd if k not in {"V_r", "P_R", "P_N"}})
    _csv("08r2_fiber_matched_pairs.csv", matched_rows, "no_matched_pairs")
    _csv("08r2_fiber_factorial_interventions.csv", factorial_rows, "no_factorial_cells")
    _csv("08r2_fiber_natural_anchor.csv", anchor_rows, "no_null_component_above_min_norm")
    _csv("08r2_fiber_controls.csv", control_rows + grad_rows, "no_controls")
    _csv("08r2_fiber_bootstrap.csv", boot_rows, "no_valid_pairs_for_bootstrap")
    np.savez_compressed(out_dir / "08r2_fiber_full_logits.npz", **{k.replace("|", "__"): v for k, v in logit_pack.items()})
    np.savez_compressed(out_dir / "08r2_fiber_probe_probabilities.npz", **{k.replace("|", "__"): v for k, v in proba_pack.items()})

    report_payload = {
        "status": status,
        "verdict": verdict,
        "tolerances": tols,
        "svd": {k: svd[k] for k in svd if k not in {"V_r", "P_R", "P_N"}},
        "n_matched": int(len(saved_pairs)),
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_demo": n_demo,
        "n_anchor": int(len(anchor_rows)),
        "n_anchor_invariance_fail": n_anchor_fail,
        "anchor_eps1_mean_abs_dY": float(np.nanmean(np.abs(anchor_eps1))) if anchor_eps1 else float("nan"),
        "random_eps1_mean_abs_dY": float(np.nanmean(np.abs(rand_eps1))) if rand_eps1 else float("nan"),
        "wrong_donor_mean_abs_dY": float(np.nanmean(np.abs(wrong))) if wrong else float("nan"),
        "slices": slices,
        "identifiability": {
            "stronger_germ": (
                "If probe-equivalent activations change F_x, a stronger germ must preserve information outside R(h). "
                "This diagnostic does not identify that information as board state, history, or policy residue. "
                "Small natural-anchor effects versus large hybrid H01 effects would favor distribution-shift over a stable nullspace coordinate."
            )
        },
    }
    write_scientific_report(out_dir / "08r2_fiber_scientific_report.md", report_payload)
    write_json(
        out_dir / "08r2_fiber_dependency_manifest.json",
        {
            "git_hash": git_hash(),
            "site": SITE,
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "probe_path": str(probe_path),
            "probe_sha256": sha256_file(probe_path),
            "holdout_sha256": holdout["manifest_sha256"],
            "pair_check": pair_check,
            "pair_flags": pair_flags,
            "validation_dir": str(val_dir),
            "results_dir": str(out_dir),
            "confirmation_sealed": True,
            "used_confirmation": 0,
            "preprocessing": prep,
            "n_contrasts": contrasts["n_contrasts"],
            "n_binary": contrasts["n_binary"],
            "n_multiclass": contrasts["n_multiclass"],
            "n_constant": contrasts["n_constant"],
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
            "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        },
    )
    completion = {
        "status": status,
        "verdict": verdict,
        "checkpoint_loaded": 1,
        "probe_loaded": 1,
        "parity_ok": int(contrast_parity["ok"] == 1),
        "nullspace_ok": int(svd.get("has_meaningful_nullspace", 0)),
        "hooks_ok": int(hooks_ok),
        "full_outputs_saved": int(len(logit_pack) >= n_valid * 4 if n_valid else len(logit_pack) > 0),
        "n_factorial_cells": int(len(factorial_rows)),
        "n_control_rows": int(len(control_rows)),
        "n_anchor_rows": int(len(anchor_rows)),
        "n_valid": n_valid,
        "n_demo": n_demo,
        "synthetic_scientific_rows": 0,
        "used_confirmation": 0,
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "section_5_3_status": "PARTIAL",
        "obstruction_ready": "NO",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "pair_counts": dict(Counter(p["r2_type"] for p in saved_pairs)),
    }
    write_json(out_dir / "08r2_fiber_completion_status.json", completion)
    gate = evaluate_fiber_completeness(out_dir, real_model=True)
    if not gate["ok"]:
        raise SystemExit(f"{STATUS_VALIDATION_FAILED} completeness {gate['failures']}")
    print("08R2-FIBER")
    print(f"  status={status}")
    print(f"  verdict={verdict}")
    print(f"  rank={svd['numerical_rank']} nullspace={svd['nullspace_dim']}")
    print(f"  matched={len(saved_pairs)} valid={n_valid} demo={n_demo} invalid={n_invalid}")
    print(f"  A: {slices['A']}")
    print(f"  B: {slices['B']}")
    print("  official letter unchanged: C / PARTIAL")
    print("  08R2_VALIDATION_COMPLETE — NEGATIVE")
    print("  SECTION_5_3 = PARTIAL")
    print("  OBSTRUCTION_READY = NO")


def kl_safe(p, q) -> float:
    from src.othello_r1 import kl_row

    return kl_row(p, q)


if __name__ == "__main__":
    main()
