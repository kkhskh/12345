"""08R2-FIBER-COST: JS-matched target-specificity diagnostic.

Does not retrain the model, probe, or editor. Does not overwrite 08R2-FIBER-SPEC.
Does not evaluate the sealed confirmation split.
Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import argparse
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
from src.othello_r2_fiber_cost import (
    ALPHA_HI_RANDOM_MULT,
    ALPHA_HI_WRONG_MULT,
    BEHAVIOR_FLOOR,
    JS_ATOL,
    JS_RTOL,
    N_BISECT,
    N_BOOT,
    STATUS_AWAITING,
    STATUS_COMPLETE,
    VERDICT_ENGINEERING,
    VERDICT_INCONCLUSIVE,
    build_cost_gate,
    decide_cost_verdict,
    evaluate_cost_completeness,
    adopt_drive_defaults,
    isolate_cost_dir,
    load_frozen_spec_manifest,
    match_output_js,
    reconstruct_frozen_pairs,
    resolve_spec_dir,
    unit_direction,
)
from src.othello_r2_fiber_spec import N_RANDOM, N_WRONG, RANDOM_SEEDS, pair_contrasts, slice_boot, y_components
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
    p = argparse.ArgumentParser(description="08R2-FIBER-COST JS-matched nullspace specificity.")
    p.add_argument("--config", default="configs/toy_othello_08r2_fiber_cost.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--spec-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--finalize-dir", default=None, help="Recompute COST verdict from written artifacts. No model.")
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
    vals = __import__("torch").tensor(np.asarray(h, dtype=np.float32).reshape(1, -1))
    logits, readback, audit = patch_and_audit(model, [tokens], SITE, vals, batch_size=1)
    return softmax_np(logits.numpy()[0]), logits.numpy()[0], audit


def kl_row_safe(p, q) -> float:
    from src.othello_r1 import kl_row

    if p is None or q is None or np.asarray(p).size == 0:
        return float("nan")
    return kl_row(p, q)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def write_report(path: Path, gate, verdict, slices, attempts, manifest, svd) -> None:
    lines = [
        "# 08R2-FIBER-COST Scientific Report",
        "",
        "JS-matched correct-versus-wrong-donor probe-null interventions on the frozen 08R2-FIBER-SPEC pair set.",
        "08R2-FIBER-SPEC remains a valid positive result at matched residual L2. This report does not revise it.",
        "Residual L2 and output JS were not controlled simultaneously.",
        "",
        "- `08R2_VALIDATION_COMPLETE — NEGATIVE`",
        "- Original 0A: **C / PARTIAL**",
        "- `SECTION_5_3 = PARTIAL`",
        "- `OBSTRUCTION_READY = NO`",
        "",
        f"Verdict: `{verdict}`",
        f"Frozen pairs: A={manifest.get('n_a')} B={manifest.get('n_b')} hash={manifest.get('pair_manifest_sha256')}",
        f"Attempts: {attempts}",
        f"Gate: {gate}",
        f"SVD rank={svd.get('numerical_rank')} null={svd.get('nullspace_dim')}",
        f"Pooled: {slices.get('pooled')}",
        f"A: {slices.get('A')}",
        f"B: {slices.get('B')}",
        "",
        "A supported gate means the correct-donor null component beat JS-matched wrong donors and random null directions on the intended legal set.",
        "It does not identify a unique game-state coordinate and does not validate obstruction theory.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_cost_summary(verdict, gate, slices, attempts, n_pairs, match_rows=None) -> None:
    pooled = slices.get("pooled", {})
    print("08R2-FIBER-COST")
    print(f"  verdict={verdict}")
    print(f"  valid_pairs={attempts.get('valid', 0)}/{n_pairs}")
    print(f"  attempts={attempts}")
    print(f"  pooled={pooled}")
    print(f"  gate={gate}")
    if match_rows:
        reasons = {}
        for row in match_rows:
            reasons[str(row.get("reason", ""))] = reasons.get(str(row.get("reason", "")), 0) + 1
        print(f"  js_match_reasons={reasons}")
    print("  SPEC at matched L2 is unchanged: TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED")
    print("  official letter unchanged: C / PARTIAL")
    print("  08R2_VALIDATION_COMPLETE — NEGATIVE")
    print("  SECTION_5_3 = PARTIAL")
    print("  OBSTRUCTION_READY = NO")


def finalize_from_dir(out_dir: Path, cfg: dict) -> None:
    import csv
    import json

    out_dir = Path(out_dir)
    status = json.loads((out_dir / "fiber_cost_completion_status.json").read_text(encoding="utf-8"))
    attempts = json.loads((out_dir / "fiber_cost_attempt_accounting.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "fiber_cost_pair_manifest.json").read_text(encoding="utf-8"))
    match_rows = []
    with (out_dir / "fiber_cost_js_match.csv").open(encoding="utf-8", newline="") as handle:
        match_rows = list(csv.DictReader(handle))
    slices = status.get("slices") or {}
    n_pairs = int(manifest.get("n", 48))
    gate = build_cost_gate(
        attempts,
        n_pairs,
        hooks_ok=1,
        parity_ok=1,
        slices=slices,
        behavior_floor=float(cfg.get("behavior_floor", BEHAVIOR_FLOOR)),
    )
    verdict = decide_cost_verdict(gate)
    status["gate"] = gate
    status["verdict"] = verdict
    status["js_matched_controls"] = int(gate.get("matching_support_ok", 0))
    status["finalized_without_rerun"] = 1
    write_json(out_dir / "fiber_cost_completion_status.json", status)
    write_report(out_dir / "fiber_cost_scientific_report.md", gate, verdict, slices, attempts, manifest, {"numerical_rank": None, "nullspace_dim": None})
    print_cost_summary(verdict, gate, slices, attempts, n_pairs, match_rows)
    gate_files = evaluate_cost_completeness(out_dir, real_model=True)
    if not gate_files["ok"]:
        raise SystemExit(f"{VERDICT_ENGINEERING} completeness {gate_files['failures']}")


def write_awaiting(out_dir: Path, reason: str, extra=None) -> None:
    payload = {
        "status": STATUS_AWAITING,
        "verdict": VERDICT_INCONCLUSIVE,
        "reason": reason,
        "used_confirmation": 0,
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "section_5_3_status": "PARTIAL",
        "obstruction_ready": "NO",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
    }
    if extra:
        payload.update(extra)
    write_json(out_dir / "fiber_cost_completion_status.json", payload)
    print(STATUS_AWAITING)
    print(f"  reason={reason}")
    if extra:
        for key in extra:
            print(f"  {key}={extra[key]}")


def main() -> None:
    args = parse_args()
    parent = load_config(args.parent_config)
    r2cfg = load_config(args.r2_config)
    cfg = merge_cfg(parent, r2cfg, load_config(args.config))
    if args.finalize_dir:
        finalize_from_dir(args.finalize_dir, cfg)
        return
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.checkpoints_dir:
        cfg["checkpoints_dir"] = args.checkpoints_dir
    adopted = adopt_drive_defaults(cfg, args)
    spec_res = resolve_spec_dir(args.spec_dir, cfg["results_dir"])
    spec_dir = Path(spec_res["spec_dir"]) if spec_res["spec_dir"] is not None else None
    out_dir = ensure_dir(isolate_cost_dir(cfg["results_dir"])["results_dir"])
    print(f"08R2-FIBER-COST PATHS spec_dir={spec_dir} source={spec_res['source']}")
    print(f"  results_dir={cfg['results_dir']}")
    print(f"  data_dir={cfg['data_dir']}")
    print(f"  checkpoints_dir={cfg['checkpoints_dir']}")
    if adopted:
        print(f"  adopted={adopted}")
    if spec_res["source"] == "missing":
        print(f"  spec_tried={spec_res['tried']}")
    prereg_src = Path(cfg.get("preregistration", "docs/experiments/08r2_fiber_cost_preregistration.md"))
    if prereg_src.is_file():
        (out_dir / "fiber_cost_preregistration.md").write_text(prereg_src.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(
        out_dir / "fiber_cost_preregistration.json",
        {
            k: cfg[k]
            for k in cfg
            if k in {
                "expected_pair_manifest_sha256",
                "n_a",
                "n_b",
                "pair_seed",
                "js_atol",
                "js_rtol",
                "n_bisect",
                "alpha_hi_wrong_mult",
                "alpha_hi_random_mult",
                "behavior_floor",
                "primary_rank_tol",
                "probe_proba_atol_f32",
                "n_boot",
                "boot_seed",
                "editor_budget",
                "random_seeds",
                "n_wrong",
                "n_random",
                "wrong_seed",
            }
        },
    )

    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    if not holdout["hash_ok"]:
        raise SystemExit("FIBER_COST_ENGINEERING_FAILED confirmation manifest hash mismatch")
    data_dir = Path(cfg["data_dir"])
    val = add_labels(load_split(data_dir, "val"))
    require_no_confirmation(val, holdout["game_ids"])

    expected_hash = cfg.get("expected_pair_manifest_sha256")
    try:
        if spec_dir is not None:
            manifest = load_frozen_spec_manifest(spec_dir, expected_hash)
            pair_source = spec_res["source"]
        else:
            manifest = reconstruct_frozen_pairs(
                val,
                holdout["game_ids"],
                expected_hash,
                n_a=int(cfg.get("n_a", 32)),
                n_b=int(cfg.get("n_b", 16)),
                pair_seed=int(cfg.get("pair_seed", 2182)),
                n_wrong=int(cfg.get("n_wrong", 3)),
                wrong_seed=int(cfg.get("wrong_seed", 2282)),
            )
            pair_source = "reconstructed_from_frozen_seeds"
    except ValueError as exc:
        raise SystemExit(f"{VERDICT_ENGINEERING} {exc}")
    write_json(out_dir / "fiber_cost_pair_manifest.json", manifest)
    pairs = manifest["pairs"]
    wrongs = manifest["wrong_donors"]
    print("08R2-FIBER-COST FROZEN PAIRS")
    print(f"  spec_dir={spec_dir}")
    print(f"  pair_source={pair_source}")
    print(f"  A={manifest['n_a']} B={manifest['n_b']} hash={manifest['pair_manifest_sha256']}")
    print(f"  hash_ok={int(manifest['pair_manifest_sha256'] == expected_hash)}")

    try:
        _require_real_stack()
    except RuntimeError as exc:
        write_awaiting(out_dir, str(exc))
        return

    probe_info = resolve_probe_path(args.probe_path, cfg["checkpoints_dir"])
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if not probe_info["exists"] or ckpt is None or not Path(ckpt).is_file():
        write_awaiting(
            out_dir,
            "BLOCKED_MISSING_ARTIFACT",
            {
                "checkpoint_loaded": 0,
                "probe_loaded": int(probe_info["exists"]),
                "checkpoint_path": str(ckpt) if ckpt is not None else str(Path(cfg["checkpoints_dir"]) / cfg.get("checkpoint_file", "synthetic_model.pth")),
                "probe_path": probe_info["probe_path"],
                "checkpoints_dir": cfg["checkpoints_dir"],
            },
        )
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
    labeled = val
    for pair in pairs:
        rec = labeled[int(pair["recip"])]
        tgt = labeled[int(pair["donor"])]
        if rec["game_id"] != pair["recip_game"] or tgt["game_id"] != pair["target_game"]:
            raise SystemExit(f"{VERDICT_ENGINEERING} pair index/game_id mismatch {pair}")
    needed = sorted({int(p["recip"]) for p in pairs} | {int(p["donor"]) for p in pairs})
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
    spec_vecs = None
    spec_vec_path = (spec_dir / "fiber_spec_nullspace_vectors.npz") if spec_dir is not None else None
    if spec_vec_path is not None and spec_vec_path.is_file():
        spec_vecs = np.load(spec_vec_path)

    rows = []
    match_rows = []
    dose_rows = []
    legal = {}
    logit_pack = {}
    probe_pack = {}
    null_pack = {}
    hooks_ok = 1
    attempts = {
        "pairs": 0,
        "editor_fail": 0,
        "invariance_fail": 0,
        "js_fail": 0,
        "direction_fail": 0,
        "valid": 0,
    }
    js_atol = float(cfg.get("js_atol", JS_ATOL))
    js_rtol = float(cfg.get("js_rtol", JS_RTOL))
    n_bisect = int(cfg.get("n_bisect", N_BISECT))
    inv_atol = float(cfg.get("probe_proba_atol_f32", 1e-5))

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
        inv = float(np.max(np.abs(probe - pe))) if name not in {"baseline", "editor_A", "full_donor"} else float("nan")
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
        rows.append(row)
        key = f"{pair['recip_game']}__{pair['target_game']}__{name}"
        logit_pack[key] = np.asarray(logits, dtype=np.float64)
        probe_pack[key] = probe
        return row, probs, probe, y

    def eval_direction(pair, unit, alpha, h_e, probs_e):
        h = np.asarray(h_e, dtype=np.float64) + float(alpha) * np.asarray(unit, dtype=np.float64)
        rec = labeled[int(pair["recip"])]
        tgt = labeled[int(pair["donor"])]
        probs, logits, audit = _patch(model, rec["tokens"], h)
        y = y_components(probs, rec["legal"], tgt["legal"])
        probe = decoder.proba_np(h).reshape(64, 3)
        pe = decoder.proba_np(h_e).reshape(64, 3)
        return {
            "js": float(js_divergence(probs_e, probs)),
            "probs": probs,
            "logits": logits,
            "audit": audit,
            "y": y,
            "probe": probe,
            "invariance_max_err": float(np.max(np.abs(probe - pe))),
            "l2": float(abs(alpha)),
        }

    for pair, wrep in zip(pairs, wrongs):
        attempts["pairs"] += 1
        rec = labeled[int(pair["recip"])]
        tgt = labeled[int(pair["donor"])]
        legal[f"{pair['recip_game']}|{pair['target_game']}"] = {
            "recip_legal": list(map(int, rec["legal"])),
            "target_legal": list(map(int, tgt["legal"])),
            "became_legal": sorted(set(tgt["legal"]) - set(rec["legal"])),
            "became_illegal": sorted(set(rec["legal"]) - set(tgt["legal"])),
        }
        h_rec = np.asarray(H[int(pair["recip"])], dtype=np.float64)
        h_d = np.asarray(H[int(pair["donor"])], dtype=np.float64)
        ed = iterative_state_edit(
            h_rec,
            tgt["rel_lab"],
            rec["rel_lab"],
            dec_t,
            steps=int(cfg.get("edit_steps", 40)),
            lr=float(cfg.get("edit_lr", 0.4)),
            budget=float(cfg.get("editor_budget", 16.0)),
            lambda_norm=float(cfg["lambda_norm"]),
            eta_preserve=float(cfg["eta_preserve"]),
            mu_manifold=float(cfg.get("mu_manifold", 0.0)),
        )
        if not ed.get("valid_edit") or ed.get("edited") is None:
            attempts["editor_fail"] += 1
            rows.append({"cell": "editor_A", "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 0, "reason": ed.get("reason")})
            continue
        h_e = np.asarray(ed["edited"], dtype=np.float64)
        rec_e, probs_e, probe_e, _ = record(
            pair,
            "editor_A",
            h_e,
            h_e,
            pair["target_game"],
            extra={"valid": 1, "editor_l2": float(np.linalg.norm(h_e - h_rec)), "board_exact_editor": decoder_state_effects(decoder, h_e, rec["rel_lab"], tgt["rel_lab"])["board_exact"]},
        )
        record(pair, "baseline", h_rec, h_e, pair["recip_game"], extra={"valid": 1})
        record(pair, "full_donor", h_d, h_e, pair["target_game"], extra={"valid": 1})
        d_n = p_n @ (h_d - h_e)
        n_c = unit_direction(d_n, float(cfg.get("l2_min_norm", 1e-8)))
        if n_c is None:
            attempts["direction_fail"] += 1
            continue
        if spec_vecs is not None:
            key = f"{pair['recip_game']}__{pair['target_game']}__correct"
            if key in spec_vecs.files and abs(cosine(d_n, spec_vecs[key])) < 0.999:
                raise SystemExit(f"{VERDICT_ENGINEERING} correct-null direction drifted from SPEC {pair}")
        target_l2 = float(np.linalg.norm(d_n))
        h_c = h_e + d_n
        rec_c, probs_c, probe_c, y_c = record(
            pair,
            "correct_null",
            h_c,
            h_e,
            pair["target_game"],
            extra={"valid": 1, "alpha": target_l2, "l2_unmatched_spec": target_l2, "js_match_role": "target"},
        )
        target_js = float(js_divergence(probs_e, probs_c))
        rec_c["js_to_editor"] = target_js
        rec_c["target_js"] = target_js
        match_rows.append(
            {
                "cell": "correct_null",
                "recip_game": pair["recip_game"],
                "target_game": pair["target_game"],
                "ok": 1,
                "target_js": target_js,
                "js": target_js,
                "alpha": target_l2,
                "l2": target_l2,
                "reason": "reference",
            }
        )
        null_pack[f"{pair['recip_game']}__{pair['target_game']}__correct"] = d_n

        y_w, g_w, q_w = [], [], []
        y_r = []
        inv_ok = 1
        js_ok = 1
        if rec_c["invariance_max_err"] > inv_atol:
            inv_ok = 0

        def run_control(name, raw, donor_game, alpha_mult, extra=None):
            nonlocal hooks_ok, inv_ok, js_ok
            unit = unit_direction(raw, float(cfg.get("l2_min_norm", 1e-8)))
            if unit is None:
                attempts["direction_fail"] += 1
                js_ok = 0
                rows.append({"cell": name, "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 0, "reason": "degenerate_direction"})
                return None
            if spec_vecs is not None:
                spec_name = name.replace("wrong_null_", "wrong_").replace("random_null_", "random_")
                key = f"{pair['recip_game']}__{pair['target_game']}__{spec_name}"
                if key in spec_vecs.files and abs(cosine(unit, spec_vecs[key])) < 0.999:
                    raise SystemExit(f"{VERDICT_ENGINEERING} {name} direction drifted from SPEC {pair}")

            def ev(alpha):
                out = eval_direction(pair, unit, alpha, h_e, probs_e)
                dose_rows.append(
                    {
                        "cell": name,
                        "recip_game": pair["recip_game"],
                        "target_game": pair["target_game"],
                        "alpha": float(alpha),
                        "js": out["js"],
                        "Y": out["y"]["Y"],
                        "G": out["y"]["G"],
                        "Q": out["y"]["Q"],
                        "l2": out["l2"],
                        "target_js": target_js,
                    }
                )
                return out

            matched = match_output_js(ev, target_js, float(alpha_mult) * target_l2, js_atol, js_rtol, n_bisect)
            match_rows.append(
                {
                    "cell": name,
                    "recip_game": pair["recip_game"],
                    "target_game": pair["target_game"],
                    "ok": matched.get("ok", 0),
                    "reason": matched.get("reason"),
                    "target_js": target_js,
                    "js": matched.get("js"),
                    "alpha": matched.get("alpha"),
                    "l2": matched.get("l2", matched.get("alpha")),
                    "js_abs_err": matched.get("js_abs_err"),
                }
            )
            if not matched.get("ok"):
                attempts["js_fail"] += 1
                js_ok = 0
                rows.append({"cell": name, "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 0, "reason": matched.get("reason")})
                return None
            vec = float(matched["alpha"]) * unit
            null_pack[f"{pair['recip_game']}__{pair['target_game']}__{name}"] = vec
            hooks_ok = hooks_ok and int(matched.get("audit", {}).get("ok", 0))
            rec_x, _, _, y_x = record(
                pair,
                name,
                h_e + vec,
                h_e,
                donor_game,
                extra={
                    "valid": 1,
                    "alpha": matched["alpha"],
                    "target_js": target_js,
                    "js_to_editor": matched["js"],
                    "js_abs_err": matched.get("js_abs_err"),
                    "js_match_role": "control",
                    **(extra or {}),
                },
            )
            if rec_x["invariance_max_err"] > inv_atol:
                inv_ok = 0
            rec_x["kl_to_editor"] = kl_row_safe(probs_e, matched["probs"])
            return y_x

        for j, wd in enumerate(wrep["donors"]):
            h_wj = np.asarray(H[int(wd["index"])], dtype=np.float64)
            y_xj = run_control(
                f"wrong_null_{j}",
                p_n @ (h_wj - h_e),
                wd["game_id"],
                float(cfg.get("alpha_hi_wrong_mult", ALPHA_HI_WRONG_MULT)),
                extra={k: wd[k] for k in wd if k != "index"},
            )
            if y_xj is not None:
                y_w.append(y_xj["Y"])
                g_w.append(y_xj["G"])
                q_w.append(y_xj["Q"])
        for j, seed in enumerate(cfg.get("random_seeds", RANDOM_SEEDS)):
            rng = np.random.default_rng(int(seed) + int(pair["recip"]))
            y_rj = run_control(
                f"random_null_{j}",
                p_n @ rng.normal(size=p_n.shape[0]),
                "random",
                float(cfg.get("alpha_hi_random_mult", ALPHA_HI_RANDOM_MULT)),
                extra={"seed": int(seed)},
            )
            if y_rj is not None:
                y_r.append(y_rj["Y"])

        if not inv_ok:
            attempts["invariance_fail"] += 1
        if inv_ok and js_ok and len(y_w) == int(cfg.get("n_wrong", N_WRONG)) and len(y_r) == int(cfg.get("n_random", N_RANDOM)):
            attempts["valid"] += 1
            ctr = pair_contrasts(y_c["Y"], y_w, y_r)
            rec_c.update(ctr)
            rec_c["dG_wrong"] = float(y_c["G"] - np.mean(g_w))
            rec_c["dQ_wrong"] = float(y_c["Q"] - np.mean(q_w))
            rec_c["pair_valid"] = 1
            rows.append(
                {
                    "cell": "pair_contrast",
                    "r2_type": pair["r2_type"],
                    "recip_game": pair["recip_game"],
                    "target_game": pair["target_game"],
                    "valid": 1,
                    **ctr,
                    "dG_wrong": rec_c["dG_wrong"],
                    "dQ_wrong": rec_c["dQ_wrong"],
                    "js_correct_vs_editor": target_js,
                    "kl_correct_vs_editor": kl_row_safe(probs_e, probs_c),
                    "l2_correct": target_l2,
                }
            )
        for row in rows:
            if row.get("recip_game") == pair["recip_game"] and row.get("target_game") == pair["target_game"] and "kl_to_editor" in row:
                key = f"{pair['recip_game']}__{pair['target_game']}__{row['cell']}"
                ed_key = f"{pair['recip_game']}__{pair['target_game']}__editor_A"
                if key in logit_pack and ed_key in logit_pack:
                    row["kl_to_editor"] = kl_row_safe(softmax(logit_pack[ed_key]), softmax(logit_pack[key]))
                    row["js_to_editor"] = js_divergence(softmax(logit_pack[ed_key]), softmax(logit_pack[key]))

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
            "mean_l2_correct": float(np.nanmean([r.get("l2_correct", np.nan) for r in g])),
            "frac_correct_wins_wrong": float(np.mean([r["delta_specific"] > 0 for r in g])),
            "frac_correct_wins_random": float(np.mean([r["delta_random"] > 0 for r in g])),
        }
        sl["delta_specific_ci"] = slice_boot([r["delta_specific"] for r in g], [r["recip_game"] for r in g], int(cfg.get("n_boot", N_BOOT)), int(cfg.get("boot_seed", 82)))
        sl["delta_random_ci"] = slice_boot([r["delta_random"] for r in g], [r["recip_game"] for r in g], int(cfg.get("n_boot", N_BOOT)), int(cfg.get("boot_seed", 82)) + 1)
        slices[name] = sl
        boot_rows.append({"slice": name, "metric": "delta_specific", **sl["delta_specific_ci"]})
        boot_rows.append({"slice": name, "metric": "delta_random", **sl["delta_random_ci"]})

    pooled = slices.get("pooled", {})
    gate = build_cost_gate(
        attempts,
        len(pairs),
        hooks_ok=hooks_ok,
        parity_ok=int(parity["ok"] == 1),
        slices=slices,
        behavior_floor=float(cfg.get("behavior_floor", BEHAVIOR_FLOOR)),
    )
    verdict = decide_cost_verdict(gate)
    write_csv(out_dir / "fiber_cost_interventions.csv", rows)
    write_csv(out_dir / "fiber_cost_js_match.csv", match_rows or [{"reason": "no_js_rows", "n": 0}])
    write_csv(out_dir / "fiber_cost_dose_search.csv", dose_rows or [{"reason": "no_search", "n": 0}])
    write_csv(out_dir / "fiber_cost_bootstrap.csv", boot_rows or [{"reason": "no_valid_contrasts", "n": 0}])
    write_json(out_dir / "fiber_cost_attempt_accounting.json", attempts)
    np.savez_compressed(out_dir / "fiber_cost_full_logits.npz", **logit_pack)
    np.savez_compressed(out_dir / "fiber_cost_probe_probabilities.npz", **probe_pack)
    write_json(
        out_dir / "fiber_cost_dependency_manifest.json",
        {
            "git_hash": git_hash(),
            "site": SITE,
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "probe_path": probe_info["probe_path"],
            "probe_sha256": sha256_file(probe_info["probe_path"]),
            "pair_manifest_sha256": manifest["pair_manifest_sha256"],
            "spec_dir": str(spec_dir),
            "nanda": ident,
            "svd_rank": svd["numerical_rank"],
            "nullspace_dim": svd["nullspace_dim"],
            "used_confirmation": 0,
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
            "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
            "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
            "matching": "output_js_from_editor_A",
            "l2_also_controlled": 0,
        },
    )
    write_report(out_dir / "fiber_cost_scientific_report.md", gate, verdict, slices, attempts, manifest, svd)
    write_json(
        out_dir / "fiber_cost_completion_status.json",
        {
            "status": STATUS_COMPLETE,
            "verdict": verdict,
            "gate": gate,
            "slices": slices,
            "attempts": attempts,
            "checkpoint_loaded": 1,
            "probe_loaded": 1,
            "pairs_ok": 1,
            "js_matched_controls": int(gate.get("matching_support_ok", 0)),
            "full_outputs_saved": int(len(logit_pack) > 0),
            "used_confirmation": 0,
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "section_5_3_status": "PARTIAL",
            "obstruction_ready": "NO",
            "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
            "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
        },
    )
    print_cost_summary(verdict, gate, slices, attempts, len(pairs), match_rows)
    gate_files = evaluate_cost_completeness(out_dir, real_model=True)
    if not gate_files["ok"]:
        raise SystemExit(f"{VERDICT_ENGINEERING} completeness {gate_files['failures']}")


if __name__ == "__main__":
    main()
