"""Restore 08R2-FIBER-COST at a frozen JS=0.01 for all seven directions.

Does not overwrite fiber_spec/, fiber_cost/, or fiber_cost_forensic/.
Does not evaluate confirmation. Official 0A letter remains C / PARTIAL.
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
    adopt_drive_defaults,
    load_frozen_spec_manifest,
    reconstruct_frozen_pairs,
    resolve_spec_dir,
)
from src.othello_r2_fiber_cost_fixed import (
    BEHAVIOR_FLOOR,
    CELLS,
    COVERAGE_MIN,
    FIXED_JS_PRIMARY,
    FIXED_JS_SECONDARY,
    FIXED_JS_TOL,
    N_BISECT,
    N_BOOT,
    SCALE_GRID,
    STATUS_AWAITING,
    STATUS_COMPLETE,
    VERDICT_ENGINEERING,
    VERDICT_INCONCLUSIVE,
    bind_literal_js_target,
    build_direction_tasks,
    build_fixed_js_gate,
    calibrate_fixed_js,
    decide_fixed_js_verdict,
    evaluate_fixed_js_completeness,
    isolate_fixed_js_dir,
    reject_donor_reference_rows,
    scaled_displacement,
)
from src.othello_r2_fiber_spec import RANDOM_SEEDS, match_l2, pair_contrasts, slice_boot, y_components
from src.othello_runutil import add_labels, load_split, tokens_of, write_json
from src.toy_othello import cache_site, load_config, load_othello_gpt, run_model_logits, softmax_np, verify_nanda_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2-FIBER-COST restored fixed-JS=0.01 protocol.")
    p.add_argument("--config", default="configs/toy_othello_08r2_fiber_cost_fixed_js.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--r2-config", default="configs/toy_othello_08r2.json")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--spec-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--js-target", type=float, default=FIXED_JS_PRIMARY)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def merge_cfg(*parts: dict) -> dict:
    out = {}
    for p in parts:
        out.update(p)
    return out


def preflight_tasks(manifest: dict, js_target: float = FIXED_JS_PRIMARY) -> list[dict]:
    tasks = build_direction_tasks(manifest["pairs"], manifest["wrong_donors"], int(manifest.get("n_random", 3) or 3))
    return bind_literal_js_target(tasks, float(js_target))


def calibrate_task(task: dict, eval_t, direction=None, *, grid=None, tol=None) -> dict:
    target = float(task.get("target_js", FIXED_JS_PRIMARY))
    if abs(target - float(FIXED_JS_PRIMARY)) > 1e-12 and target not in {float(x) for x in FIXED_JS_SECONDARY}:
        raise ValueError(f"runner refused non-frozen JS target {target}")
    out = calibrate_fixed_js(eval_t, target_js=target, grid=grid or SCALE_GRID, tol=tol if tol is not None else FIXED_JS_TOL)
    if direction is not None and out.get("t") == out.get("t"):
        sc = scaled_displacement(direction, out["t"])
        out["l2"] = sc["l2"]
        out["vec"] = sc["vec"]
        out["base_l2"] = sc["base_l2"]
        if abs(out["l2"] - float(np.linalg.norm(sc["vec"]))) > 1e-10:
            raise RuntimeError("reported L2 drifted from ||t δ||")
    out["cell"] = task.get("cell")
    out["target_js"] = target
    if out.get("reason") == "reference" or out.get("status") == "reference":
        raise RuntimeError("fixed-JS calibrator returned a reference row")
    return out


def print_preflight(tasks: list[dict], js_target: float) -> dict:
    n = len(tasks)
    n_correct = sum(1 for t in tasks if t["cell"] == "correct_null")
    ok = int(n == 336 and n_correct == 48 and all(abs(float(t["target_js"]) - float(js_target)) < 1e-15 for t in tasks))
    print("08R2-FIBER-COST-FIXED-JS PREFLIGHT")
    print(f"  n_tasks={n} n_correct={n_correct} target_js={js_target}")
    print(f"  all_literal_target={ok}")
    if n <= 14:
        for t in tasks:
            print(f"  {t['recip_game']} {t['cell']} target={t['target_js']}")
    else:
        for t in tasks[:3] + tasks[48:51] + tasks[-3:]:
            print(f"  {t['recip_game']} {t['cell']} target={t['target_js']}")
        print(f"  ... {n} rows, all target_js={js_target}")
    if not ok:
        raise SystemExit(f"{VERDICT_ENGINEERING} preflight target check failed")
    return {"ok": ok, "n": n, "n_correct": n_correct, "target_js": js_target}


def kl_row_safe(p, q) -> float:
    from src.othello_r1 import kl_row

    if p is None or q is None:
        return float("nan")
    return kl_row(p, q)


def write_report(path: Path, gate, verdict, slices, attempts, manifest, js_target) -> None:
    path.write_text(
        "\n".join(
            [
                "# 08R2-FIBER-COST Fixed-JS Scientific Report",
                "",
                "Protocol-compliant exploratory follow-up. All seven directions calibrated to a frozen JS target.",
                "This does not revise the donor-reference COST verdict or SPEC.",
                "",
                "- `08R2_VALIDATION_COMPLETE — NEGATIVE`",
                "- Original 0A: **C / PARTIAL**",
                "- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE` preserved",
                "- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED` preserved",
                "",
                f"Verdict: `{verdict}`",
                f"JS target: {js_target}",
                f"Frozen pairs hash={manifest.get('pair_manifest_sha256')}",
                f"Attempts: {attempts}",
                f"Gate: {gate}",
                f"Pooled: {slices.get('pooled')}",
                f"A: {slices.get('A')}",
                f"B: {slices.get('B')}",
                "",
                "Equal JS matches disruption magnitude, not the output distribution.",
                "",
            ]
        ),
        encoding="utf-8",
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
    adopt_drive_defaults(cfg, args)
    js_target = float(args.js_target)
    if abs(js_target - float(FIXED_JS_PRIMARY)) > 1e-12:
        print(f"SECONDARY TARGET {js_target}; primary remains {FIXED_JS_PRIMARY}")
    out_dir = Path(isolate_fixed_js_dir(cfg["results_dir"])["results_dir"])
    if args.smoke:
        out_dir = out_dir / "smoke"
    ensure_dir(out_dir)
    spec_res = resolve_spec_dir(args.spec_dir, cfg["results_dir"])
    spec_dir = Path(spec_res["spec_dir"]) if spec_res["spec_dir"] is not None else None

    prereg_src = Path(cfg.get("preregistration", "docs/experiments/08r2_fiber_cost_fixed_js_preregistration.md"))
    if prereg_src.is_file():
        (out_dir / "fixed_js_preregistration.md").write_text(prereg_src.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(out_dir / "fixed_js_preregistration.json", {k: cfg[k] for k in cfg if k in {
        "expected_pair_manifest_sha256", "fixed_js_primary", "fixed_js_secondary", "fixed_js_tol",
        "scale_grid", "n_bisect", "behavior_floor", "coverage_min", "pair_seed", "n_a", "n_b",
        "random_seeds", "n_wrong", "n_random", "editor_budget",
    }})

    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    if not holdout["hash_ok"]:
        raise SystemExit(f"{VERDICT_ENGINEERING} confirmation manifest hash mismatch")
    val = add_labels(load_split(Path(cfg["data_dir"]), "val"))
    require_no_confirmation(val, holdout["game_ids"])
    expected_hash = cfg.get("expected_pair_manifest_sha256")
    try:
        if spec_dir is not None:
            manifest = load_frozen_spec_manifest(spec_dir, expected_hash)
        else:
            manifest = reconstruct_frozen_pairs(
                val, holdout["game_ids"], expected_hash,
                n_a=int(cfg.get("n_a", 32)), n_b=int(cfg.get("n_b", 16)),
                pair_seed=int(cfg.get("pair_seed", 2182)),
                n_wrong=int(cfg.get("n_wrong", 3)), wrong_seed=int(cfg.get("wrong_seed", 2282)),
            )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"{VERDICT_ENGINEERING} {exc}")
    write_json(out_dir / "fixed_js_pair_manifest.json", manifest)
    pairs = list(manifest["pairs"])
    wrongs = list(manifest["wrong_donors"])
    if args.smoke:
        idx = int(cfg.get("smoke_pair_index", 0))
        pairs = [pairs[idx]]
        wrongs = [wrongs[idx]]
    tasks = preflight_tasks({"pairs": pairs if not args.smoke else manifest["pairs"], "wrong_donors": wrongs if not args.smoke else manifest["wrong_donors"]}, js_target)
    if args.smoke:
        rg, tg = pairs[0]["recip_game"], pairs[0]["target_game"]
        tasks = [t for t in tasks if t["recip_game"] == rg and t["target_game"] == tg]
        preflight = {
            "ok": int(len(tasks) == 7 and all(abs(t["target_js"] - js_target) < 1e-15 for t in tasks) and any(t["cell"] == "correct_null" for t in tasks)),
            "n": len(tasks),
            "n_correct": sum(1 for t in tasks if t["cell"] == "correct_null"),
            "target_js": js_target,
            "smoke": 1,
        }
        print("08R2-FIBER-COST-FIXED-JS SMOKE PREFLIGHT")
        print(f"  pair={rg}|{tg} n={len(tasks)} target={js_target} ok={preflight['ok']}")
        if not preflight["ok"]:
            raise SystemExit(f"{VERDICT_ENGINEERING} smoke preflight failed")
    else:
        preflight = print_preflight(tasks, js_target)
    write_csv(out_dir / "fixed_js_preflight.csv", tasks)
    chk = reject_donor_reference_rows(tasks, js_target)
    if not chk["ok"]:
        raise SystemExit(f"{VERDICT_ENGINEERING} preflight {chk['failures']}")
    print(f"  preflight_ok=1 n={len(tasks)} target_js={js_target}")
    if args.preflight:
        write_json(out_dir / "fixed_js_completion_status.json", {
            "status": STATUS_AWAITING,
            "reason": "preflight_only",
            "preflight": preflight,
            "used_confirmation": 0,
            "official_0A_letter": OFFICIAL_0A_LETTER,
            "prior_cost_verdict_preserved": "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
        })
        print(STATUS_AWAITING)
        return

    try:
        import torch  # noqa: F401
        import transformer_lens  # noqa: F401
    except Exception as exc:
        write_json(out_dir / "fixed_js_completion_status.json", {"status": STATUS_AWAITING, "reason": str(exc), "preflight": preflight, "used_confirmation": 0})
        print(STATUS_AWAITING)
        return

    probe_info = resolve_probe_path(args.probe_path, cfg["checkpoints_dir"])
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if not probe_info["exists"] or ckpt is None or not Path(ckpt).is_file():
        write_json(out_dir / "fixed_js_completion_status.json", {"status": STATUS_AWAITING, "reason": "BLOCKED_MISSING_ARTIFACT", "preflight": preflight, "used_confirmation": 0})
        print(STATUS_AWAITING)
        return

    with Path(probe_info["probe_path"]).open("rb") as handle:
        pack = pickle.load(handle)
    prep = reject_unexpected_preprocessing(pack)
    if not prep["ok"] or str(pack.get("hook")) != SITE:
        raise SystemExit(f"{VERDICT_ENGINEERING} probe identity {prep}")
    decoder = LinearBoardDecoder(pack)
    contrasts = independent_contrasts(decoder)
    dec_t = decoder.as_torch()
    model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=str(ckpt))
    ident = verify_nanda_checkpoint(model)
    if not ident.get("nanda_argmax_match"):
        raise SystemExit(f"{VERDICT_ENGINEERING} checkpoint {ident}")
    params0 = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    coef0 = decoder.coef.copy()
    H = cache_site(model, tokens_of(val), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    labeled = val
    svd = svd_probe_geometry(contrasts["W"], float(cfg.get("primary_rank_tol", 1e-8)), cfg.get("rank_tolerances", [1e-8]))
    if not svd.get("has_meaningful_nullspace"):
        raise SystemExit(f"{VERDICT_ENGINEERING} no nullspace")
    p_n = svd["P_N"]
    needed = sorted({int(p["recip"]) for p in pairs} | {int(p["donor"]) for p in pairs})
    for w in wrongs:
        needed.extend(int(d["index"]) for d in w["donors"])
    needed = sorted(set(needed))
    parity = contrast_parity_report(decoder, pack["models"], H[needed[:16]], contrasts, float(cfg.get("parity_proba_atol", 1e-6)))
    if parity["ok"] != 1:
        raise SystemExit(f"{VERDICT_ENGINEERING} contrast parity {parity}")
    torch_p = torch_parity_report(decoder, pack["models"], H[needed[:16]])
    write_csv(out_dir / "fixed_js_probe_parity.csv", [parity, {"backend": "torch", **{k: torch_p[k] for k in torch_p if k != "class_sets"}}])

    grid = tuple(cfg.get("scale_grid", SCALE_GRID))
    tol = float(cfg.get("fixed_js_tol", FIXED_JS_TOL))
    inv_atol = float(cfg.get("probe_proba_atol_f32", 1e-5))
    rows = []
    cal_rows = []
    dose_rows = []
    logit_pack = {}
    probe_pack = {}
    hooks_ok = 1
    attempts = {"pairs": 0, "editor_fail": 0, "invariance_fail": 0, "js_fail": 0, "direction_fail": 0, "valid": 0, "kept_failures": 0}

    def patch(tokens, h):
        vals = __import__("torch").tensor(np.asarray(h, dtype=np.float32).reshape(1, -1))
        logits, readback, audit = patch_and_audit(model, [tokens], SITE, vals, batch_size=1)
        return softmax_np(logits.numpy()[0]), logits.numpy()[0], audit

    for pair, wrep in zip(pairs, wrongs):
        attempts["pairs"] += 1
        rec = labeled[int(pair["recip"])]
        tgt = labeled[int(pair["donor"])]
        if rec["game_id"] != pair["recip_game"] or tgt["game_id"] != pair["target_game"]:
            raise SystemExit(f"{VERDICT_ENGINEERING} pair index mismatch {pair}")
        h_rec = np.asarray(H[int(pair["recip"])], dtype=np.float64)
        h_d = np.asarray(H[int(pair["donor"])], dtype=np.float64)
        ed = iterative_state_edit(
            h_rec, tgt["rel_lab"], rec["rel_lab"], dec_t,
            steps=int(cfg.get("edit_steps", 40)), lr=float(cfg.get("edit_lr", 0.4)),
            budget=float(cfg.get("editor_budget", 16.0)), lambda_norm=float(cfg["lambda_norm"]),
            eta_preserve=float(cfg["eta_preserve"]), mu_manifold=float(cfg.get("mu_manifold", 0.0)),
        )
        if not ed.get("valid_edit") or ed.get("edited") is None:
            attempts["editor_fail"] += 1
            continue
        h_e = np.asarray(ed["edited"], dtype=np.float64)
        probs_e, logits_e, audit_e = patch(rec["tokens"], h_e)
        hooks_ok = hooks_ok and int(audit_e.get("ok", 0))
        logit_pack[f"{pair['recip_game']}__{pair['target_game']}__editor_A"] = np.asarray(logits_e, dtype=np.float64)
        probe_e = decoder.proba_np(h_e).reshape(64, 3)
        probe_pack[f"{pair['recip_game']}__{pair['target_game']}__editor_A"] = probe_e
        rows.append({
            "cell": "editor_A", "recip_game": pair["recip_game"], "target_game": pair["target_game"],
            "r2_type": pair["r2_type"], "valid": 1, "board_exact_editor": decoder_state_effects(decoder, h_e, rec["rel_lab"], tgt["rel_lab"])["board_exact"],
            "editor_l2": float(np.linalg.norm(h_e - h_rec)),
        })
        d_n = p_n @ (h_d - h_e)
        target_l2 = float(np.linalg.norm(d_n))
        directions = {"correct_null": d_n}
        for j, wd in enumerate(wrep["donors"]):
            raw = p_n @ (np.asarray(H[int(wd["index"])], dtype=np.float64) - h_e)
            matched = match_l2(raw, target_l2)
            if not matched["ok"]:
                attempts["direction_fail"] += 1
                directions[f"wrong_null_{j}"] = None
            else:
                directions[f"wrong_null_{j}"] = matched["vec"]
        for j, seed in enumerate(cfg.get("random_seeds", RANDOM_SEEDS)):
            rng = np.random.default_rng(int(seed) + int(pair["recip"]))
            matched = match_l2(p_n @ rng.normal(size=p_n.shape[0]), target_l2)
            directions[f"random_null_{j}"] = matched["vec"] if matched["ok"] else None
        y_by = {}
        inv_ok = 1
        js_ok = 1
        pair_tasks = [t for t in tasks if t["recip_game"] == pair["recip_game"] and t["target_game"] == pair["target_game"]]
        for task in pair_tasks:
            delta = directions.get(task["cell"])
            if delta is None:
                attempts["direction_fail"] += 1
                attempts["js_fail"] += 1
                attempts["kept_failures"] += 1
                js_ok = 0
                cal_rows.append({**task, "ok": 0, "status": "degenerate_direction", "reason": "degenerate_direction", "target_js": js_target})
                continue

            def ev(t, _delta=delta):
                h = h_e + float(t) * np.asarray(_delta, dtype=np.float64)
                probs, logits, audit = patch(rec["tokens"], h)
                js = float(js_divergence(probs_e, probs))
                y = y_components(probs, rec["legal"], tgt["legal"])
                dose_rows.append({"cell": task["cell"], "recip_game": pair["recip_game"], "target_game": pair["target_game"], "t": float(t), "js": js, "Y": y["Y"], "target_js": js_target})
                return {"js": js, "probs": probs, "logits": logits, "audit": audit, "y": y, "h": h}

            matched = calibrate_task(task, ev, delta, grid=grid, tol=tol)
            sc = scaled_displacement(delta, matched.get("t", float("nan")))
            probe = decoder.proba_np(h_e + sc["vec"]).reshape(64, 3) if matched.get("t") == matched.get("t") else None
            inv = float(np.max(np.abs(probe - probe_e))) if probe is not None else float("nan")
            cal = {
                **task,
                "ok": matched.get("ok", 0),
                "status": matched.get("status"),
                "reason": matched.get("reason"),
                "target_js": js_target,
                "t": matched.get("t"),
                "js": matched.get("js"),
                "js_abs_err": matched.get("js_abs_err"),
                "l2": sc["l2"] if matched.get("t") == matched.get("t") else float("nan"),
                "base_l2": sc["base_l2"],
                "invariance_max_err": inv,
            }
            cal_rows.append(cal)
            if not matched.get("ok"):
                attempts["js_fail"] += 1
                attempts["kept_failures"] += 1
                js_ok = 0
                rows.append({"cell": task["cell"], "recip_game": pair["recip_game"], "target_game": pair["target_game"], "valid": 0, "reason": matched.get("reason"), "target_js": js_target})
                continue
            if inv > inv_atol:
                inv_ok = 0
            hooks_ok = hooks_ok and int(matched.get("audit", {}).get("ok", 0))
            y = matched["y"]
            y_by[task["cell"]] = y
            key = f"{pair['recip_game']}__{pair['target_game']}__{task['cell']}"
            logit_pack[key] = np.asarray(matched["logits"], dtype=np.float64)
            probe_pack[key] = probe
            dec = decoder_state_effects(decoder, matched["h"], rec["rel_lab"], tgt["rel_lab"])
            rows.append({
                "cell": task["cell"], "r2_type": pair["r2_type"], "recip_game": pair["recip_game"],
                "target_game": pair["target_game"], "donor_game": task.get("donor_id"),
                "valid": 1, "target_js": js_target, "t": matched["t"], "js": matched["js"],
                "l2": sc["l2"], "Y": y["Y"], "G": y["G"], "Q": y["Q"],
                "board_exact": dec["board_exact"], "invariance_max_err": inv,
                "kl_to_editor": kl_row_safe(probs_e, matched["probs"]),
            })
        if not inv_ok:
            attempts["invariance_fail"] += 1
        have = [c for c in CELLS if c in y_by]
        if inv_ok and js_ok and len(have) == 7:
            attempts["valid"] += 1
            y_c = y_by["correct_null"]
            y_w = [y_by[f"wrong_null_{j}"]["Y"] for j in range(3)]
            y_r = [y_by[f"random_null_{j}"]["Y"] for j in range(3)]
            ctr = pair_contrasts(y_c["Y"], y_w, y_r)
            rows.append({
                "cell": "pair_contrast", "r2_type": pair["r2_type"],
                "recip_game": pair["recip_game"], "target_game": pair["target_game"],
                "valid": 1, **ctr,
                "dG_wrong": float(y_c["G"] - np.mean([y_by[f"wrong_null_{j}"]["G"] for j in range(3)])),
                "dQ_wrong": float(y_c["Q"] - np.mean([y_by[f"wrong_null_{j}"]["Q"] for j in range(3)])),
                "board_exact_editor": rows[-8]["board_exact"] if False else decoder_state_effects(decoder, h_e, rec["rel_lab"], tgt["rel_lab"])["board_exact"],
                "target_js": js_target,
            })

    if any(not np.allclose(a, b) for a, b in zip(params0, [p.detach().cpu().numpy() for p in model.parameters()])) or not np.allclose(coef0, decoder.coef):
        raise SystemExit(f"{VERDICT_ENGINEERING} parameters changed")

    ref = reject_donor_reference_rows(cal_rows, js_target)
    if not ref["ok"]:
        raise SystemExit(f"{VERDICT_ENGINEERING} donor-reference leaked {ref['failures'][:6]}")

    contrasts_rows = [r for r in rows if r.get("cell") == "pair_contrast" and r.get("valid") == 1]
    boot_rows = []
    slices = {}
    for name, mask in (("pooled", None), ("A", "A_legal_diff"), ("B", "B_similar_board_legal_change"), ("exact_editor", None), ("nonexact_editor", None)):
        if name == "exact_editor":
            exact_pairs = { (r["recip_game"], r["target_game"]) for r in rows if r.get("cell") == "editor_A" and r.get("board_exact_editor") in {1, True, "1"} }
            g = [r for r in contrasts_rows if (r["recip_game"], r["target_game"]) in exact_pairs]
        elif name == "nonexact_editor":
            exact_pairs = { (r["recip_game"], r["target_game"]) for r in rows if r.get("cell") == "editor_A" and r.get("board_exact_editor") in {1, True, "1"} }
            g = [r for r in contrasts_rows if (r["recip_game"], r["target_game"]) not in exact_pairs]
        else:
            g = contrasts_rows if mask is None else [r for r in contrasts_rows if r.get("r2_type") == mask]
        if not g:
            slices[name] = {"n": 0}
            continue
        sl = {
            "n": len(g),
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

    n_pairs = 1 if args.smoke else 48
    gate = build_fixed_js_gate(attempts, n_pairs, hooks_ok=hooks_ok, parity_ok=int(parity["ok"] == 1), slices=slices, behavior_floor=float(cfg.get("behavior_floor", BEHAVIOR_FLOOR)), coverage_min=1 if args.smoke else int(cfg.get("coverage_min", COVERAGE_MIN)))
    verdict = decide_fixed_js_verdict(gate)
    write_csv(out_dir / "fixed_js_calibrations.csv", cal_rows)
    write_csv(out_dir / "fixed_js_interventions.csv", rows)
    write_csv(out_dir / "fixed_js_dose_search.csv", dose_rows or [{"reason": "no_search", "n": 0}])
    write_csv(out_dir / "fixed_js_bootstrap.csv", boot_rows or [{"reason": "no_valid_contrasts", "n": 0}])
    write_json(out_dir / "fixed_js_attempt_accounting.json", attempts)
    np.savez_compressed(out_dir / "fixed_js_full_logits.npz", **logit_pack)
    np.savez_compressed(out_dir / "fixed_js_probe_probabilities.npz", **probe_pack)
    write_json(out_dir / "fixed_js_dependency_manifest.json", {
        "git_hash": git_hash(), "site": SITE, "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt), "probe_path": probe_info["probe_path"],
        "probe_sha256": sha256_file(probe_info["probe_path"]),
        "pair_manifest_sha256": manifest["pair_manifest_sha256"],
        "js_target": js_target, "js_role": "primary" if abs(js_target - FIXED_JS_PRIMARY) < 1e-12 else "secondary",
        "used_confirmation": 0, "official_0A_letter": OFFICIAL_0A_LETTER,
        "prior_cost_verdict_preserved": "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
        "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
        "wrote_into_original_fiber_cost": 0,
    })
    write_report(out_dir / "fixed_js_scientific_report.md", gate, verdict, slices, attempts, manifest, js_target)
    write_json(out_dir / "fixed_js_completion_status.json", {
        "status": STATUS_COMPLETE if not args.smoke else "SMOKE_COMPLETE",
        "verdict": verdict if not args.smoke else "SMOKE_ONLY",
        "gate": gate, "slices": slices, "attempts": attempts, "preflight": preflight,
        "checkpoint_loaded": 1, "probe_loaded": 1, "pairs_ok": 1, "preflight_ok": 1,
        "full_outputs_saved": int(len(logit_pack) > 0), "js_target": js_target,
        "smoke": int(args.smoke), "used_confirmation": 0,
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "prior_cost_verdict_preserved": "TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE",
        "fiber_spec_status_preserved": "TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED",
        "validation_status": "08R2_VALIDATION_COMPLETE — NEGATIVE",
        "obstruction_ready": "NO",
    })
    if not args.smoke:
        gate_files = evaluate_fixed_js_completeness(out_dir, real_model=True, expected_target=js_target)
        if not gate_files["ok"]:
            raise SystemExit(f"{VERDICT_ENGINEERING} completeness {gate_files['failures']}")
    print("08R2-FIBER-COST-FIXED-JS")
    print(f"  verdict={verdict if not args.smoke else 'SMOKE_ONLY'}")
    print(f"  js_target={js_target} smoke={int(args.smoke)}")
    print(f"  valid_pairs={attempts['valid']}/{n_pairs}")
    print(f"  attempts={attempts}")
    print(f"  gate={gate}")
    print(f"  pooled={slices.get('pooled')}")
    print("  prior COST remains TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE")
    print("  SPEC remains TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED")
    print("  official letter unchanged: C / PARTIAL")
    if args.smoke:
        smoke_ok = int(attempts["valid"] == 1 and attempts["js_fail"] == 0)
        print(f"  smoke_calibrated_all_seven={smoke_ok}")
        if not smoke_ok:
            raise SystemExit(f"{VERDICT_ENGINEERING} smoke failed to calibrate all seven directions")


if __name__ == "__main__":
    main()
