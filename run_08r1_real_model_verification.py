"""08R1 scientific verification on frozen Nanda Othello-GPT.

This runner does not mock or substitute a fake residual model.
Does not change the official 0A letter C / PARTIAL.
Synthetic residual tests belong in tests/ and 08r1_engineering_* files only.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from src.othello_r1 import (
    cluster_bootstrap,
    donor_dependence,
    intervention_probability_metrics,
    kl_row,
    paired_cluster_diff,
)
from src.othello_r1_eval import (
    REQUIRED_REAL_FILES,
    choose_wrong_donors,
    dist_stats,
    evaluate_completion,
    intended_disagreed_squares,
    isolate_r1_results_dir,
    record_hf_revision,
    resolve_probe_path,
    score_hook_event,
    score_written_outputs,
    sha256_file,
    wrong_square_linear_edit,
)
from src.othello_runutil import add_labels, induced_legal, load_split, pair_candidates, tokens_of, write_json
from src.toy_othello import (
    MODEL_SPEC,
    BoardState,
    cache_site,
    hook_name,
    load_config,
    load_othello_gpt,
    patch_site,
    random_untrained_othello,
    run_model_logits,
    softmax_np,
    to_absolute_labels,
    verify_nanda_checkpoint,
)


NOOP_ATOL = 1e-4
DONOR_LOGIT_ATOL = 1e-3
READBACK_ATOL = 1e-5
FINAL_HOOK = "blocks.7.hook_resid_post"
L6_MID = "blocks.6.hook_resid_mid"
L6_POST = "blocks.6.hook_resid_post"
WINNER_HOOK = L6_MID
REQUIRED_HOOKS = (L6_MID, L6_POST, FINAL_HOOK)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R1 real Othello-GPT verification.")
    p.add_argument("--config", default="configs/toy_othello.json")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--probe-path", default=None, help="Exact 08c_selected_probe.pkl. No silent substitute.")
    p.add_argument("--device", default=None)
    p.add_argument("--n-pairs", type=int, default=None)
    p.add_argument("--n-inspect", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=2000, help=">=1000 required; 2000 preferred.")
    p.add_argument("--boot-seed", type=int, default=81)
    p.add_argument("--skip-untrained", action="store_true")
    p.add_argument("--untrained-max-train", type=int, default=None)
    p.add_argument("--untrained-max-test", type=int, default=None)
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Rescore existing 08r1_real_* files. Does not load the model.",
    )
    return p.parse_args()


def _require_real_stack():
    try:
        import torch
        import transformer_lens
    except Exception as exc:
        raise SystemExit(
            "08R1_INCOMPLETE — REAL_MODEL_VERIFICATION_PENDING\n"
            f"Real stack missing ({exc}). This runner does not fall back to a fake model."
        )
    return torch, transformer_lens


def resolve_checkpoint(cfg: dict, explicit: str | None) -> Path | None:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    ckpt_dir = Path(cfg["checkpoints_dir"])
    cands.append(ckpt_dir / cfg.get("checkpoint_file", "synthetic_model.pth"))
    for path in cands:
        if path.exists():
            return path
    return None


def git_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def patch_and_audit(model, token_lists, expected_hook: str, values, batch_size: int):
    """Patch one hook, record hook_obj.name, fire count, shape, last-token-only edit, readback."""
    import torch
    from src.toy_othello import pad_batch

    device = next(model.parameters()).device
    values = values.to(device)
    logits_out = []
    readback = []
    observed_names: list[str] = []
    fire_count = 0
    shapes: list[tuple[int, ...]] = []
    others_ok = True

    with torch.no_grad():
        for start in range(0, len(token_lists), batch_size):
            chunk = token_lists[start : start + batch_size]
            toks, lengths = pad_batch(chunk, device)
            chunk_vals = values[start : start + len(chunk)]
            grabbed = []

            def patch_fn(act, hook, vals=chunk_vals, lens=lengths, expected=expected_hook):
                nonlocal fire_count, others_ok
                hook_obj = hook  # TransformerLens calls hook(act, hook=self)
                name = getattr(hook_obj, "name", None)
                if name is None:
                    raise RuntimeError(f"hook_obj.name missing for expected {expected}")
                observed_names.append(str(name))
                if str(name) != expected:
                    raise RuntimeError(f"hook_obj.name={name!r} != expected {expected!r}")
                fire_count += 1
                shapes.append(tuple(act.shape))
                out = act.clone()
                for i, t in enumerate(lens):
                    out[i, t - 1, :] = vals[i].to(dtype=out.dtype)
                    if t > 1 and not torch.equal(out[i, : t - 1], act[i, : t - 1]):
                        others_ok = False
                    if t < act.shape[1] and not torch.equal(out[i, t:], act[i, t:]):
                        others_ok = False
                return out

            def grab_fn(act, hook, lens=lengths, expected=expected_hook):
                hook_obj = hook  # TransformerLens calls hook(act, hook=self)
                name = getattr(hook_obj, "name", None)
                if str(name) != expected:
                    raise RuntimeError(f"readback hook_obj.name={name!r} != {expected!r}")
                grabbed.append(act.detach())
                return act

            logits = model.run_with_hooks(toks, fwd_hooks=[(expected_hook, patch_fn), (expected_hook, grab_fn)])
            if not grabbed:
                raise RuntimeError(f"expected hook did not execute: {expected_hook}")
            act = grabbed[0]
            readback.append(torch.stack([act[i, t - 1, :].float().cpu() for i, t in enumerate(lengths)], dim=0))
            logits_out.append(torch.stack([logits[i, t - 1, :].float().cpu() for i, t in enumerate(lengths)], dim=0))

    rb = torch.cat(readback, dim=0)
    lg = torch.cat(logits_out, dim=0)
    donor_np = values.detach().cpu().numpy()
    last_shape = shapes[0] if shapes else None
    audit = score_hook_event(
        expected_hook,
        observed_names,
        fire_count,
        tuple(rb.shape),
        tuple(donor_np.shape),
        rb.numpy(),
        donor_np,
        others_ok,
        atol=READBACK_ATOL,
    )
    rank_ok = last_shape is not None and len(last_shape) == 3 and last_shape[-1] == int(donor_np.shape[-1])
    if not rank_ok:
        audit["ok"] = 0
        audit["shape_ok"] = 0
        extra = f"activation_rank_or_width invalid got={last_shape}"
        audit["fail_reason"] = (audit.get("fail_reason") + ";" + extra).strip(";")
    if fire_count <= 0 or not audit["ok"]:
        raise RuntimeError(f"hook audit failed for {expected_hook}: {audit}")
    return lg, rb, audit


def cache_named(model, token_lists, hooks: list[str], batch_size: int):
    import torch
    from src.toy_othello import pad_batch

    device = next(model.parameters()).device
    want = set(hooks)
    collected = {h: [] for h in hooks}
    with torch.no_grad():
        for start in range(0, len(token_lists), batch_size):
            chunk = token_lists[start : start + batch_size]
            toks, lengths = pad_batch(chunk, device)
            _, cache = model.run_with_cache(toks, names_filter=lambda n, w=want: n in w)
            missing = [h for h in hooks if h not in cache]
            if missing:
                raise RuntimeError(f"hooks did not fire: {missing}")
            for h in hooks:
                rows = [cache[h][i, t - 1, :].float().cpu() for i, t in enumerate(lengths)]
                collected[h].append(torch.stack(rows, dim=0))
    return {h: torch.cat(v, dim=0) for h, v in collected.items()}


def _metric_row(pb, pa, rec, don) -> dict:
    row = intervention_probability_metrics(pb, pa, rec["legal"], don["legal"])
    row["kl"] = kl_row(pb, pa)
    row["total_legal_mass_target_after"] = row["donor_legal_mass_after"]
    row["mean_probability_per_move_target_after"] = row["old_mean_donor_legal_after"]
    row["became_legal_delta_mass"] = row["became_legal_d_mass"]
    row["became_illegal_delta_mass"] = row["became_illegal_d_mass"]
    row["unchanged_legal_delta_mass"] = row["stayed_legal_d_mass"]
    row["unchanged_illegal_delta_mass"] = row["stayed_illegal_d_mass"]
    return row


def _bootstrap_table(method, metric, values, recip, donor, n_boot, seed, extra):
    rec = list(recip)
    dep = donor_dependence(rec, list(donor))
    pair_ids = [f"{r}|{d}" for r, d in zip(rec, donor)]
    donor_ids = list(donor)
    primary = cluster_bootstrap(np.asarray(values, dtype=float), rec, n_boot, seed)
    pair_ci = cluster_bootstrap(np.asarray(values, dtype=float), pair_ids, n_boot, seed + 1)
    donor_ci = cluster_bootstrap(np.asarray(values, dtype=float), donor_ids, n_boot, seed + 2)
    return {
        "method": method,
        "metric": metric,
        "cluster": "recipient_game",
        **primary,
        "ci95_low_recip_donor": pair_ci["ci95_low"],
        "ci95_high_recip_donor": pair_ci["ci95_high"],
        "ci95_low_donor_game": donor_ci["ci95_low"],
        "ci95_high_donor_game": donor_ci["ci95_high"],
        "n_pairs": dep["n_pairs"],
        "n_distinct_recipient_games": dep["n_distinct_recipient_games"],
        "n_distinct_donor_games": dep["n_distinct_donor_games"],
        "donor_reuse": int(dep["donor_reuse"]),
        "independence_claim": "none" if dep["donor_reuse"] or dep["n_distinct_recipient_games"] < dep["n_pairs"] else "pairs_unique_games",
        "provenance": "real_model",
        **extra,
    }


def _untrained_decode(cfg, device, train, test, hook, max_train, max_test) -> dict:
    from sklearn.linear_model import LogisticRegression

    tr = train[:max_train] if max_train else train
    te = test[:max_test] if max_test else test
    print(f"  untrained cache {hook} n_train={len(tr)} n_test={len(te)}")
    umodel = random_untrained_othello(0, device=device)
    Xtr = cache_site(umodel, tokens_of(tr), hook, cfg["cache_batch"]).numpy()
    Xte = cache_site(umodel, tokens_of(te), hook, cfg["cache_batch"]).numpy()
    ytr = np.stack([r["rel_lab"] for r in tr])
    yte = np.stack([r["rel_lab"] for r in te])
    pred = np.zeros_like(yte)
    for j in range(64):
        clf = LogisticRegression(C=float(cfg["linear_c"]), max_iter=200)
        clf.fit(Xtr, ytr[:, j])
        pred[:, j] = clf.predict(Xte)
    exact = float((pred == yte).all(axis=1).mean())
    square = float((pred == yte).mean())
    print(f"    exact={exact:.6f} square={square:.6f}")
    return {
        "control": "untrained_othello_gpt",
        "hook": hook,
        "status": "PASS",
        "exact_board": exact,
        "square_acc": square,
        "n_train": len(tr),
        "n_test": len(te),
        "provenance": "real_untrained_othello_gpt",
        "note": "Architecture-matched random-init Othello-GPT after CPU-draw generator fix.",
    }


def _probe_consistency(pack, X, test) -> list[dict]:
    from src.toy_othello import relative_to_absolute, SYMMETRY_NAMES, transform_board, transform_square, legal_squares

    models = pack["models"]
    target = pack["target"]
    pred_lab = np.stack([m.predict(X) for m in models], axis=1)
    if target == "rel":
        pred_abs = np.stack([relative_to_absolute(pred_lab[i], int(test[i]["player"])) for i in range(len(test))])
    else:
        pred_abs = pred_lab
    jacc = []
    exact = []
    for i, r in enumerate(test):
        pred_legal = set(induced_legal(pred_abs[i], int(r["player"])))
        true_legal = set(r["legal"])
        union = pred_legal | true_legal
        jacc.append(len(pred_legal & true_legal) / len(union) if union else 1.0)
        exact.append(float((pred_abs[i] == r["abs_lab"]).all()))
    trans = []
    by_game = defaultdict(list)
    for i, r in enumerate(test):
        by_game[r["game_id"]].append(i)
    for items in by_game.values():
        items = sorted(items, key=lambda i: test[i]["t"])
        for a, b in zip(items, items[1:]):
            if test[b]["t"] != test[a]["t"] + 1:
                continue
            move = test[a].get("next_square")
            if move is None:
                continue
            from src.toy_othello import labels_to_board

            board_t = labels_to_board(pred_abs[a])
            st = BoardState(board_t.copy(), int(test[a]["player"]))
            if int(move) not in st.legal_moves():
                continue
            stepped = st.play(int(move)).board
            dec_next = labels_to_board(pred_abs[b])
            trans.append(float((to_absolute_labels(stepped) == to_absolute_labels(dec_next)).mean()))
    rel_ok = []
    if target == "rel":
        for i, r in enumerate(test):
            back = relative_to_absolute(pred_lab[i], int(r["player"]))
            rel_ok.append(float((back == pred_abs[i]).mean()))
    sym = []
    from src.toy_othello import labels_to_board

    for i, r in enumerate(test[:200]):
        board = labels_to_board(pred_abs[i])
        player = int(r["player"])
        base_legal = set(legal_squares(board, player))
        for name in SYMMETRY_NAMES:
            tb = transform_board(board, name)
            tlegal = {transform_square(s, name) for s in base_legal}
            sym.append(int(tlegal == set(legal_squares(tb, player))))
    n_games = len({r["game_id"] for r in test})
    return [
        {
            "test": "decoded_vs_true_legal_jaccard",
            "status": "PASS",
            **{k: v for k, v in dist_stats(jacc).items() if k != "n"},
            "n_positions": len(jacc),
            "n_games": n_games,
            "left": "induced_legal(decoded_board)",
            "right": "simulator legal set on true board",
            "used_probe": 1,
            "used_model_activations": 1,
            "provenance": "real_model",
        },
        {
            "test": "decoded_transition_F_vs_next_decode",
            "status": "PASS" if trans else "FAIL",
            **{k: v for k, v in dist_stats(trans).items() if k != "n"},
            "n_positions": len(trans),
            "n_games": n_games,
            "left": "simulator.play(decoded_board_t, true_next_square)",
            "right": "independently decoded board at t+1",
            "used_probe": 1,
            "used_model_activations": 1,
            "provenance": "real_model",
        },
        {
            "test": "whole_board_exact",
            "status": "PASS",
            **{k: v for k, v in dist_stats(exact).items() if k != "n"},
            "n_positions": len(exact),
            "n_games": n_games,
            "left": "decoded absolute labels",
            "right": "true absolute labels",
            "used_probe": 1,
            "used_model_activations": 1,
            "provenance": "real_model",
        },
        {
            "test": "player_relative_alignment",
            "status": "PASS" if rel_ok else "NOT_APPLICABLE",
            **{k: v for k, v in dist_stats(rel_ok).items() if k != "n"},
            "n_positions": len(rel_ok),
            "n_games": n_games,
            "left": "relative_to_absolute(decoded_rel)",
            "right": "decoded abs used for Jaccard",
            "used_probe": 1,
            "used_model_activations": 1,
            "provenance": "real_model",
        },
        {
            "test": "decoded_board_symmetry_legal",
            "status": "PASS" if sym else "NOT_APPLICABLE",
            **{k: v for k, v in dist_stats(sym).items() if k != "n"},
            "n_positions": len(sym),
            "n_games": min(n_games, 200),
            "left": "transform(decoded legal set)",
            "right": "legal_squares(transform(decoded board))",
            "used_probe": 1,
            "used_model_activations": 1,
            "provenance": "real_model",
            "note": "Property of the decoded board, not a new model forward.",
        },
        {
            "test": "legalizer_self_consistent",
            "status": "NOT_APPLICABLE",
            "mean": float("nan"),
            "n_positions": 0,
            "n_games": 0,
            "left": "removed",
            "right": "removed",
            "used_probe": 0,
            "used_model_activations": 0,
            "provenance": "excluded_tautology",
            "note": "Tautological 08d row is not evidence.",
        },
    ]


def write_status(path: Path, payload: dict) -> None:
    write_json(path, payload)
    print(json.dumps({k: payload.get(k) for k in ("scientific_decision", "nanda_argmax_match", "failed_required")}, indent=2))


def validate_only(args) -> None:
    """Re-run the completion validator on already-written real-model files."""
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    isolation = isolate_r1_results_dir(cfg["results_dir"])
    results_dir = Path(isolation["results_dir"])
    status_path = results_dir / "08r1_real_model_status.json"
    if not status_path.exists():
        raise SystemExit(f"08R1_INCOMPLETE missing {status_path}")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    stages = payload.get("stages")
    if not stages:
        raise SystemExit("08R1_INCOMPLETE status JSON has no stages")
    completion = score_written_outputs(results_dir, stages)
    payload["scientific_decision"] = completion["decision"]
    payload["completion"] = completion
    payload["validator"] = "validate_only"
    payload["official_0A_letter"] = "C / PARTIAL"
    write_status(status_path, payload)
    print("=== stages ===")
    for k, v in stages.items():
        print(f"  {k}={v}")
    print(f"numeric_ok={completion.get('numeric_ok')} {completion.get('numeric_reason')}")
    print(f"scientific_decision={completion['decision']}")
    print("Official 0A letter unchanged: C / PARTIAL")
    if completion["decision"] != "08R1_COMPLETE":
        raise SystemExit(completion["decision"])
    print("08R1_COMPLETE")


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_only(args)
        return
    torch, transformer_lens = _require_real_stack()
    if int(args.n_boot) < 1000:
        raise SystemExit("08R1 scientific run requires --n-boot >= 1000")
    t0 = time.time()
    cfg = load_config(args.config)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.checkpoints_dir:
        cfg["checkpoints_dir"] = args.checkpoints_dir
    isolation = isolate_r1_results_dir(cfg["results_dir"])
    if isolation["collisions"]:
        raise SystemExit("REFUSING to overwrite historical 08a-08i artifacts: " + ",".join(isolation["collisions"]))
    results_dir = ensure_dir(isolation["results_dir"])
    print(f"  results_dir={results_dir} requested={isolation['requested']} nested={isolation['nested']}")
    status_path = Path(results_dir) / "08r1_real_model_status.json"
    n_pairs = int(args.n_pairs or cfg["n_pairs"])
    n_boot = int(args.n_boot)
    device = args.device or cfg.get("device")
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    stages = {s: "PENDING" for s in (
        "checkpoint_identity",
        "hook_blocks.6.hook_resid_mid",
        "hook_blocks.6.hook_resid_post",
        "hook_blocks.7.hook_resid_post",
        "noop_identity",
        "final_resid_donor_logit",
        "mid_post_readback",
        "corrected_probability_metrics",
        "probe_consistency_08d",
        "wrong_donor_control",
        "wrong_square_control",
        "untrained_control",
        "bootstrap_cis",
    )}

    print("=== 08R1 real-model verification ===")
    print(f"  checkpoint_path={ckpt} n_boot={n_boot}")
    try:
        model = load_othello_gpt(device=device, checkpoint_path=ckpt)
    except Exception as exc:
        stages["checkpoint_identity"] = "FAIL"
        write_status(status_path, {"scientific_decision": "REAL_MODEL_VALIDATION_FAILED", "reason": str(exc), **evaluate_completion(stages, {})})
        raise SystemExit("FAILED: checkpoint load; no substitute checkpoint is used")

    fixture = verify_nanda_checkpoint(model)
    print(f"  Nanda fixture match={fixture['nanda_argmax_match']} mismatches={fixture['n_mismatch']}")
    if not fixture["nanda_argmax_match"]:
        stages["checkpoint_identity"] = "FAIL"
        write_status(status_path, {"scientific_decision": "REAL_MODEL_VALIDATION_FAILED", "reason": "fixture", **evaluate_completion(stages, {})})
        raise SystemExit("FAILED: checkpoint identity")
    stages["checkpoint_identity"] = "PASS"

    probe_info = resolve_probe_path(args.probe_path, cfg["checkpoints_dir"])
    probe_path = Path(probe_info["probe_path"])
    probe_hash = sha256_file(probe_path)
    pack = None
    print(
        f"  probe_path={probe_path} exists={probe_info['exists']} "
        f"source={probe_info['source']} fallback_used={probe_info['fallback_used']}"
    )
    if probe_info["exists"]:
        with probe_path.open("rb") as handle:
            pack = pickle.load(handle)

    test = add_labels(load_split(Path(cfg["data_dir"]), "test"))
    train = add_labels(load_split(Path(cfg["data_dir"]), "train"))
    pairs = pair_candidates(test, n_pairs, int(cfg["seed"]) + 17)
    if len(pairs) < 8:
        write_status(status_path, {"scientific_decision": "REAL_MODEL_VALIDATION_FAILED", "reason": "too few pairs"})
        raise SystemExit("FAILED: too few pairs")
    rec_rows = [test[p["recip"]] for p in pairs]
    don_rows = [test[p["donor"]] for p in pairs]
    rec_tok = tokens_of(rec_rows)
    don_tok = tokens_of(don_rows)
    rec_games = [r["game_id"] for r in rec_rows]
    don_games = [d["game_id"] for d in don_rows]
    dep = donor_dependence(rec_games, don_games)
    inspect_n = min(int(args.n_inspect), len(pairs))
    batch = int(cfg["cache_batch"])

    print("=== required hook audits ===")
    hook_rows = []
    cached = cache_named(model, rec_tok[:inspect_n], [L6_MID, L6_POST, FINAL_HOOK], batch)
    cached_d = cache_named(model, don_tok[:inspect_n], [L6_MID, L6_POST, FINAL_HOOK], batch)
    base_all = run_model_logits(model, rec_tok, cfg["batch_size"])
    don_logits = run_model_logits(model, don_tok, cfg["batch_size"])
    rec_final = cache_site(model, rec_tok, FINAL_HOOK, batch)
    don_final = cache_site(model, don_tok, FINAL_HOOK, batch)

    noop_z, noop_rb, noop_audit = patch_and_audit(model, rec_tok, FINAL_HOOK, rec_final, batch)
    hook_rows.append({**noop_audit, "check": "noop_final_resid", "hook": FINAL_HOOK})
    noop_diff = (noop_z - base_all).abs()
    stages["noop_identity"] = "PASS" if bool(torch.all(noop_diff <= NOOP_ATOL)) and noop_audit["ok"] else "FAIL"
    stages["hook_blocks.7.hook_resid_post"] = "PASS" if noop_audit["ok"] else "FAIL"

    don_z, don_rb, don_audit = patch_and_audit(model, rec_tok, FINAL_HOOK, don_final, batch)
    hook_rows.append({**don_audit, "check": "final_resid_donor_patch", "hook": FINAL_HOOK})
    donor_logit_diff = (don_z - don_logits).abs()
    stages["final_resid_donor_logit"] = (
        "PASS" if bool(torch.all(donor_logit_diff <= DONOR_LOGIT_ATOL)) and don_audit["ok"] else "FAIL"
    )

    mid_z, mid_rb, mid_audit = patch_and_audit(model, rec_tok[:inspect_n], L6_MID, cached_d[L6_MID], batch)
    post_z, post_rb, post_audit = patch_and_audit(model, rec_tok[:inspect_n], L6_POST, cached_d[L6_POST], batch)
    hook_rows.append({**mid_audit, "check": "l6_mid_patch", "hook": mid_audit["csv_hook"] or L6_MID})
    hook_rows.append({**post_audit, "check": "l6_post_patch", "hook": post_audit["csv_hook"] or L6_POST})
    stages["hook_blocks.6.hook_resid_mid"] = "PASS" if mid_audit["ok"] else "FAIL"
    stages["hook_blocks.6.hook_resid_post"] = "PASS" if post_audit["ok"] else "FAIL"
    stages["mid_post_readback"] = "PASS" if mid_audit["ok"] and post_audit["ok"] else "FAIL"

    mid_post_act = torch.norm(cached[L6_MID] - cached[L6_POST], dim=-1)
    mid_post_logit = (mid_z - post_z).abs()
    print(
        f"  mid/post act L2 mean={float(mid_post_act.mean()):.6f} "
        f"logit max-abs mean={float(mid_post_logit.mean()):.6f} "
        f"identical_tensors={bool(torch.allclose(cached[L6_MID], cached[L6_POST], atol=1e-6))}"
    )

    print("=== mid/post diagnostics (inspect + aggregate) ===")
    pb_i = softmax_np(base_all[:inspect_n])
    diag = []
    for i in range(inspect_n):
        rec, don = rec_rows[i], don_rows[i]
        pa_m = softmax_np(mid_z[i].unsqueeze(0))[0]
        pa_p = softmax_np(post_z[i].unsqueeze(0))[0]
        mm = _metric_row(pb_i[i], pa_m, rec, don)
        mp = _metric_row(pb_i[i], pa_p, rec, don)
        diag.append(
            {
                "pair_index": i,
                "recip_game": rec["game_id"],
                "donor_game": don["game_id"],
                "recip_mid_norm": float(torch.norm(cached[L6_MID][i])),
                "recip_post_norm": float(torch.norm(cached[L6_POST][i])),
                "donor_mid_norm": float(torch.norm(cached_d[L6_MID][i])),
                "donor_post_norm": float(torch.norm(cached_d[L6_POST][i])),
                "recip_mid_post_l2": float(torch.norm(cached[L6_MID][i] - cached[L6_POST][i])),
                "donor_mid_post_l2": float(torch.norm(cached_d[L6_MID][i] - cached_d[L6_POST][i])),
                "patched_mid_minus_donor_mid_l2": float(torch.norm(mid_rb[i] - cached_d[L6_MID][i])),
                "patched_post_minus_donor_post_l2": float(torch.norm(post_rb[i] - cached_d[L6_POST][i])),
                "logit_max_abs_mid_vs_post": float((mid_z[i] - post_z[i]).abs().max()),
                "logit_l2_mid_vs_post": float(torch.norm(mid_z[i] - post_z[i])),
                "became_legal_delta_mass_mid": mm["became_legal_delta_mass"],
                "became_legal_delta_mass_post": mp["became_legal_delta_mass"],
                "became_legal_mass_mid_minus_post": mm["became_legal_delta_mass"] - mp["became_legal_delta_mass"],
                "donor_legal_mass_lift_mid": mm["donor_legal_mass_lift"],
                "donor_legal_mass_lift_post": mp["donor_legal_mass_lift"],
                "old_mean_lift_mid": mm["old_mean_donor_legal_lift"],
                "old_mean_lift_post": mp["old_mean_donor_legal_lift"],
                "executed_mid_hook": mid_audit["csv_hook"],
                "executed_post_hook": post_audit["csv_hook"],
                "identical_activation_tensors": int(bool(torch.allclose(cached[L6_MID][i], cached[L6_POST][i], atol=1e-6))),
                "identical_logits": int(bool(torch.allclose(mid_z[i], post_z[i], atol=1e-6))),
                "provenance": "real_model",
            }
        )
    # hypothesis flags from inspect
    if all(r["identical_activation_tensors"] for r in diag):
        midpost_cause = "C_duplicate_activation_tensors"
    elif all(r["executed_mid_hook"] == r["executed_post_hook"] for r in diag):
        midpost_cause = "D_or_E_hook_name_collision_or_label"
    elif all(abs(r["became_legal_mass_mid_minus_post"]) < 1e-12 for r in diag) and not all(r["identical_logits"] for r in diag):
        midpost_cause = "B_metric_aggregation"
    elif all(r["identical_logits"] for r in diag):
        midpost_cause = "A_or_F_identical_behavior_or_impl"
    else:
        midpost_cause = "A_distinguishable_or_mixed"
    print(f"  mid/post cause_code={midpost_cause} (inspect n={inspect_n}; not a rounded-mean claim)")
    diag.append(
        {
            "pair_index": -1,
            "recip_game": "AGGREGATE_INSPECT",
            "donor_game": "AGGREGATE_INSPECT",
            "recip_mid_post_l2_mean": float(np.mean([r["recip_mid_post_l2"] for r in diag])),
            "donor_mid_post_l2_mean": float(np.mean([r["donor_mid_post_l2"] for r in diag])),
            "logit_l2_mid_vs_post_mean": float(np.mean([r["logit_l2_mid_vs_post"] for r in diag])),
            "became_legal_mass_mid_minus_post_mean": float(np.mean([r["became_legal_mass_mid_minus_post"] for r in diag])),
            "became_legal_mass_mid_minus_post_maxabs": float(np.max(np.abs([r["became_legal_mass_mid_minus_post"] for r in diag]))),
            "n_identical_activation_tensors": int(sum(r["identical_activation_tensors"] for r in diag)),
            "n_identical_logits": int(sum(r["identical_logits"] for r in diag)),
            "n_inspect": inspect_n,
            "cause_code": midpost_cause,
            "note": "Aggregate of inspect examples. Equal rounded means are not treated as equality.",
            "provenance": "real_model",
        }
    )

    print("=== regenerate 08e corrected metrics ===")
    pb = softmax_np(base_all)
    metric_rows = []
    for layer in list(cfg["probe_layers"]):
        for site in ("resid_mid", "resid_post"):
            hook = hook_name(site, layer)
            rec_h = cache_site(model, rec_tok, hook, batch)
            don_h = cache_site(model, don_tok, hook, batch)
            patched = patch_site(model, rec_tok, hook, don_h, batch)
            pa = softmax_np(patched)
            for i, p in enumerate(pairs):
                mets = _metric_row(pb[i], pa[i], rec_rows[i], don_rows[i])
                metric_rows.append(
                    {
                        "source": "08e_regenerated",
                        "method": f"full_patch_{site}",
                        "hook": hook,
                        "layer": layer,
                        "site": site,
                        "type": p["type"],
                        "recip_game": rec_rows[i]["game_id"],
                        "donor_game": don_rows[i]["game_id"],
                        "pert_norm": float(torch.norm(don_h[i] - rec_h[i]).item()),
                        "provenance": "real_model",
                        **mets,
                    }
                )
            sub = [r for r in metric_rows if r["hook"] == hook]
            print(
                f"  {hook} mass_lift={np.nanmean([r['donor_legal_mass_lift'] for r in sub]):+.6f} "
                f"old_mean={np.nanmean([r['old_mean_donor_legal_lift'] for r in sub]):+.6f} "
                f"became_legal_d_mass={np.nanmean([r['became_legal_delta_mass'] for r in sub]):+.6f}"
            )
    finite = [r for r in metric_rows if r["became_legal_delta_mass"] == r["became_legal_delta_mass"]]
    stages["corrected_probability_metrics"] = "PASS" if finite else "FAIL"

    print("=== wrong-donor control (target specificity of legal-set change) ===")
    wrong_idx = choose_wrong_donors(test, pairs, int(cfg["seed"]) + 99)
    control_rows = []
    if sum(i is not None for i in wrong_idx) < 8:
        stages["wrong_donor_control"] = "FAIL"
        print("  too few wrong-donor matches")
    else:
        w_rows = [test[i] if i is not None else don_rows[k] for k, i in enumerate(wrong_idx)]
        w_tok = tokens_of(w_rows)
        w_h = cache_site(model, w_tok, FINAL_HOOK, batch)
        w_z, _, w_audit = patch_and_audit(model, rec_tok, FINAL_HOOK, w_h, batch)
        w_pa = softmax_np(w_z)
        for i, p in enumerate(pairs):
            if wrong_idx[i] is None:
                continue
            mets_true = _metric_row(pb[i], softmax_np(don_z[i].unsqueeze(0))[0], rec_rows[i], don_rows[i])
            mets_wrong = _metric_row(pb[i], w_pa[i], rec_rows[i], don_rows[i])
            control_rows.append(
                {
                    "control": "wrong_donor",
                    "control_kind": "target_specificity",
                    "hook": FINAL_HOOK,
                    "recip_game": rec_rows[i]["game_id"],
                    "true_donor_game": don_rows[i]["game_id"],
                    "wrong_donor_game": w_rows[i]["game_id"],
                    "became_legal_delta_mass_true_target": mets_true["became_legal_delta_mass"],
                    "became_legal_delta_mass_wrong_donor_on_true_target": mets_wrong["became_legal_delta_mass"],
                    "executed_hook": w_audit["csv_hook"],
                    "provenance": "real_model",
                    "natural_state": 1,
                }
            )
        stages["wrong_donor_control"] = "PASS" if control_rows and w_audit["ok"] else "FAIL"

    print("=== random same-norm residual (general output damage, not target specificity) ===")
    rng_noise = np.random.default_rng(int(cfg["seed"]) + 123)
    rec_np = rec_final.cpu().numpy()
    don_np = don_final.cpu().numpy()
    noise = []
    for i in range(len(pairs)):
        step = don_np[i] - rec_np[i]
        want = float(np.linalg.norm(step))
        vec = rng_noise.normal(size=rec_np[i].shape)
        nrm = float(np.linalg.norm(vec))
        if nrm <= 1e-12 or want <= 0:
            noise.append(rec_np[i])
        else:
            noise.append(rec_np[i] + vec * (want / nrm))
    n_z = patch_site(model, rec_tok, FINAL_HOOK, torch.tensor(np.stack(noise), dtype=torch.float32), batch)
    n_pa = softmax_np(n_z)
    for i in range(len(pairs)):
        mets_true = _metric_row(pb[i], softmax_np(don_z[i].unsqueeze(0))[0], rec_rows[i], don_rows[i])
        mets_noise = _metric_row(pb[i], n_pa[i], rec_rows[i], don_rows[i])
        control_rows.append(
            {
                "control": "random_same_norm",
                "control_kind": "general_output_damage",
                "hook": FINAL_HOOK,
                "recip_game": rec_rows[i]["game_id"],
                "became_legal_delta_mass_true_target": mets_true["became_legal_delta_mass"],
                "became_legal_delta_mass_random_same_norm": mets_noise["became_legal_delta_mass"],
                "kl_true_donor": mets_true["kl"],
                "kl_random_same_norm": mets_noise["kl"],
                "provenance": "real_model",
                "natural_state": 0,
                "note": "Tests whether any matched-norm residual change damages output. Not a specificity claim.",
            }
        )

    print("=== wrong-square probe edit (target specificity; not a natural board claim) ===")
    if pack is None:
        stages["wrong_square_control"] = "BLOCKED_MISSING_ARTIFACT"
        control_rows.append(
            {
                "control": "wrong_square",
                "status": "BLOCKED_MISSING_ARTIFACT",
                "artifact": str(probe_path),
                "control_kind": "target_specificity",
                "natural_state": 0,
                "provenance": "blocked",
            }
        )
        print(f"  BLOCKED_MISSING_ARTIFACT {probe_path}")
    else:
        from run_08f_othello_state_causal_edits import linear_edit

        key = "rel_lab" if pack["target"] == "rel" else "abs_lab"
        hook = pack["hook"]
        rec_h = cache_site(model, rec_tok, hook, batch).numpy()
        budget = float(cfg["edit_norm_budget"])
        rng = np.random.default_rng(7)
        intended_H = []
        wrong_H = []
        meta = []
        for i in range(len(pairs)):
            intended_H.append(
                linear_edit(rec_h[i], rec_rows[i][key], don_rows[i][key], pack["models"], budget)
            )
            wr = wrong_square_linear_edit(rec_h[i], rec_rows[i][key], don_rows[i][key], pack["models"], budget, rng)
            if wr is None:
                wrong_H.append(rec_h[i])
                meta.append(None)
            else:
                # match intended intervention norm
                step = wr["edited"] - rec_h[i]
                want = float(np.linalg.norm(intended_H[-1] - rec_h[i]))
                sn = float(np.linalg.norm(step))
                if sn > 1e-12 and want > 0:
                    step = step * (want / sn)
                wrong_H.append(rec_h[i] + step)
                meta.append({**wr, "edit_norm": float(np.linalg.norm(step))})
        if sum(m is not None for m in meta) < 8:
            stages["wrong_square_control"] = "FAIL"
        else:
            z_int = patch_site(model, rec_tok, hook, torch.tensor(np.stack(intended_H), dtype=torch.float32), batch)
            z_w = patch_site(model, rec_tok, hook, torch.tensor(np.stack(wrong_H), dtype=torch.float32), batch)
            pa_i = softmax_np(z_int)
            pa_w = softmax_np(z_w)
            for i in range(len(pairs)):
                if meta[i] is None:
                    continue
                mt = _metric_row(pb[i], pa_i[i], rec_rows[i], don_rows[i])
                mw = _metric_row(pb[i], pa_w[i], rec_rows[i], don_rows[i])
                control_rows.append(
                    {
                        "control": "wrong_square",
                        "control_kind": "target_specificity",
                        "hook": hook,
                        "recip_game": rec_rows[i]["game_id"],
                        "wrong_square": meta[i]["wrong_square"],
                        "intended_squares": "|".join(map(str, meta[i]["intended_squares"])),
                        "edit_norm": meta[i]["edit_norm"],
                        "became_legal_delta_mass_intended": mt["became_legal_delta_mass"],
                        "became_legal_delta_mass_wrong_square": mw["became_legal_delta_mass"],
                        "natural_state": 0,
                        "probe_path": str(probe_path),
                        "probe_sha256": probe_hash,
                        "probe_hook": pack["hook"],
                        "probe_target": pack["target"],
                        "provenance": "real_model",
                    }
                )
            stages["wrong_square_control"] = "PASS"

    print("=== 08d probe consistency ===")
    if pack is None:
        stages["probe_consistency_08d"] = "BLOCKED_MISSING_ARTIFACT"
        cons_rows = [
            {
                "test": "all_08d_probe_comparisons",
                "status": "BLOCKED_MISSING_ARTIFACT",
                "mean": float("nan"),
                "n_positions": 0,
                "n_games": 0,
                "artifact": str(probe_path),
                "left": "required 08c_selected_probe.pkl",
                "right": "missing",
                "used_probe": 0,
                "used_model_activations": 0,
                "note": "Required. No Gaussian/arbitrary probe substitute.",
            }
        ]
        print(f"  BLOCKED_MISSING_ARTIFACT {probe_path}")
    else:
        Xte = cache_site(model, tokens_of(test), pack["hook"], batch).numpy()
        cons_rows = _probe_consistency(pack, Xte, test)
        stages["probe_consistency_08d"] = "PASS" if all(r["status"] in {"PASS", "NOT_APPLICABLE"} for r in cons_rows) else "FAIL"

    print("=== untrained control ===")
    untrained_rows = []
    if args.skip_untrained:
        stages["untrained_control"] = "PENDING"
        untrained_rows.append({"control": "untrained_othello_gpt", "status": "PENDING", "note": "--skip-untrained is not a scientific pass"})
    else:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            untrained_rows.append(_untrained_decode(cfg, device, train, test, WINNER_HOOK, args.untrained_max_train, args.untrained_max_test))
            untrained_rows.append(
                _untrained_decode(cfg, device, train, test, str(cfg["primary_site"]), args.untrained_max_train, args.untrained_max_test)
            )
            stages["untrained_control"] = "PASS"
        except Exception as exc:
            stages["untrained_control"] = "FAIL"
            untrained_rows.append({"control": "untrained_othello_gpt", "status": "FAIL", "note": str(exc)})

    print("=== bootstrap ===")
    boot_rows = []
    for hook in sorted({r["hook"] for r in metric_rows}):
        sub = [r for r in metric_rows if r["hook"] == hook]
        rec = [r["recip_game"] for r in sub]
        don = [r["donor_game"] for r in sub]
        extra = {"source": "08e_regenerated", "n_boot": n_boot, "seed": args.boot_seed}
        for metric in (
            "became_legal_delta_mass",
            "became_illegal_delta_mass",
            "donor_legal_mass_lift",
            "pert_norm",
            "kl",
            "old_mean_donor_legal_lift",
        ):
            boot_rows.append(_bootstrap_table(hook, metric, [r[metric] for r in sub], rec, don, n_boot, args.boot_seed, extra))
    l6m = [r for r in metric_rows if r["hook"] == L6_MID]
    l6p = [r for r in metric_rows if r["hook"] == L6_POST]
    if l6m and l6p:
        rec = [r["recip_game"] for r in l6m]
        boot_rows.append(
            {
                **paired_cluster_diff(
                    np.array([r["became_legal_delta_mass"] for r in l6m]),
                    np.array([r["became_legal_delta_mass"] for r in l6p]),
                    rec,
                    n_boot,
                    args.boot_seed + 5,
                ),
                "method": "L6_resid_mid_minus_resid_post",
                "metric": "became_legal_delta_mass",
                "cluster": "recipient_game",
                "provenance": "real_model",
            }
        )
    stages["bootstrap_cis"] = "PASS" if boot_rows and all(r.get("ci95_low") == r.get("ci95_low") for r in boot_rows if "ci95_low" in r) else "FAIL"

    write_csv(Path(results_dir) / "08r1_real_hook_verification.csv", hook_rows)
    write_csv(Path(results_dir) / "08r1_real_mid_post_diagnostics.csv", diag)
    write_csv(Path(results_dir) / "08r1_real_corrected_probability_metrics.csv", metric_rows)
    write_csv(Path(results_dir) / "08r1_real_negative_controls.csv", control_rows)
    write_csv(Path(results_dir) / "08r1_real_consistency_checks.csv", cons_rows)
    write_csv(Path(results_dir) / "08r1_real_untrained_control.csv", untrained_rows)
    write_csv(Path(results_dir) / "08r1_real_bootstrap_results.csv", boot_rows)
    np.savez_compressed(
        Path(results_dir) / "08r1_real_activation_tensors.npz",
        **{f"recip_mid_{i}": cached[L6_MID][i].cpu().numpy() for i in range(inspect_n)},
        **{f"recip_post_{i}": cached[L6_POST][i].cpu().numpy() for i in range(inspect_n)},
        **{f"donor_mid_{i}": cached_d[L6_MID][i].cpu().numpy() for i in range(inspect_n)},
        **{f"donor_post_{i}": cached_d[L6_POST][i].cpu().numpy() for i in range(inspect_n)},
        **{f"patched_mid_{i}": mid_rb[i].cpu().numpy() for i in range(inspect_n)},
        **{f"patched_post_{i}": post_rb[i].cpu().numpy() for i in range(inspect_n)},
        logits_base=base_all[:inspect_n].cpu().numpy(),
        logits_mid=mid_z.cpu().numpy(),
        logits_post=post_z.cpu().numpy(),
    )

    # Status must exist before scoring so the required-file check is not tautologically incomplete.
    write_json(status_path, {"scientific_decision": "PENDING", "stages": stages, "official_0A_letter": "C / PARTIAL"})
    completion = score_written_outputs(Path(results_dir), stages)
    numeric_ok = completion.get("numeric_ok", False)
    if not numeric_ok:
        print(f"  numeric/schema check failed: {completion.get('numeric_reason')}")
    data_hashes = {
        "prefixes_train": sha256_file(Path(cfg["data_dir"]) / "prefixes_train.jsonl"),
        "prefixes_val": sha256_file(Path(cfg["data_dir"]) / "prefixes_val.jsonl"),
        "prefixes_test": sha256_file(Path(cfg["data_dir"]) / "prefixes_test.jsonl"),
        "games": sha256_file(Path(cfg["data_dir"]) / "games.json"),
    }
    summary = {
        "scientific_decision": completion["decision"],
        "completion": completion,
        "stages": stages,
        "nanda_argmax_match": True,
        "n_mismatch": fixture["n_mismatch"],
        "checkpoint_path": str(ckpt) if ckpt else f"HF:{MODEL_SPEC['hf_repo']}/{MODEL_SPEC['checkpoint_file']}",
        "checkpoint_sha256": sha256_file(ckpt) if ckpt else None,
        "checkpoint_identity_note": "Nanda argmax fixture is behavioral, not a cryptographic identity proof."
        if ckpt is None or sha256_file(ckpt)
        else "fixture only",
        "hf_repo": MODEL_SPEC["hf_repo"],
        "hf_revision_recorded": record_hf_revision(MODEL_SPEC["hf_repo"]),
        "checkpoint_file": MODEL_SPEC["checkpoint_file"],
        "evaluation_seed": int(cfg["seed"]),
        "architecture": {k: MODEL_SPEC[k] for k in ("n_layers", "n_heads", "d_model", "d_mlp", "d_vocab", "n_ctx", "act_fn", "normalization_type")},
        "transformer_lens_version": getattr(transformer_lens, "__version__", None),
        "torch_version": torch.__version__,
        "device": str(next(model.parameters()).device),
        "git_commit": git_hash(),
        "dataset_sha256": data_hashes,
        "probe_path": str(probe_path) if probe_path.exists() else str(probe_path),
        "probe_exists": bool(probe_info["exists"]),
        "probe_source": probe_info["source"],
        "probe_fallback_used": probe_info["fallback_used"],
        "probe_sha256": probe_hash,
        "results_isolation": isolation,
        "n_pairs": len(pairs),
        "n_inspect": inspect_n,
        "n_test_prefixes": len(test),
        "n_train_prefixes": len(train),
        "sample_counts": {
            "n_pairs": len(pairs),
            "n_inspect": inspect_n,
            "n_test_prefixes": len(test),
            "n_train_prefixes": len(train),
            "n_distinct_recipient_games": dep["n_distinct_recipient_games"],
            "n_distinct_donor_games": dep["n_distinct_donor_games"],
        },
        "n_distinct_recipient_games": dep["n_distinct_recipient_games"],
        "n_distinct_donor_games": dep["n_distinct_donor_games"],
        "donor_reuse": dep["donor_reuse"],
        "independence_claim": "none" if dep["donor_reuse"] or dep["n_distinct_recipient_games"] < dep["n_pairs"] else "pairs_unique_games",
        "n_boot": n_boot,
        "boot_seed": args.boot_seed,
        "pair_seed": int(cfg["seed"]) + 17,
        "mid_post_cause_code": midpost_cause,
        "official_0A_letter": "C / PARTIAL",
        "elapsed_sec": time.time() - t0,
        "failed_required": [s for s, st in stages.items() if st == "FAIL"],
        "blocked_required": [s for s, st in stages.items() if st == "BLOCKED_MISSING_ARTIFACT"],
    }
    write_status(status_path, summary)
    print("=== stages ===")
    for k, v in stages.items():
        print(f"  {k}={v}")
    print(f"scientific_decision={completion['decision']}")
    print("Official 0A letter unchanged: C / PARTIAL")
    if completion["decision"] != "08R1_COMPLETE":
        raise SystemExit(completion["decision"])
    print("08R1_COMPLETE")


if __name__ == "__main__":
    main()
