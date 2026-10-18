"""08R2 validation on the frozen Nanda Othello-GPT.

Does not evaluate the sealed confirmation split.
Does not retrain the model or replace the 08c probe.
Does not fall back to a fake residual model.
Official 0A letter remains C / PARTIAL.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_07f_modular_causal_sanity_validation import ensure_dir, write_csv
from run_08r1_real_model_verification import git_hash, patch_and_audit, resolve_checkpoint
from src.othello_r1 import cluster_bootstrap, paired_cluster_diff
from src.othello_r1_eval import resolve_probe_path, sha256_file
from src.othello_r2 import (
    OFFICIAL_0A_LETTER,
    PROTOCOL_ABD,
    REQUIRED_BASELINES,
    REQUIRED_VALIDATION_FILES,
    SITE,
    STATUS_EDITOR_C_FIT_FAILED,
    STATUS_EXECUTED,
    STATUS_FAILED,
    LinearBoardDecoder,
    abd_pair_quotas,
    apply_reusable_map,
    attribution_sign_edit,
    causal_scores,
    confirmation_guard_text,
    decoder_state_effects,
    editor_attempt_accounting,
    finite_difference_grad_check,
    fit_reusable_map,
    isolate_r2_results_dir,
    iterative_state_edit,
    kl_norm_comparisons,
    load_holdout_game_ids,
    mean_square_edit_safe,
    outcome_bucket,
    pair_census_public,
    pca_patch,
    probe_direction_edit_safe,
    r2_pair_categories,
    random_edit_safe,
    random_kl_threshold,
    require_no_confirmation,
    torch_parity_report,
    train_activation_stats,
    wrong_square_edit_safe,
)
from src.othello_r2_eval import evaluate_r2_completeness
from src.othello_runutil import add_labels, induced_legal, load_split, tokens_of, write_json
from src.toy_othello import (
    cache_site,
    load_config,
    load_othello_gpt,
    run_model_logits,
    softmax_np,
    verify_nanda_checkpoint,
)


PRIMARY_TYPES = {"A_legal_diff", "B_similar_board_legal_change"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="08R2 real-model validation.")
    p.add_argument("--config", default="configs/toy_othello_08r2.json")
    p.add_argument("--parent-config", default="configs/toy_othello.json")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--probe-path", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkpoints-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--n-pairs", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="Small real-model smoke test. Not full validation.")
    return p.parse_args()


def _require_real_stack():
    try:
        import torch
        import transformer_lens
    except Exception as exc:
        raise SystemExit(
            "LOCAL_ENGINEERING_VERIFIED — COLAB_REQUIRED\n"
            f"Real stack missing ({exc}). This runner does not fall back to a fake model."
        )
    return torch, transformer_lens


def merge_cfg(parent: dict, child: dict) -> dict:
    out = dict(parent)
    out.update(child)
    return out


def _snapshot_params(model) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def _params_unchanged(before, model) -> bool:
    after = _snapshot_params(model)
    return all(np.allclose(a, b) for a, b in zip(before, after))


def _logit_steer(model, tokens, hook, h, rec, tgt, budget, steps, lr, batch_size):
    """Behavioral baseline. Uses model logits. Not a state editor."""
    import torch
    from src.toy_othello import pad_batch, square_to_token

    device = next(model.parameters()).device
    h0 = torch.as_tensor(h, dtype=torch.float32, device=device)
    delta = torch.zeros_like(h0, requires_grad=True)
    opt = torch.optim.SGD([delta], lr=float(lr))
    became = [square_to_token(s) for s in set(tgt["legal"]) - set(rec["legal"]) if square_to_token(s) is not None]
    left = [square_to_token(s) for s in set(rec["legal"]) - set(tgt["legal"]) if square_to_token(s) is not None]
    if not became and not left:
        return h0.detach().cpu().numpy(), 0
    toks, lengths = pad_batch([tokens], device)
    t = int(lengths[0])
    completed = 0
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)

        def patch_fn(act, hook_obj, val=h0 + delta, last=t - 1):
            out = act.clone()
            out[0, last, :] = val.to(dtype=out.dtype)
            return out

        logits = model.run_with_hooks(toks, fwd_hooks=[(hook, patch_fn)])[0, t - 1]
        loss = logits.new_zeros(())
        if became:
            loss = loss - logits[became].sum()
        if left:
            loss = loss + logits[left].sum()
        if not torch.isfinite(loss):
            break
        loss.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            break
        opt.step()
        with torch.no_grad():
            nrm = float(delta.norm())
            if nrm > float(budget) and nrm > 0:
                delta.mul_(float(budget) / nrm)
        completed += 1
    return (h0 + delta.detach()).detach().cpu().numpy(), completed


def _boot_rows(rows, field, n_boot, seed, label):
    vals = [r[field] for r in rows]
    clusters = [r["recip_game"] for r in rows]
    out = cluster_bootstrap(vals, clusters, n_boot, seed)
    out["metric"] = field
    out["method"] = label
    out["n_pairs"] = len(rows)
    out["n_recipient_games"] = len(set(clusters))
    out["n_target_games"] = len({r["target_game"] for r in rows})
    return out


def _select_editor(state_rows, causal_rows, floors, *, smoke: bool = False) -> dict:
    by = defaultdict(list)
    for r in causal_rows:
        if r.get("r2_type") not in PRIMARY_TYPES:
            continue
        if not r.get("is_primary_editor"):
            continue
        if not r.get("valid_edit"):
            continue
        if r.get("used_model_logits"):
            continue
        if not r.get("joint_metric_eligible"):
            continue
        by[(r["method"], r["budget"])].append(r)
    ranked = []
    for key, group in by.items():
        st = [s for s in state_rows if s["method"] == key[0] and float(s["budget"]) == float(key[1]) and s.get("r2_type") in PRIMARY_TYPES and s.get("valid_edit")]
        if not st:
            continue
        ch = float(np.nanmean([s["changed_acc"] for s in st]))
        kp = float(np.nanmean([s["unchanged_acc"] for s in st]))
        gain = float(np.nanmean([g["dP_gain"] for g in group if g.get("dP_gain_defined")]))
        loss = float(np.nanmean([g["dP_loss"] for g in group if g.get("dP_loss_defined")]))
        sleg = float(np.nanmean([g["S_legal"] for g in group if g.get("joint_metric_eligible")]))
        kl = float(np.nanmean([g["kl"] for g in group]))
        nrm = float(np.nanmean([g["delta_norm"] for g in group]))
        rand = [r for r in causal_rows if r.get("method") == "random_same_norm" and r.get("budget") == key[1] and r.get("valid_edit") and r.get("r2_type") in PRIMARY_TYPES]
        gate = random_kl_threshold([r["kl"] for r in rand])
        cand = {
            "method": key[0],
            "budget": float(key[1]),
            "changed_acc": ch,
            "unchanged_acc": kp,
            "dP_gain": gain,
            "dP_loss": loss,
            "S_legal": sleg,
            "mean_kl": kl,
            "mean_l2": nrm,
            "random_kl": gate["random_kl_mean"],
            "kl_threshold": gate["threshold"],
            "kl_gate_status": gate["status"],
            "S_legal_per_kl": (sleg / kl) if sleg == sleg and kl and kl == kl else float("nan"),
            "S_legal_per_l2": (sleg / nrm) if sleg == sleg and nrm and nrm == nrm else float("nan"),
            "n": len(group),
            "rep_ok": ch >= floors["changed"] and kp >= floors["unchanged"],
            "beh_ok": gain >= floors["causal"] and loss <= -floors["causal"],
            "kl_ok": gate["threshold"] is not None and kl == kl and kl <= gate["threshold"],
            "used_donor_activation": int(group[0].get("used_donor_activation") or 0),
            "used_model_logits": 0,
            "note": "Do not claim superiority from donor access, larger KL, or unequal coverage.",
        }
        ranked.append(cand)
    if smoke:
        return {
            "method": None,
            "status": "NOT_SELECTED_SMOKE",
            "reason": "Smoke does not select hyperparameters or the official editor.",
            "comparisons": ranked,
        }
    if not ranked:
        return {"method": None, "status": "NO_EDITOR_SELECTED", "comparisons": ranked}
    passing = [c for c in ranked if c["rep_ok"] and c["beh_ok"] and c["kl_ok"]]
    chosen = max(passing or ranked, key=lambda c: (c["S_legal"] if c["S_legal"] == c["S_legal"] else -1e9))
    return {**chosen, "comparisons": ranked, "status": "SELECTED" if passing else "NO_PASSING_EDITOR"}


def _write_report(path: Path, payload: dict) -> None:
    lines = [
        "# 08R2 Validation Scientific Report",
        "",
        f"Status: **{payload['status']}**",
        "",
        "Official labels unchanged:",
        "",
        f"- Original 0A: {OFFICIAL_0A_LETTER}",
        "- 08R1 execution: COMPLETE",
        "- Section 5.3: PARTIAL",
        "- OBSTRUCTION_READY = NO",
        "",
        "Confirmation split was not evaluated.",
        "",
        "## Provenance",
        "",
        f"- git: `{payload.get('git_hash')}`",
        f"- checkpoint: `{payload.get('checkpoint_path')}` sha256 `{payload.get('checkpoint_sha256')}`",
        f"- probe: `{payload.get('probe_path')}` sha256 `{payload.get('probe_sha256')}`",
        f"- site: `{SITE}`",
        f"- results: `{payload.get('results_dir')}`",
        "",
        "## Decoder parity",
        "",
        f"- torch-sklearn max abs proba: {payload.get('parity', {}).get('max_abs_proba')}",
        f"- torch-sklearn mean abs proba: {payload.get('parity', {}).get('mean_abs_proba')}",
        f"- board agree: {payload.get('parity', {}).get('board_agree')}",
        f"- torch_parity_status: {payload.get('torch_parity_status')}",
        "",
        "## Pair counts",
        "",
        f"`{payload.get('pair_counts')}`",
        "",
        "Category C is unavailable and was not fabricated. A/B/D cannot establish same-legal-state invariance.",
        "",
        "## Category slices",
        "",
        f"`{payload.get('category_slices')}`",
        "",
        "Pooled A+B is descriptive only.",
        "",
        "## Editor attempt accounting",
        "",
        f"`{payload.get('editor_attempt_accounting')}`",
        "",
        "Unconditional rates use all attempted edits as the denominator. Conditional effect sizes use valid edits only.",
        "",
        "## KL / L2 comparisons",
        "",
        "Threshold = 1.5 × max(mean random-same-norm KL, 1e-3). Near-zero random KL is floored, not a free pass.",
        "",
        f"`{payload.get('kl_norm_comparisons')}`",
        "",
        "## Selected editor",
        "",
        f"`{payload.get('selected')}`",
        "",
        "## Outcome",
        "",
        f"- validation outcome: {payload.get('validation_outcome')}",
        f"- note: {payload.get('outcome_note')}",
        "",
        "This report records validation-stage evidence only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch, _tl = _require_real_stack()
    parent = load_config(args.parent_config)
    child = load_config(args.config)
    cfg = merge_cfg(parent, child)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.checkpoints_dir:
        cfg["checkpoints_dir"] = args.checkpoints_dir
    isolated = isolate_r2_results_dir(cfg["results_dir"], smoke=bool(args.smoke))  # .../08r2/smoke when --smoke
    out_dir = ensure_dir(isolated["results_dir"])
    n_pairs = int(args.n_pairs or (4 if args.smoke else cfg["n_val_pairs"]))
    n_boot = int(args.n_boot or (200 if args.smoke else cfg["n_boot"]))
    if (not args.smoke) and n_boot < 1000:
        raise SystemExit("n_boot must be >= 1000")
    n_a, n_b = abd_pair_quotas(n_pairs, smoke=bool(args.smoke))

    holdout = load_holdout_game_ids(cfg["holdout_manifest"])
    holdout_hash = holdout["manifest_sha256"]
    if not holdout["hash_ok"]:
        raise SystemExit("confirmation manifest payload hash mismatch")

    probe_info = resolve_probe_path(args.probe_path, cfg["checkpoints_dir"])
    probe_path = Path(probe_info["probe_path"])
    if not probe_info["exists"]:
        raise SystemExit(f"BLOCKED_MISSING_ARTIFACT probe {probe_path}")
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if ckpt is None or not Path(ckpt).is_file():
        raise SystemExit("BLOCKED_MISSING_ARTIFACT checkpoint")

    data_dir = Path(cfg["data_dir"])
    train = add_labels(load_split(data_dir, "train"))
    val = add_labels(load_split(data_dir, "val"))
    require_no_confirmation(train, holdout["game_ids"])
    require_no_confirmation(val, holdout["game_ids"])
    pairs = r2_pair_categories(
        val,
        n_pairs,
        int(cfg["pair_seed"]),
        holdout["game_ids"],
        n_a=n_a,
        n_b=n_b,
        include_c=False,
    )
    pair_counts = dict(Counter(p["r2_type"] for p in pairs))
    if pair_counts.get("A_legal_diff", 0) < 1 or pair_counts.get("B_similar_board_legal_change", 0) < 1:
        raise SystemExit(f"ABD sampler must include A and B; got {pair_counts}")
    write_json(
        out_dir / "08r2_validation_pairs.json",
        {
            "n": len(pairs),
            "counts": pair_counts,
            "n_a": n_a,
            "n_b": n_b,
            "protocol": PROTOCOL_ABD["name"],
            "seed": int(cfg["pair_seed"]),
            "used_confirmation": 0,
            "smoke": int(args.smoke),
            "pairs": pairs,
        },
    )

    with probe_path.open("rb") as handle:
        pack = pickle.load(handle)
    if str(pack.get("hook")) != SITE:
        raise SystemExit(f"REAL_MODEL_VALIDATION_FAILED probe site {pack.get('hook')} != {SITE}")
    decoder = LinearBoardDecoder(pack)
    dec_t = decoder.as_torch()
    rowspace = decoder.rowspace_basis(cfg.get("rowspace_max_rank"))

    model = load_othello_gpt(device=args.device or cfg.get("device"), checkpoint_path=str(ckpt))
    ident = verify_nanda_checkpoint(model)
    if not ident.get("nanda_argmax_match"):
        raise SystemExit(f"REAL_MODEL_VALIDATION_FAILED checkpoint identity {ident}")
    params_before = _snapshot_params(model)
    for p in model.parameters():
        p.requires_grad_(False)
    probe_coef_before = decoder.coef.copy()

    needed = sorted({p["recip"] for p in pairs} | {p["donor"] for p in pairs})
    if args.smoke:
        val_idx = needed
        pos = {i: j for j, i in enumerate(val_idx)}
        val_sub = [val[i] for i in val_idx]
        H_sub = cache_site(model, tokens_of(val_sub), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
        logits_sub = run_model_logits(model, tokens_of(val_sub), batch_size=int(cfg.get("batch_size", 64))).numpy()
        H = np.full((len(val), H_sub.shape[1]), np.nan, dtype=H_sub.dtype)
        base_logits = np.full((len(val), logits_sub.shape[1]), np.nan, dtype=logits_sub.dtype)
        for i, j in pos.items():
            H[i] = H_sub[j]
            base_logits[i] = logits_sub[j]
    else:
        val_tokens = tokens_of(val)
        H = cache_site(model, val_tokens, SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
        base_logits = run_model_logits(model, val_tokens, batch_size=int(cfg.get("batch_size", 64))).numpy()
    base_probs = np.stack([softmax_np(row) if np.isfinite(row).all() else np.full_like(row, np.nan) for row in base_logits], axis=0)

    rng = np.random.default_rng(int(cfg["pair_seed"]) + 3)
    stat_idx = rng.choice(len(train), size=min(int(cfg["n_train_stat_rows"]), len(train)), replace=False)
    train_sub = [train[i] for i in stat_idx]
    Xtr = cache_site(model, tokens_of(train_sub), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
    ytr = np.stack([r["rel_lab"] for r in train_sub], axis=0)
    stats = train_activation_stats(Xtr, rank=int(cfg["pca_rank"]))

    map_n = int(cfg["n_train_map_pairs"] if not args.smoke else min(32, int(cfg["n_train_map_pairs"])))
    map_a, map_b = abd_pair_quotas(map_n, smoke=False)
    map_fail = None
    try:
        map_pairs = [
            p
            for p in r2_pair_categories(
                train,
                map_n,
                int(cfg["pair_seed"]) + 5,
                holdout["game_ids"],
                n_a=map_a,
                n_b=map_b,
                include_c=False,
            )
            if p["r2_type"] in PRIMARY_TYPES
        ]
    except RuntimeError as exc:
        map_pairs = []
        map_fail = f"TRAIN_MAP_PAIRING_FAILED:{exc}"
    if map_pairs:
        map_idx = sorted({p["recip"] for p in map_pairs} | {p["donor"] for p in map_pairs})
        train_map_rows = [train[i] for i in map_idx]
        pos = {i: j for j, i in enumerate(map_idx)}
        Xmap = cache_site(model, tokens_of(train_map_rows), SITE, batch_size=int(cfg.get("cache_batch", 32))).numpy()
        reusable = fit_reusable_map(
            [Xmap[pos[p["recip"]]] for p in map_pairs],
            [Xmap[pos[p["donor"]]] for p in map_pairs],
            [train[p["recip"]]["rel_lab"] for p in map_pairs],
            [train[p["donor"]]["rel_lab"] for p in map_pairs],
            rank=int(cfg["reusable_rank"]),
            ridge=float(cfg["reusable_ridge"]),
        )
    else:
        reusable = {"status": STATUS_EDITOR_C_FIT_FAILED, "reason": map_fail or "INSUFFICIENT_TRAIN_PAIRS", "n": 0}

    if args.smoke:
        parity_idx = np.array(needed[: min(8, len(needed))], dtype=int)
    else:
        parity_idx = rng.choice(len(val), size=min(int(cfg["n_parity"]), len(val)), replace=False)
    torch_parity = torch_parity_report(decoder, pack["models"], H[parity_idx])
    write_csv(out_dir / "08r2_probe_parity.csv", [torch_parity])
    if torch_parity["max_abs_proba"] > 1e-4 or torch_parity["board_agree"] < 1.0:
        raise SystemExit(f"REAL_MODEL_VALIDATION_FAILED torch-sklearn parity {torch_parity}")

    grad_ok = []
    for p in pairs[:2]:
        rec = val[p["recip"]]
        tgt = val[p["donor"]]
        g = finite_difference_grad_check(dec_t, H[p["recip"]], tgt["rel_lab"], rec["rel_lab"])
        g["pair"] = f"{p['recip']}-{p['donor']}"
        grad_ok.append(g)
        if not g["grad_finite"] or g["max_abs_fd_error"] > 5e-2:
            raise SystemExit(f"REAL_MODEL_VALIDATION_FAILED gradient check {g}")

    if not _params_unchanged(params_before, model) or not np.allclose(probe_coef_before, decoder.coef):
        raise SystemExit("REAL_MODEL_VALIDATION_FAILED model or probe parameters changed")

    budgets = [float(b) for b in cfg["norm_budgets"]]
    steps = int(cfg["edit_steps"])
    lr = float(cfg["edit_lr"])
    state_rows = []
    causal_rows = []
    base_rows = []
    traces = []
    ledger = []
    hook_ok = 1

    failures = []

    def record_failure(method, pair, rec, tgt, reason, *, budget, extra=None, is_primary=0):
        extra = extra or {}
        row = {
            "method": method,
            "budget": float(budget),
            "r2_type": pair["r2_type"],
            "recip": pair["recip"],
            "donor": pair.get("donor"),
            "recip_game": pair["recip_game"],
            "target_game": pair["target_game"],
            "valid_edit": 0,
            "baseline_status": extra.get("status", STATUS_FAILED),
            "reason": reason,
            "is_primary_editor": int(is_primary),
            "used_donor_activation": int(extra.get("used_donor_activation") or 0),
            "used_model_logits": int(extra.get("used_model_logits") or 0),
            "synthetic": 0,
        }
        causal_rows.append(row)
        ledger.append({**row, "trial": len(ledger)})
        failures.append(f"{method}:{reason}")
        return row

    def record(method, pair, rec, tgt, edited, *, budget, donor=0, logits=0, extra=None, is_primary=0, traces_in=None, baseline_status=STATUS_EXECUTED, eval_subset="all_pairs"):
        nonlocal hook_ok
        extra = extra or {}
        if extra.get("valid_edit") == 0 or edited is None:
            return record_failure(method, pair, rec, tgt, extra.get("reason", "invalid_edit"), budget=budget, extra=extra, is_primary=is_primary)
        toks = [val[pair["recip"]]["tokens"]]
        vals = torch.tensor(np.asarray(edited, dtype=np.float32).reshape(1, -1))
        logits_e, rb, audit = patch_and_audit(model, toks, SITE, vals, batch_size=1)
        hook_ok = hook_ok and int(audit.get("ok", 0))
        pe = softmax_np(logits_e.numpy()[0])
        pb = base_probs[pair["recip"]]
        causal = causal_scores(pb, pe, rec["legal"], tgt["legal"])
        dec = decoder_state_effects(decoder, edited, rec["rel_lab"], tgt["rel_lab"])
        from src.toy_othello import relative_to_absolute

        pred_abs_lab = relative_to_absolute(dec["pred_rel"], int(rec["player"]))
        induced = induced_legal(pred_abs_lab, int(rec["player"]))
        tgt_set, ind_set = set(tgt["legal"]), set(induced)
        jacc = float(len(tgt_set & ind_set) / len(tgt_set | ind_set)) if (tgt_set or ind_set) else float("nan")
        row = {
            "method": method,
            "budget": float(budget),
            "r2_type": pair["r2_type"],
            "recip": pair["recip"],
            "donor": pair["donor"],
            "recip_game": pair["recip_game"],
            "target_game": pair["target_game"],
            "hamming": pair.get("hamming"),
            "used_donor_activation": int(donor),
            "used_model_logits": int(logits),
            "is_primary_editor": int(is_primary),
            "valid_edit": 1,
            "baseline_status": baseline_status,
            "eval_subset": eval_subset,
            "synthetic": 0,
            "delta_norm": float(np.linalg.norm(np.asarray(edited) - H[pair["recip"]])),
            "kl": causal["kl"],
            "dP_gain": causal["dP_gain"],
            "dP_loss": causal["dP_loss"],
            "S_legal": causal["S_legal"],
            "hook_ok": int(audit.get("ok", 0)),
            "board_exact": dec["board_exact"],
            "changed_acc": dec["changed_acc"],
            "unchanged_acc": dec["unchanged_acc"],
            "square_acc": dec["square_acc"],
            "induced_jaccard": jacc,
            "outcome": outcome_bucket(dec["board_exact"], causal["dP_gain"], causal["dP_loss"]),
            **{k: causal[k] for k in causal if k not in {"dP_gain", "dP_loss", "S_legal", "kl"}},
            **extra,
        }
        causal_rows.append(row)
        state_rows.append({**row})
        if method in {"full_donor_patch", "pca32_donor", "pca32_complement", "probe_direction", "mean_square", "random_same_norm", "wrong_square", "wrong_donor", "noop", "attribution_sign", "logit_steer"}:
            base_rows.append(row)
        ledger.append(
            {
                "trial": len(ledger),
                "method": method,
                "budget": float(budget),
                "r2_type": pair["r2_type"],
                "recip_game": pair["recip_game"],
                "target_game": pair["target_game"],
                "dP_gain": row["dP_gain"],
                "dP_loss": row["dP_loss"],
                "S_legal": row["S_legal"],
                "changed_acc": row["changed_acc"],
                "unchanged_acc": row["unchanged_acc"],
                "board_exact": row["board_exact"],
                "delta_norm": row["delta_norm"],
                "kl": row["kl"],
                "used_donor_activation": int(donor),
                "used_model_logits": int(logits),
            }
        )
        if traces_in:
            for tr in traces_in:
                traces.append({"method": method, "recip": pair["recip"], "donor": pair["donor"], "budget": float(budget), **tr})
        return row

    t0 = time.time()
    a_pairs = [p for p in pairs if p["r2_type"] in PRIMARY_TYPES]
    wrong_donor_pool = [H[p["donor"]] for p in a_pairs]
    for pair in pairs:
        rec = val[pair["recip"]]
        tgt = val[pair["donor"]]
        h = H[pair["recip"]]
        h_don = H[pair["donor"]]
        record("noop", pair, rec, tgt, h, budget=0.0, extra={"steps_completed": 0})
        record("full_donor_patch", pair, rec, tgt, h_don, budget=float(np.linalg.norm(h_don - h)), donor=1)
        pca_d = pca_patch(h, h_don, stats["pca_basis"], False)
        record("pca32_donor", pair, rec, tgt, pca_d.get("edited"), budget=pca_d.get("delta_norm", 32.0), donor=1, extra=pca_d)
        pca_c = pca_patch(h, h_don, stats["pca_basis"], True)
        record("pca32_complement", pair, rec, tgt, pca_c.get("edited"), budget=pca_c.get("delta_norm", 32.0), donor=1, extra=pca_c)
        alt = next((v for v in wrong_donor_pool if not np.allclose(v, h_don)), None)
        if alt is None:
            record_failure("wrong_donor", pair, rec, tgt, "no_alternate_donor", budget=0.0, extra={"used_donor_activation": 1})
        else:
            record("wrong_donor", pair, rec, tgt, alt, budget=float(np.linalg.norm(alt - h)), donor=1)
        if pair["r2_type"] == "D_wrong_target":
            record("wrong_target", pair, rec, tgt, None, budget=0.0, extra={"valid_edit": 0, "reason": "control_pair_not_an_edit", "status": "NOT_APPLICABLE"})
        for budget in budgets:
            rec_lab, tgt_lab = rec["rel_lab"], tgt["rel_lab"]
            ed_a = iterative_state_edit(
                h, tgt_lab, rec_lab, dec_t,
                steps=steps, lr=lr, budget=budget,
                lambda_norm=float(cfg["lambda_norm"]),
                eta_preserve=float(cfg["eta_preserve"]),
                mu_manifold=float(cfg["mu_manifold"]),
                mean_act=stats["mean"],
                pca_basis=stats["pca_basis"],
            )
            record("editor_A_fullspace", pair, rec, tgt, ed_a.get("edited"), budget=budget, is_primary=1, extra={k: ed_a[k] for k in ed_a if k not in {"edited", "delta", "traces"}}, traces_in=ed_a.get("traces"))
            ed_b = iterative_state_edit(
                h, tgt_lab, rec_lab, dec_t,
                steps=steps, lr=lr, budget=budget,
                lambda_norm=float(cfg["lambda_norm"]),
                eta_preserve=float(cfg["eta_preserve"]),
                mu_manifold=float(cfg["mu_manifold"]),
                mean_act=stats["mean"],
                pca_basis=stats["pca_basis"],
                rowspace=rowspace,
            )
            record("editor_B_rowspace", pair, rec, tgt, ed_b.get("edited"), budget=budget, is_primary=1, extra={k: ed_b[k] for k in ed_b if k not in {"edited", "delta", "traces"}}, traces_in=ed_b.get("traces"))
            if reusable.get("status") == "FIT":
                hc = apply_reusable_map(reusable["W"], h, rec_lab, tgt_lab, budget)
                record("editor_C_reusable", pair, rec, tgt, hc.get("edited"), budget=budget, is_primary=1, extra=hc)
            else:
                record_failure(
                    "editor_C_reusable",
                    pair,
                    rec,
                    tgt,
                    reusable.get("reason", STATUS_EDITOR_C_FIT_FAILED),
                    budget=budget,
                    extra={"status": STATUS_EDITOR_C_FIT_FAILED, "reusable_n": reusable.get("n", 0)},
                    is_primary=1,
                )
            hp = probe_direction_edit_safe(h, rec_lab, tgt_lab, pack["models"], budget)
            record("probe_direction", pair, rec, tgt, hp.get("edited"), budget=budget, extra=hp)
            hm = mean_square_edit_safe(Xtr, ytr, h, rec_lab, tgt_lab, budget)
            record("mean_square", pair, rec, tgt, hm.get("edited"), budget=budget, extra=hm)
            hr = random_edit_safe(h, budget, int(cfg["pair_seed"]) + pair["recip"])
            record("random_same_norm", pair, rec, tgt, hr.get("edited"), budget=budget, extra=hr)
            wr = wrong_square_edit_safe(h, rec_lab, tgt_lab, pack["models"], budget, np.random.default_rng(int(cfg["pair_seed"]) + pair["recip"] + 11))
            record("wrong_square", pair, rec, tgt, wr.get("edited"), budget=budget, extra=wr)
            ha = attribution_sign_edit(h, tgt_lab, rec_lab, dec_t, budget)
            record("attribution_sign", pair, rec, tgt, ha.get("edited") if isinstance(ha, dict) else ha, budget=budget, extra=ha if isinstance(ha, dict) else {"loss_name": "decoder_state_nll"})
            if budget == budgets[-1] and pair["r2_type"] in PRIMARY_TYPES:
                hs, nstep = _logit_steer(model, rec["tokens"], SITE, h, rec, tgt, budget, steps=8, lr=lr, batch_size=1)
                record("logit_steer", pair, rec, tgt, hs, budget=budget, logits=1, extra={"steps_completed": nstep, "reason": "model_logit_steer"}, eval_subset="primary_last_budget")
            elif budget != budgets[-1]:
                record_failure("logit_steer", pair, rec, tgt, "restricted_to_last_budget_primary_pairs", budget=budget, extra={"status": "NOT_APPLICABLE", "used_model_logits": 1})

    if not _params_unchanged(params_before, model):
        raise SystemExit("REAL_MODEL_VALIDATION_FAILED model weights changed during edits")

    primary_causal = [r for r in causal_rows if r["r2_type"] in PRIMARY_TYPES and r.get("valid_edit")]
    boot_rows = []
    for method in sorted({r["method"] for r in primary_causal}):
        for budget in sorted({r["budget"] for r in primary_causal if r["method"] == method}):
            group = [r for r in primary_causal if r["method"] == method and r["budget"] == budget and r.get("eval_subset", "all_pairs") == "all_pairs"]
            if not group:
                continue
            for field in ("dP_gain", "dP_loss", "S_legal", "changed_acc", "unchanged_acc", "kl", "delta_norm"):
                subset = [r for r in group if field != "S_legal" or r.get("joint_metric_eligible")]
                if field == "dP_gain":
                    subset = [r for r in group if r.get("dP_gain_defined")]
                if field == "dP_loss":
                    subset = [r for r in group if r.get("dP_loss_defined")]
                if subset:
                    boot_rows.append(_boot_rows(subset, field, n_boot, int(cfg["boot_seed"]), f"{method}|{budget}"))
            probe_g = [r for r in primary_causal if r["method"] == "probe_direction" and r["budget"] == budget and r.get("joint_metric_eligible")]
            rand_g = [r for r in primary_causal if r["method"] == "random_same_norm" and r["budget"] == budget and r.get("joint_metric_eligible")]
            wrong_g = [r for r in causal_rows if r["method"] == method and r["r2_type"] == "D_wrong_target" and r["budget"] == budget and r.get("valid_edit") and r.get("joint_metric_eligible")]
            if probe_g and len(probe_g) == len(group):
                d = paired_cluster_diff([r["S_legal"] for r in group], [r["S_legal"] for r in probe_g], [r["recip_game"] for r in group], n_boot, int(cfg["boot_seed"]))
                boot_rows.append({**d, "metric": "S_legal_minus_probe_direction", "method": f"{method}|{budget}"})
            if rand_g and len(rand_g) == len(group):
                d = paired_cluster_diff([r["S_legal"] for r in group], [r["S_legal"] for r in rand_g], [r["recip_game"] for r in group], n_boot, int(cfg["boot_seed"]))
                boot_rows.append({**d, "metric": "S_legal_minus_random", "method": f"{method}|{budget}"})
            if wrong_g:
                boot_rows.append({**_boot_rows(wrong_g, "S_legal", n_boot, int(cfg["boot_seed"]), f"{method}|{budget}|wrong_target"), "metric": "S_legal_wrong_target"})

    floors = {
        "changed": float(cfg["representation_changed_acc_min"]),
        "unchanged": float(cfg["representation_unchanged_acc_min"]),
        "causal": float(cfg["causal_abs_min"]),
    }
    selected = _select_editor(state_rows, causal_rows, floors, smoke=bool(args.smoke))
    accounting = editor_attempt_accounting(causal_rows, floors)
    comparisons = kl_norm_comparisons(causal_rows)
    write_json(out_dir / "08r2_editor_attempt_accounting.json", accounting)
    write_csv(out_dir / "08r2_kl_norm_comparisons.csv", comparisons)
    buckets = Counter(r["outcome"] for r in primary_causal if r.get("is_primary_editor"))
    slice_rows = {}
    for name, types in (
        ("A", {"A_legal_diff"}),
        ("B", {"B_similar_board_legal_change"}),
        ("pooled_AB", PRIMARY_TYPES),
    ):
        group = [r for r in primary_causal if r.get("r2_type") in types and r.get("is_primary_editor")]
        slice_rows[name] = {
            "n": len(group),
            "mean_S_legal": float(np.nanmean([r["S_legal"] for r in group if r.get("joint_metric_eligible")])) if group else float("nan"),
            "mean_dP_gain": float(np.nanmean([r["dP_gain"] for r in group if r.get("dP_gain_defined")])) if group else float("nan"),
            "mean_dP_loss": float(np.nanmean([r["dP_loss"] for r in group if r.get("dP_loss_defined")])) if group else float("nan"),
        }
    if args.smoke:
        outcome = "SMOKE_TECHNICAL_INTEGRATION_ONLY"
        note = (
            "A smoke-test pass establishes technical integration only. "
            "It is not scientific evidence that the mathematical state editor works. "
            "A/B/D cannot establish same-legal-state invariance. Category C is unavailable."
        )
    elif selected.get("rep_ok") and selected.get("beh_ok") and selected.get("kl_ok"):
        outcome = "A_promising_state_specific_causal_editor"
        note = "Validation suggests a promising editor. Confirmation remains sealed. A/B/D cannot establish same-legal-state invariance."
    elif selected.get("rep_ok") and not selected.get("beh_ok"):
        outcome = "B_probe_editing_without_causal_control"
        note = "Decoder moves toward the target board, but legal-set behavior does not."
    elif selected.get("beh_ok") and not selected.get("rep_ok"):
        outcome = "C_generic_output_intervention"
        note = "Behavior moved without a corresponding decoded target-state edit."
    else:
        outcome = "D_inconclusive_or_weak"
        note = "Validation did not meet representation and causal floors together."

    write_csv(out_dir / "08r2_optimization_traces.csv", traces)
    write_csv(out_dir / "08r2_validation_state_effects.csv", state_rows)
    write_csv(out_dir / "08r2_validation_causal_effects.csv", causal_rows)
    write_csv(out_dir / "08r2_validation_baselines.csv", base_rows)
    write_csv(out_dir / "08r2_validation_bootstrap.csv", boot_rows)
    write_csv(out_dir / "08r2_validation_trial_ledger.csv", ledger)

    summary = {
        "status": "PENDING_COMPLETENESS",
        "official_0A_letter": OFFICIAL_0A_LETTER,
        "section_5_3_status": "PARTIAL",
        "obstruction_ready": "NO",
        "confirmation_evaluated": 0,
        "git_hash": git_hash(),
        "site": SITE,
        "probe_site": str(pack.get("hook")),
        "nanda_ok": int(bool(ident.get("nanda_argmax_match"))),
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "probe_path": str(probe_path),
        "probe_sha256": sha256_file(probe_path),
        "parity": torch_parity,
        "torch_parity_status": "PASS" if torch_parity["max_abs_proba"] <= 1e-4 and torch_parity["board_agree"] >= 1.0 else "FAIL",
        "grad_checks": grad_ok,
        "pair_counts": dict(Counter(p["r2_type"] for p in pairs)),
        "pair_census": None if args.smoke else pair_census_public(val, holdout["game_ids"]),
        "protocol": PROTOCOL_ABD,
        "n_a": n_a,
        "n_b": n_b,
        "category_slices": slice_rows,
        "editor_attempt_accounting": accounting,
        "kl_norm_comparisons": comparisons,
        "same_legal_invariance": "NOT_ESTABLISHABLE_ABD_NO_CATEGORY_C",
        "n_pairs": len(pairs),
        "n_boot": n_boot,
        "boot_seed": int(cfg["boot_seed"]),
        "rowspace_rank": int(rowspace.shape[1]),
        "reusable": {k: reusable[k] for k in reusable if k != "W"},
        "editor_c_status": reusable.get("status"),
        "logit_steer_restriction": "primary_last_budget",
        "selected": selected,
        "outcome_buckets": dict(buckets),
        "validation_outcome": outcome,
        "outcome_note": note,
        "hook_ok": int(hook_ok),
        "elapsed_sec": time.time() - t0,
        "nanda": ident,
        "model_params_unchanged": True,
        "probe_params_unchanged": True,
        "smoke": int(args.smoke),
        "explicit_failures": failures[:50],
        "required_baselines": list(REQUIRED_BASELINES),
    }
    write_json(out_dir / "08r2_validation_summary.json", summary)
    write_json(
        out_dir / "08r2_reproducibility_manifest.json",
        {
            "git_hash": git_hash(),
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "probe_sha256": summary["probe_sha256"],
            "holdout_sha256": holdout_hash,
            "config": {k: cfg[k] for k in cfg if k in child or k in {"site", "n_val_pairs", "n_boot"}},
            "selected": selected,
            "confirmation_sealed": True,
            "confirmation_evaluator": "CONFIRMATION_EVALUATOR_NOT_IMPLEMENTED",
        },
    )
    write_json(
        out_dir / "08r2_confirmation_protocol.json",
        {
            "status": "CONFIRMATION_EVALUATOR_NOT_IMPLEMENTED",
            "execute": False,
            "reason": confirmation_guard_text(),
            "frozen_editor": selected,
            "holdout_sha256": holdout_hash,
        },
    )
    _write_report(out_dir / "08r2_scientific_report.md", {**summary, "results_dir": str(out_dir)})
    completeness = evaluate_r2_completeness(out_dir, smoke=bool(args.smoke))
    summary["completeness"] = completeness
    if completeness["ok"]:
        summary["status"] = completeness["status"]
    else:
        summary["status"] = "REAL_MODEL_VALIDATION_FAILED"
    write_json(out_dir / "08r2_validation_summary.json", summary)
    print(summary["status"])
    print(f"  outcome={outcome}")
    print(f"  selected={selected}")
    print(f"  completeness_failures={completeness['failures']}")
    print(f"  official letter unchanged: {OFFICIAL_0A_LETTER}")
    print("  confirmation sealed: evaluator not implemented; not evaluated")
    if not completeness["ok"]:
        raise SystemExit("REAL_MODEL_VALIDATION_FAILED completeness")


if __name__ == "__main__":
    main()
