"""Experiment 0B causal-mechanism sanity checks at the frozen 07d checkpoint.

This script does not retrain, does not pick another checkpoint, and does not
retune the 07d grid. Official 07d remains NO_GO. The question is whether that
NO_GO was a noncausal 10-D probe readout, a distributed causal encoding, a
site that cannot interchange the sum, or an inconclusive control failure.

All maps that see labels or activations are fit on the training split only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from src.toy_modular import (
    apply_ridge,
    candidate_margin,
    decode_sum_from_latent,
    fit_ridge,
    germ_subspace,
    germ_target_delta,
    hardest_incorrect,
    key_frequencies,
    kl_divergence,
    latent_targets,
    load_checkpoint,
    make_dataset,
    match_norm_in_subspace,
    pca_subspace,
    r_squared,
    random_subspace,
    split_dataset,
)


HOOK_MLP_POST = "blocks.0.mlp.hook_post"
SITE_ID = "L0_mlp_post"
POSITION = 2
DECODE_GATE = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-checkpoint causal sanity validation for Experiment 0B."
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/toy_modular/checkpoints/step_0009080.pt",
    )
    parser.add_argument("--results-dir", default="artifacts/results/toy_modular")
    parser.add_argument("--figures-dir", default="artifacts/figures/toy_modular")
    parser.add_argument("--n-freqs", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-recipients", type=int, default=128)
    parser.add_argument("--n-donors-each", type=int, default=2)
    parser.add_argument("--n-phase-examples", type=int, default=256)
    parser.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sv-tol", type=float, default=1e-2, help="σ_i/σ_max threshold for effective rank")
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def project_onto(vectors, basis):
    basis = basis.to(device=vectors.device, dtype=vectors.float().dtype)
    return (vectors.float() @ basis) @ basis.T


def project_out(vectors, basis):
    return vectors.float() - project_onto(vectors, basis)


def energy_fraction(vectors, basis) -> float:
    proj = project_onto(vectors, basis)
    num = proj.pow(2).sum(dim=-1)
    den = vectors.float().pow(2).sum(dim=-1).clamp_min(1e-12)
    return float((num / den).mean())


def mean_pert_norm(vectors, basis) -> float:
    return float(project_onto(vectors, basis).norm(dim=-1).mean())


def variance_fraction(vectors, basis) -> float:
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    proj = project_onto(centered, basis)
    num = float(proj.pow(2).sum())
    den = float(centered.pow(2).sum())
    return num / den if den > 0 else float("nan")


def effective_rank(weight, tol: float) -> tuple[int, list[float]]:
    import torch

    singular = torch.linalg.svdvals(weight.float())
    values = [float(v) for v in singular]
    if not values or values[0] <= 0:
        return 0, values
    cutoff = tol * values[0]
    rank = sum(1 for v in values if v > cutoff)
    return rank, values


def softmax(logits):
    import torch.nn.functional as F

    return F.softmax(logits.float(), dim=-1)


def mean_margin_true(logits, labels):
    return float(candidate_margin(logits, labels, hardest_incorrect(logits, labels)).mean())


def class_prob(probs, class_ids):
    import torch

    idx = torch.arange(probs.shape[0], device=probs.device)
    return probs[idx, class_ids]


def rotate_true_latent(p: int, freqs: list[int], sums, delta: int):
    """Simulator y(s+Δ) via frequency-specific phase rotations of y(s)."""
    import torch

    device = sums.device if hasattr(sums, "device") else None
    s = sums.float()
    cols = []
    for k in freqs:
        theta0 = 2 * math.pi * k * s / p
        theta = 2 * math.pi * k * delta / p
        cos_s = torch.cos(theta0)
        sin_s = torch.sin(theta0)
        c, sn = math.cos(theta), math.sin(theta)
        cols.append(c * cos_s - sn * sin_s)
        cols.append(sn * cos_s + c * sin_s)
    stacked = torch.stack(cols, dim=-1)
    return stacked.to(device) if device is not None else stacked


def cache_mlp_post(model, tokens, batch_size: int = 2048):
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start : start + batch_size]
            _, cache = model.run_with_cache(batch, names_filter=lambda n: n == HOOK_MLP_POST)
            chunks.append(cache[HOOK_MLP_POST][:, POSITION, :].detach().cpu())
    return torch.cat(chunks, dim=0)


def logits_with_mlp_post(model, tokens, replacement, batch_size: int = 256):
    """Forward `tokens` after replacing mlp_post at '=' with `replacement` [n, d]."""
    import torch

    outs = []
    n = tokens.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = tokens[start:end]
            h_new = replacement[start:end]

            def hook(acts, hook, h=h_new):  # noqa: ARG001
                acts[:, POSITION, :] = h.to(device=acts.device, dtype=acts.dtype)
                return acts

            with model.hooks(fwd_hooks=[(HOOK_MLP_POST, hook)]):
                outs.append(model(batch)[:, -1, :].detach())
    return torch.cat(outs, dim=0)


def logits_with_delta(model, tokens, base_h, delta, batch_size: int = 256):
    return logits_with_mlp_post(model, tokens, base_h.float() + delta.float(), batch_size)


def sample_pairs(
    tokens,
    labels,
    correct,
    delta: int,
    n_recipients: int,
    n_donors_each: int,
    seed: int,
    p: int,
):
    """Donors are correctly predicted. Recipient sum + delta (mod p) equals donor sum."""
    import torch

    generator = torch.Generator().manual_seed(seed + 17 * (delta + 1))
    labels_cpu = labels.detach().cpu()
    correct_cpu = correct.detach().cpu()
    tokens_cpu = tokens.detach().cpu()
    correct_by_sum: dict[int, list[int]] = {}
    for i in range(labels_cpu.shape[0]):
        if bool(correct_cpu[i]):
            correct_by_sum.setdefault(int(labels_cpu[i]), []).append(i)

    def different_pair(i: int, j: int) -> bool:
        return int(tokens_cpu[i, 0]) != int(tokens_cpu[j, 0]) or int(tokens_cpu[i, 1]) != int(tokens_cpu[j, 1])

    candidates = []
    for i in range(labels_cpu.shape[0]):
        donor_sum = (int(labels_cpu[i]) + delta) % p
        donors = [j for j in correct_by_sum.get(donor_sum, []) if j != i and different_pair(i, j)]
        if donors:
            candidates.append(i)
    if not candidates:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    incorrect = [i for i in candidates if not bool(correct_cpu[i])]
    correct_r = [i for i in candidates if bool(correct_cpu[i])]
    n_inc = min(len(incorrect), n_recipients // 2)
    n_cor = min(len(correct_r), n_recipients - n_inc)
    chosen = []
    if n_inc:
        perm = torch.randperm(len(incorrect), generator=generator).tolist()
        chosen.extend(incorrect[k] for k in perm[:n_inc])
    if n_cor:
        perm = torch.randperm(len(correct_r), generator=generator).tolist()
        chosen.extend(correct_r[k] for k in perm[:n_cor])
    if len(chosen) < min(n_recipients, len(candidates)):
        leftover = [i for i in candidates if i not in set(chosen)]
        perm = torch.randperm(len(leftover), generator=generator).tolist()
        chosen.extend(leftover[k] for k in perm[: n_recipients - len(chosen)])

    rec_idx, don_idx = [], []
    for r in chosen:
        donor_sum = (int(labels_cpu[r]) + delta) % p
        pool = [j for j in correct_by_sum.get(donor_sum, []) if j != r and different_pair(r, j)]
        if not pool:
            continue
        perm = torch.randperm(len(pool), generator=generator)
        take = min(n_donors_each, len(pool))
        for k in range(take):
            rec_idx.append(r)
            don_idx.append(pool[int(perm[k])])
    rec = torch.tensor(rec_idx, dtype=torch.long)
    don = torch.tensor(don_idx, dtype=torch.long)
    return rec, don


def pair_metrics(
    logits_before,
    logits_after,
    labels,
    donor_sums,
    h_rec,
    h_don,
    recipient_correct,
):
    import torch

    probs_b = softmax(logits_before)
    probs_a = softmax(logits_after)
    pred_b = logits_before.argmax(dim=-1)
    pred_a = logits_after.argmax(dim=-1)
    p_donor_b = class_prob(probs_b, donor_sums)
    p_donor_a = class_prob(probs_a, donor_sums)
    p_true_b = class_prob(probs_b, labels)
    p_true_a = class_prob(probs_a, labels)
    margin_donor_b = candidate_margin(logits_before, donor_sums, hardest_incorrect(logits_before, donor_sums))
    margin_donor_a = candidate_margin(logits_after, donor_sums, hardest_incorrect(logits_after, donor_sums))
    return {
        "n_pairs": int(logits_before.shape[0]),
        "frac_recipient_correct_before": float(recipient_correct.float().mean()),
        "mean_abs_logit_change": float((logits_after - logits_before).abs().mean()),
        "mean_kl": float(kl_divergence(logits_before, logits_after).mean()),
        "mean_patch_norm": float((h_don.float() - h_rec.float()).norm(dim=-1).mean()),
        "mean_p_donor_before": float(p_donor_b.mean()),
        "mean_p_donor_after": float(p_donor_a.mean()),
        "mean_delta_p_donor": float((p_donor_a - p_donor_b).mean()),
        "mean_p_true_before": float(p_true_b.mean()),
        "mean_p_true_after": float(p_true_a.mean()),
        "frac_pred_donor_before": float((pred_b == donor_sums).float().mean()),
        "frac_pred_donor_after": float((pred_a == donor_sums).float().mean()),
        "frac_pred_true_before": float((pred_b == labels).float().mean()),
        "frac_pred_true_after": float((pred_a == labels).float().mean()),
        "mean_margin_donor_before": float(margin_donor_b.mean()),
        "mean_margin_donor_after": float(margin_donor_a.mean()),
        "mean_delta_margin_donor": float((margin_donor_a - margin_donor_b).mean()),
        "frac_logits_changed": float(((logits_after - logits_before).abs().amax(dim=-1) > 1e-6).float().mean()),
    }


def intervention_row(name, logits_before, logits_after, labels, h_before, h_after):
    pred_b = logits_before.argmax(dim=-1)
    pred_a = logits_after.argmax(dim=-1)
    acc_b = float((pred_b == labels).float().mean())
    acc_a = float((pred_a == labels).float().mean())
    delta = h_after.float() - h_before.float()
    energy = h_before.float().pow(2).sum(dim=-1).clamp_min(1e-12)
    return {
        "method": name,
        "mean_pert_norm": float(delta.norm(dim=-1).mean()),
        "mean_energy_removed_frac": float((delta.pow(2).sum(dim=-1) / energy).mean()),
        "mean_kl": float(kl_divergence(logits_before, logits_after).mean()),
        "acc_before": acc_b,
        "acc_after": acc_a,
        "acc_change": acc_a - acc_b,
        "mean_margin_before": mean_margin_true(logits_before, labels),
        "mean_margin_after": mean_margin_true(logits_after, labels),
        "mean_abs_logit_change": float((logits_after - logits_before).abs().mean()),
    }


def test1_representation(args, model, cfg, freqs, train_tokens, test_tokens, test_labels):
    import torch

    p = cfg.p
    train_h = cache_mlp_post(model, train_tokens)
    test_h = cache_mlp_post(model, test_tokens)
    train_y = latent_targets(train_tokens, p, freqs).cpu()
    test_y = latent_targets(test_tokens, p, freqs).cpu()
    true_sum = ((test_tokens[:, 0] + test_tokens[:, 1]) % p).cpu()

    W = fit_ridge(train_h, train_y, args.ridge)
    B = germ_subspace(W)
    pred = apply_ridge(W, test_h)
    decoded = decode_sum_from_latent(pred, p, freqs)
    decode_acc = float((decoded == true_sum).float().mean())
    r2 = r_squared(pred, test_y)

    shuffle_idx = torch.randperm(train_y.shape[0], generator=torch.Generator().manual_seed(args.seed))
    W_shuf = fit_ridge(train_h, train_y[shuffle_idx], args.ridge)
    dec_shuf = decode_sum_from_latent(apply_ridge(W_shuf, test_h), p, freqs)
    decode_shuf = float((dec_shuf == true_sum).float().mean())

    basis_rand = random_subspace(train_h.shape[1], train_y.shape[1], args.seed)
    W_rand = fit_ridge(train_h.float() @ basis_rand, train_y, args.ridge)
    dec_rand = decode_sum_from_latent(apply_ridge(W_rand, test_h.float() @ basis_rand), p, freqs)
    decode_rand = float((dec_rand == true_sum).float().mean())

    rank, singular = effective_rank(W[:-1], args.sv_tol)
    var_frac = variance_fraction(train_h, B)
    energy_frac = energy_fraction(test_h, B)

    test_h_abl = project_out(test_h, B)
    train_h_abl = project_out(train_h, B)
    lin_before = test_h.float() @ W[:-1].float()
    lin_after = test_h_abl @ W[:-1].float()
    lin_rel = float(lin_after.norm() / (lin_before.norm() + 1e-12))
    pred_abl = apply_ridge(W, test_h_abl)
    decode_abl = float((decode_sum_from_latent(pred_abl, p, freqs) == true_sum).float().mean())
    r2_abl = r_squared(pred_abl, test_y)

    W_resid = fit_ridge(train_h_abl, train_y, args.ridge)
    pred_resid = apply_ridge(W_resid, test_h_abl)
    decode_resid = float((decode_sum_from_latent(pred_resid, p, freqs) == true_sum).float().mean())
    r2_resid = r_squared(pred_resid, test_y)

    row = {
        "checkpoint_step": 9080,
        "site_id": SITE_ID,
        "hook_name": HOOK_MLP_POST,
        "position": POSITION,
        "d_site": int(train_h.shape[1]),
        "latent_dim": int(train_y.shape[1]),
        "key_frequencies": " ".join(str(k) for k in freqs),
        "decode_acc_test": decode_acc,
        "r2_test": r2,
        "decode_acc_shuffled": decode_shuf,
        "decode_acc_random_subspace": decode_rand,
        "chance_decode_acc": 1.0 / p,
        "effective_rank": rank,
        "singular_values": " ".join(f"{v:.6g}" for v in singular),
        "train_variance_fraction_in_germ": var_frac,
        "test_energy_fraction_in_germ": energy_frac,
        "linear_probe_residual_relnorm_after_ablation": lin_rel,
        "decode_acc_original_probe_after_ablation": decode_abl,
        "r2_original_probe_after_ablation": r2_abl,
        "decode_acc_residual_probe": decode_resid,
        "r2_residual_probe": r2_resid,
        "residual_still_decodes": decode_resid > max(0.25, 5.0 / p),
    }
    return row, {
        "train_h": train_h,
        "test_h": test_h,
        "W": W,
        "B": B,
        "true_sum": true_sum,
        "decode_acc": decode_acc,
        "r2": r2,
    }


def test2_controls(args, model, test_tokens, test_labels, base_logits, test_h, B, train_h):
    import torch

    B_rand = random_subspace(test_h.shape[1], B.shape[1], args.seed)
    B_pca = pca_subspace(train_h, B.shape[1])

    cyc_delta = project_onto(test_h, B)
    pca_delta = project_onto(test_h, B_pca)
    rand_delta = project_onto(test_h, B_rand)
    cyc_norm = cyc_delta.norm(dim=-1, keepdim=True)
    pca_scale = torch.where(
        pca_delta.norm(dim=-1, keepdim=True) > 1e-8,
        cyc_norm / (pca_delta.norm(dim=-1, keepdim=True) + 1e-12),
        torch.zeros_like(cyc_norm),
    )
    rand_scale = torch.where(
        rand_delta.norm(dim=-1, keepdim=True) > 1e-8,
        cyc_norm / (rand_delta.norm(dim=-1, keepdim=True) + 1e-12),
        torch.zeros_like(cyc_norm),
    )

    methods = {
        "noop": test_h,
        "ridge_germ_project_out": project_out(test_h, B),
        "random_same_rank_project_out": project_out(test_h, B_rand),
        "pca_same_rank_project_out": project_out(test_h, B_pca),
        "random_energy_matched": test_h.float() - rand_scale * rand_delta,
        "pca_energy_matched": test_h.float() - pca_scale * pca_delta,
    }
    rows = []
    logits_by_method = {"base": base_logits}
    for name, h_new in methods.items():
        logits = logits_with_mlp_post(model, test_tokens, h_new, args.batch_size)
        logits_by_method[name] = logits
        rows.append(intervention_row(name, base_logits, logits, test_labels, test_h, h_new.cpu()))
    return rows, {"B_rand": B_rand, "B_pca": B_pca, "logits": logits_by_method}


def eval_pairs(model, tokens, labels, correct, rec, don, test_h, batch_size, extra=None):
    import torch

    if rec.numel() == 0:
        return None, None
    rec_tokens = tokens[rec]
    rec_labels = labels[rec]
    donor_sums = labels[don]
    h_rec = test_h[rec.cpu()]
    h_don = test_h[don.cpu()]
    with torch.no_grad():
        logits_before = model(rec_tokens)[:, -1, :]
    logits_after = logits_with_mlp_post(model, rec_tokens, h_don, batch_size)
    stats = pair_metrics(
        logits_before, logits_after, rec_labels, donor_sums, h_rec, h_don, correct[rec]
    )
    if extra:
        stats.update(extra)
    payload = {
        "rec": rec,
        "don": don,
        "rec_tokens": rec_tokens,
        "rec_labels": rec_labels,
        "donor_sums": donor_sums,
        "h_rec": h_rec,
        "h_don": h_don,
        "logits_before": logits_before,
        "recipient_correct": correct[rec],
    }
    return stats, payload


def test4_decompose(args, model, payload, B, B_rand):
    h_rec = payload["h_rec"]
    h_don = payload["h_don"]
    delta = h_don.float() - h_rec.float()
    delta_probe = project_onto(delta, B)
    delta_comp = delta - delta_probe
    delta_rand = match_norm_in_subspace(delta_probe, B_rand)

    edits = {
        "full_delta": delta,
        "probe_subspace_delta": delta_probe,
        "complement_delta": delta_comp,
        "random_norm_matched_to_probe": delta_rand,
    }
    rows = []
    for name, dlt in edits.items():
        logits_after = logits_with_delta(
            model, payload["rec_tokens"], h_rec, dlt, args.batch_size
        )
        stats = pair_metrics(
            payload["logits_before"],
            logits_after,
            payload["rec_labels"],
            payload["donor_sums"],
            h_rec,
            h_rec.float() + dlt,
            payload["recipient_correct"],
        )
        stats["edit"] = name
        stats["mean_delta_norm"] = float(dlt.norm(dim=-1).mean())
        rows.append(stats)
    return rows


def test5_phase(args, model, cfg, freqs, test_tokens, test_labels, test_h, W, B, decode_acc, B_rand, B_pca):
    import torch

    if decode_acc < DECODE_GATE:
        return [
            {
                "skipped": True,
                "reason": f"Fourier decode_acc={decode_acc:.4f} < {DECODE_GATE}; probe-based phase edit not attempted",
            }
        ], []

    p = cfg.p
    generator = torch.Generator().manual_seed(args.seed + 101)
    n = min(args.n_phase_examples, test_tokens.shape[0])
    idx = torch.randperm(test_tokens.shape[0], generator=generator)[:n]
    tokens = test_tokens[idx]
    labels = test_labels[idx]
    h = test_h[idx.cpu()]
    with torch.no_grad():
        logits_before = model(tokens)[:, -1, :]
    rows = []
    per_example = []
    for delta in args.deltas:
        sums = (tokens[:, 0] + tokens[:, 1]) % p
        target = rotate_true_latent(p, freqs, labels.cpu(), delta)
        y_shift = rotate_true_latent(p, freqs, sums.cpu(), delta)
        agree = float((target - y_shift).abs().max())
        delta_act = germ_target_delta(h, W, target)
        act_norm = h.float().norm(dim=-1, keepdim=True)
        d_norm = delta_act.float().norm(dim=-1, keepdim=True)
        scale = (act_norm / (d_norm + 1e-12)).clamp(max=1.0)
        delta_capped = delta_act * scale
        controls = {
            "probe_y_s_plus_delta": delta_capped,
            "matched_norm_random": match_norm_in_subspace(delta_capped, B_rand),
            "matched_norm_pca": match_norm_in_subspace(delta_capped, B_pca),
        }
        target_sum = (labels + delta) % p
        for name, dlt in controls.items():
            logits_after = logits_with_delta(model, tokens, h, dlt, args.batch_size)
            probs_b = softmax(logits_before)
            probs_a = softmax(logits_after)
            row = {
                "delta": delta,
                "edit": name,
                "n": int(n),
                "max_y_simulator_mismatch": agree,
                "mean_edit_norm": float(dlt.float().norm(dim=-1).mean()),
                "frac_edit_capped": float((scale < 1.0).float().mean()) if name.startswith("probe") else float("nan"),
                "mean_p_shifted_before": float(class_prob(probs_b, target_sum).mean()),
                "mean_p_shifted_after": float(class_prob(probs_a, target_sum).mean()),
                "mean_delta_p_shifted": float(
                    (class_prob(probs_a, target_sum) - class_prob(probs_b, target_sum)).mean()
                ),
                "frac_pred_shifted_before": float((logits_before.argmax(dim=-1) == target_sum).float().mean()),
                "frac_pred_shifted_after": float((logits_after.argmax(dim=-1) == target_sum).float().mean()),
                "mean_kl": float(kl_divergence(logits_before, logits_after).mean()),
                "mean_p_original_after": float(class_prob(probs_a, labels).mean()),
            }
            rows.append(row)
            per_example.append(row)
    return rows, per_example


def test6_graph(results_dir: Path) -> str:
    text = """# 07f computation-graph audit (no obstruction rebuild)

Official 07d remains NO_GO. This note only specifies the graph that a later
compatibility / obstruction model would have to use.

## One-layer Transformer, tokens `[a, b, '=']`

Positions: 0 = operand a, 1 = operand b, 2 = `'='`.

```
embed[0], embed[1], embed[2]          # hook_embed, each [d_model]
        \\         |          /
         \\        |         /
          v       v        v
     resid_pre[0,1,2]                 # layer 0: residual is the embeddings
          |       |        |            (plus positional embedding)
          +-------+--------+
                  |
        attention, 4 heads in PARALLEL
          Q,K,V from resid_pre at all positions
          hook_z: z[head] at pos 2 is [d_head]
          attn_out = concat_h(z_h) @ W_O     # linear, heads summed
                  |
            resid_mid[2] = resid_pre[2] + attn_out[2]
                  |
            MLP in:  resid_mid @ W_in + b_in     # linear
            MLP nl:  ReLU(pre)                   # hook_pre / hook_post, [d_mlp]
            MLP out: post @ W_out + b_out        # linear, [d_model]
                  |
            resid_post[2] = resid_mid[2] + mlp_out[2]
                  |
            logits = resid_post[2] @ W_U         # no LayerNorm in this run
```

## Invalid edges in the 07c/07d imposed chain

Imposed order:

`embed_pos0 → embed_pos1 → L0_resid_pre → L0_H0 → L0_H1 → L0_H2 → L0_H3 → L0_resid_mid → L0_mlp_post → L0_resid_post`

| Imposed edge | Why it is not a computational dependency |
| --- | --- |
| `embed_pos0 → embed_pos1` | Operand embeddings are siblings, not a path. |
| `embed_pos1 → L0_resid_pre` | `resid_pre` at `'='` is the `'='` embedding (plus pos), not a child of `b`. `b` reaches `'='` only through attention. |
| `L0_H0 → L0_H1 → L0_H2 → L0_H3` | Heads read `resid_pre` in parallel. None is an input to another. |
| `L0_H3 → L0_resid_mid` as if H3 were the attention output | `resid_mid = resid_pre + Σ_h z_h W_O_h`. Dropping H0–H2 from the edge is a graph error. |
| `L0_mlp_post → L0_resid_post` as a replacement of the residual | `resid_post = resid_mid + post @ W_out`. The skip from `resid_mid` is missing. |

## Maps that can be linear transports

These are architecture-linear, input-independent given the frozen weights:

- `resid_mid → mlp.hook_pre`: `W_in`
- `mlp.hook_post → mlp_out`: `W_out`
- `mlp_out → resid_post`: identity plus the `resid_mid` skip (the skip is not a map from `hook_post`)
- `concat(z_h) → attn_out`: `W_O`
- `resid_post → logits`: `W_U`

A germ-space ridge map on those pairs is at least the right *support*. It is still a
probe-space map, not the weight matrix itself.

## Maps that cannot be a single linear transport

- `embed[a], embed[b] → z_h['=']`: attention. Mixing weights are input-dependent (softmax of QK).
- `resid_pre → resid_mid`: same, plus the residual add.
- `mlp.hook_pre → mlp.hook_post`: ReLU, nonlinear.
- Any map from a single operand embedding to `y(a+b)`: the sum is not a function of one operand.

If obstruction is rebuilt, allowed edges are only the first list, plus optionally a
*nonlinear* attention transport that conditions on the other positions. Chaining
parallel heads or operand embeddings as if they were a path will manufacture
incompatibility that is not a computational residual.

## Consequence for 07c monitoring

Learned-vs-identity-vs-random at a single checkpoint can still show that *some*
ridge maps fit better than noise. It cannot show that the residual is incompatibility
of the modular algorithm along the true DAG. 07f does not rebuild O.
"""
    write_text(results_dir / "07f_computation_graph_audit.md", text)
    return text


def make_figures(figures_dir: Path, t1, t2_rows, t3_rows, t4_groups, t5_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [
        "ridge",
        "shuffled",
        "random 10-D",
        "ridge after abl.",
        "new probe on complement",
        "chance",
    ]
    vals = [
        t1["decode_acc_test"],
        t1["decode_acc_shuffled"],
        t1["decode_acc_random_subspace"],
        t1["decode_acc_original_probe_after_ablation"],
        t1["decode_acc_residual_probe"],
        t1["chance_decode_acc"],
    ]
    ax.bar(labels, vals, color=["#1f77b4", "#aaa", "#aaa", "#d62728", "#2ca02c", "#888"])
    ax.axhline(t1["chance_decode_acc"], color="black", lw=0.8, ls="--")
    ax.set_ylabel("held-out decode acc")
    ax.set_title("07f Test 1 — L0_mlp_post at step 9080")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures_dir / "07f_test1_decode.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for row in t2_rows:
        ax.scatter(row["mean_pert_norm"], row["acc_change"], label=row["method"])
        ax.annotate(row["method"], (row["mean_pert_norm"], row["acc_change"]), fontsize=7)
    ax.set_xlabel("mean ||Δh||")
    ax.set_ylabel("accuracy change")
    ax.set_title("07f Test 2 — damage vs perturbation size")
    ax.axhline(0, color="black", lw=0.6)
    fig.tight_layout()
    fig.savefig(figures_dir / "07f_test2_acc_vs_energy.png", dpi=150)
    plt.close(fig)

    if t3_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [r["condition"] for r in t3_rows]
        ax.bar(names, [r["mean_delta_p_donor"] for r in t3_rows])
        ax.set_ylabel("Δ P(donor sum)")
        ax.set_title("07f Test 3 — full mlp_post patch")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figures_dir / "07f_test3_full_patch.png", dpi=150)
        plt.close(fig)

    if t4_groups:
        fig, ax = plt.subplots(figsize=(7, 4))
        edits = ["full_delta", "probe_subspace_delta", "complement_delta", "random_norm_matched_to_probe"]
        x = range(len(edits))
        width = 0.35
        for i, (cond, rows) in enumerate(t4_groups.items()):
            by_edit = {r["edit"]: r["mean_delta_p_donor"] for r in rows}
            ax.bar([xx + i * width for xx in x], [by_edit.get(e, 0.0) for e in edits], width, label=cond)
        ax.set_xticks([xx + width / 2 for xx in x], edits, rotation=20)
        ax.set_ylabel("Δ P(donor sum)")
        ax.set_title("07f Test 4 — where the full-patch effect lives")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "07f_test4_delta_decomposition.png", dpi=150)
        plt.close(fig)

    t5_real = [r for r in t5_rows if "delta" in r and r.get("edit") == "probe_y_s_plus_delta"]
    if t5_real:
        fig, ax = plt.subplots(figsize=(6, 4))
        deltas = sorted({int(r["delta"]) for r in t5_real})
        for edit in ("probe_y_s_plus_delta", "matched_norm_random", "matched_norm_pca"):
            ys = []
            for d in deltas:
                match = [r for r in t5_rows if r.get("delta") == d and r.get("edit") == edit]
                ys.append(match[0]["mean_delta_p_shifted"] if match else 0.0)
            ax.plot(deltas, ys, marker="o", label=edit)
        ax.set_xlabel("Δ")
        ax.set_ylabel("Δ P(s+Δ)")
        ax.set_title("07f Test 5 — frequency-specific phase target")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "07f_test5_phase.png", dpi=150)
        plt.close(fig)


def decide(t1, t2_rows, t3_rows, t4_groups, t5_rows, t2_noop_ok: bool) -> tuple[str, str, str]:
    """Return (letter, title, explanation). Does not touch 07d NO_GO."""

    if not t2_noop_ok:
        return (
            "D",
            "Inconclusive",
            "The no-op replacement of cached L0_mlp_post did not leave logits/accuracy "
            "essentially unchanged, so later patch numbers cannot be trusted as hook effects.",
        )

    t3_diff = [r for r in t3_rows if r.get("condition", "").startswith("B_")]
    t3_same = [r for r in t3_rows if r.get("condition") == "A_same_sum"]
    if not t3_rows:
        return (
            "D",
            "Inconclusive",
            "No donor/recipient pairs could be constructed from the held-out set.",
        )

    mean_dP_diff = (
        sum(r["mean_delta_p_donor"] for r in t3_diff) / len(t3_diff) if t3_diff else 0.0
    )
    mean_frac_donor_after = (
        sum(r["frac_pred_donor_after"] for r in t3_diff) / len(t3_diff) if t3_diff else 0.0
    )
    mean_frac_donor_before = (
        sum(r["frac_pred_donor_before"] for r in t3_diff) / len(t3_diff) if t3_diff else 0.0
    )
    full_patch_targeted = mean_dP_diff >= 0.05 and mean_frac_donor_after > mean_frac_donor_before + 0.03
    logits_move = all(r["frac_logits_changed"] > 0.9 for r in t3_rows)

    if logits_move and not full_patch_targeted:
        return (
            "C",
            "Site not sufficient for targeted modular interchange",
            "Replacing the full 512-D mlp_post activation at '=' changes logits, but does not "
            "systematically move probability toward the donor's modular sum. The site is not a "
            "sufficient interchange register for a+b mod p at this checkpoint.",
        )

    if not logits_move:
        return (
            "D",
            "Inconclusive",
            "Full donor patches did not change logits. Either the hook did not execute or the "
            "cached activations are not the live forward tensor.",
        )

    # Full patch is targeted. Locate the effect.
    def edit_dP(groups, edit):
        vals = []
        for rows in groups.values():
            for r in rows:
                if r.get("edit") == edit:
                    vals.append(r["mean_delta_p_donor"])
        return sum(vals) / len(vals) if vals else 0.0

    dP_full = edit_dP(t4_groups, "full_delta")
    dP_probe = edit_dP(t4_groups, "probe_subspace_delta")
    dP_comp = edit_dP(t4_groups, "complement_delta")
    residual_decodes = bool(t1.get("residual_still_decodes"))

    # Energy-matched PCA: if still much more damaging than cyclic, that is not an
    # implementation bug; if energy matching collapses the PCA effect, 07d's headline
    # comparison was energy, not subspace identity.
    t2 = {r["method"]: r for r in t2_rows}
    cyc = t2.get("ridge_germ_project_out", {})
    pca_em = t2.get("pca_energy_matched", {})
    pca_full = t2.get("pca_same_rank_project_out", {})
    energy_confound = abs(pca_full.get("acc_change", 0.0)) > 0.2 and abs(
        pca_em.get("acc_change", 0.0)
    ) < abs(cyc.get("acc_change", 0.0)) + 0.05

    probe_carries = dP_full > 0.04 and dP_probe >= 0.5 * dP_full
    complement_carries = dP_full > 0.04 and dP_comp >= 0.5 * dP_full and dP_probe < 0.25 * dP_full

    if complement_carries:
        extra = (
            " The complement still linearly decodes the sum, so 07d removed one readout "
            "inside a larger encoding."
            if residual_decodes
            else " The complement does not linearly decode y(s) as well, but still carries "
            "the interchange effect, so the causal directions are not the ridge-germ basis."
        )
        return (
            "A",
            "Probe readout not causal",
            "Full same-hook donor patches move the recipient toward the donor sum, but the "
            "10-D ridge-germ component of the donor delta does not. The complement does."
            + extra,
        )

    if probe_carries:
        return (
            "B",
            "Probe/subspace implementation problem",
            "A genuine interchange in the ridge-germ subspace moves probability toward the "
            "donor sum, unlike 07d's min-norm germ-target edit. The 07d NO_GO still stands "
            "as an operational gate on that edit, but it did not test interchange. "
            + (
                "Energy-matched PCA also ceases to dominate, so the 0.03-vs-0.63 headline "
                "was perturbation size."
                if energy_confound
                else "Energy-matched PCA remains more damaging than cyclic ablation, so "
                "necessity-vs-PCA was not only an energy artefact."
            ),
        )

    if full_patch_targeted and not (probe_carries or complement_carries):
        return (
            "D",
            "Inconclusive",
            "Full patches are targeted, but the probe/complement split is not decisive "
            f"(ΔP_full={dP_full:.3f}, ΔP_probe={dP_probe:.3f}, ΔP_comp={dP_comp:.3f}). "
            "Need a larger pair sample or a rank scan of the complement.",
        )

    _ = t3_same, t5_rows
    return (
        "D",
        "Inconclusive",
        "Diagnostics did not cleanly separate a noncausal readout from a distributed "
        "causal encoding or a failed interchange.",
    )


def recommend(letter: str) -> str:
    if letter == "A":
        return (
            "Next justified experiment: a distributed-representation test (higher-rank "
            "cyclic / causal Jacobian / multi-site joint subspace), not a rebuilt "
            "obstruction score on the old 10-D germ. A corrected DAG is required before "
            "any new O(t) claim, but it is not sufficient on its own."
        )
    if letter == "B":
        return (
            "Next justified experiment: interchange-style edits in the subspace that "
            "actually carries the donor effect (the 07f decomposition), still at "
            "step 9080, then a corrected computational DAG for monitoring. Do not "
            "rerun 07d's min-norm grid."
        )
    if letter == "C":
        return (
            "Next justified experiment: move the interchange site to residual-space MLP "
            "output (`hook_mlp_out` / `resid_post`) or a joint attn+MLP patch. The "
            "post-ReLU 512-D vector at '=' is not a sufficient modular register here."
        )
    return (
        "Next justified experiment: fix or enlarge the failed positive control "
        "(no-op, pair count, or hook identity) before any new scientific claim. "
        "Do not interpret 07d as a mechanism result until then."
    )


def main() -> None:
    import torch

    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    figures_dir = ensure_dir(args.figures_dir)

    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Pass the 07d checkpoint, e.g. "
            "--checkpoint=/content/drive/MyDrive/ml_research_artifacts/toy_modular/checkpoints/step_0009080.pt"
        )

    model, cfg, step = load_checkpoint(path, device=args.device)
    if step != 9080:
        raise RuntimeError(f"Refusing to run 07f on step {step}. Checkpoint must be 9080.")
    model.eval()
    device = next(model.parameters()).device

    tokens, labels = make_dataset(cfg)
    (train_tokens, train_labels), (test_tokens, test_labels) = split_dataset(cfg, tokens, labels)
    train_tokens, train_labels = train_tokens.to(device), train_labels.to(device)
    test_tokens, test_labels = test_tokens.to(device), test_labels.to(device)
    freqs = key_frequencies(model, cfg.p, top_k=args.n_freqs)
    print(f"07f checkpoint step {step} device={device} freqs={freqs}")
    print("07d operational label remains NO_GO. This script does not change it.")

    with torch.no_grad():
        base_logits = model(test_tokens)[:, -1, :]
    base_pred = base_logits.argmax(dim=-1)
    base_correct = base_pred == test_labels
    print(
        f"test_acc={float(base_correct.float().mean()):.4f} "
        f"n_incorrect={int((~base_correct).sum())}"
    )

    print("Test 1 — representation at 9080")
    t1, cache = test1_representation(
        args, model, cfg, freqs, train_tokens, test_tokens, test_labels
    )
    write_csv(results_dir / "07f_test1_representation.csv", [t1])
    print(
        f"  decode={t1['decode_acc_test']:.4f} R2={t1['r2_test']:.4f} "
        f"shuffled={t1['decode_acc_shuffled']:.4f} random={t1['decode_acc_random_subspace']:.4f} "
        f"rank={t1['effective_rank']} energy={t1['test_energy_fraction_in_germ']:.4f}"
    )
    print(
        f"  after abl. original probe decode={t1['decode_acc_original_probe_after_ablation']:.4f} "
        f"lin_rel={t1['linear_probe_residual_relnorm_after_ablation']:.3e} "
        f"residual-probe decode={t1['decode_acc_residual_probe']:.4f}"
    )

    print("Test 2 — intervention controls")
    t2_rows, t2_extra = test2_controls(
        args, model, test_tokens, test_labels, base_logits, cache["test_h"], cache["B"], cache["train_h"]
    )
    write_csv(results_dir / "07f_test2_intervention_controls.csv", t2_rows)
    for row in t2_rows:
        print(
            f"  {row['method']}: dAcc={row['acc_change']:+.4f} "
            f"||dh||={row['mean_pert_norm']:.4f} energy={row['mean_energy_removed_frac']:.4f} "
            f"KL={row['mean_kl']:.4f}"
        )
    noop = next(r for r in t2_rows if r["method"] == "noop")
    t2_noop_ok = abs(noop["acc_change"]) < 1e-4 and noop["mean_kl"] < 1e-3

    print("Test 3 — full activation interchange")
    t3_rows = []
    payloads = {}
    rec_a, don_a = sample_pairs(
        test_tokens, test_labels, base_correct, 0, args.n_recipients, args.n_donors_each, args.seed, cfg.p
    )
    stats_a, payload_a = eval_pairs(
        model, test_tokens, test_labels, base_correct, rec_a, don_a, cache["test_h"], args.batch_size,
        extra={"condition": "A_same_sum", "delta": 0},
    )
    if stats_a:
        t3_rows.append(stats_a)
        payloads["A_same_sum"] = payload_a
        print(f"  A same-sum n={stats_a['n_pairs']} ΔP(donor)={stats_a['mean_delta_p_donor']:+.4f}")
    for delta in args.deltas:
        rec, don = sample_pairs(
            test_tokens, test_labels, base_correct, delta, args.n_recipients, args.n_donors_each, args.seed, cfg.p
        )
        stats, payload = eval_pairs(
            model, test_tokens, test_labels, base_correct, rec, don, cache["test_h"], args.batch_size,
            extra={"condition": f"B_delta_{delta}", "delta": delta},
        )
        if stats:
            t3_rows.append(stats)
            payloads[f"B_delta_{delta}"] = payload
            print(
                f"  B Δ={delta} n={stats['n_pairs']} ΔP(donor)={stats['mean_delta_p_donor']:+.4f} "
                f"frac_pred_donor {stats['frac_pred_donor_before']:.3f}->{stats['frac_pred_donor_after']:.3f}"
            )
    write_csv(results_dir / "07f_test3_full_patch.csv", t3_rows)

    print("Test 4 — decompose donor delta")
    t4_rows = []
    t4_groups: dict[str, list] = {}
    for cond, payload in payloads.items():
        rows = test4_decompose(args, model, payload, cache["B"], t2_extra["B_rand"])
        for row in rows:
            row["condition"] = cond
            t4_rows.append(row)
        t4_groups[cond] = rows
        brief = ", ".join(f"{r['edit']}={r['mean_delta_p_donor']:+.3f}" for r in rows)
        print(f"  {cond}: {brief}")
    write_csv(results_dir / "07f_test4_delta_decomposition.csv", t4_rows)

    print("Test 5 — frequency-specific phase target")
    t5_rows, _ = test5_phase(
        args,
        model,
        cfg,
        freqs,
        test_tokens,
        test_labels,
        cache["test_h"],
        cache["W"],
        cache["B"],
        t1["decode_acc_test"],
        t2_extra["B_rand"],
        t2_extra["B_pca"],
    )
    write_csv(results_dir / "07f_test5_phase_counterfactual.csv", t5_rows)
    for row in t5_rows:
        if row.get("skipped"):
            print(f"  skipped: {row['reason']}")
        elif row.get("edit") == "probe_y_s_plus_delta":
            print(
                f"  Δ={row['delta']} probe ΔP(s+Δ)={row['mean_delta_p_shifted']:+.4f} "
                f"pred {row['frac_pred_shifted_before']:.3f}->{row['frac_pred_shifted_after']:.3f}"
            )

    print("Test 6 — computation graph audit")
    test6_graph(results_dir)

    make_figures(figures_dir, t1, t2_rows, t3_rows, t4_groups, t5_rows)

    letter, title, explanation = decide(t1, t2_rows, t3_rows, t4_groups, t5_rows, t2_noop_ok)
    rec = recommend(letter)
    summary = {
        "official_07d_label": "NO_GO",
        "checkpoint_step": 9080,
        "decision": letter,
        "decision_title": title,
        "explanation": explanation,
        "recommendation": rec,
        "test1_decode_acc": t1["decode_acc_test"],
        "test1_residual_probe_decode": t1["decode_acc_residual_probe"],
        "test1_linear_probe_killed": t1["linear_probe_residual_relnorm_after_ablation"] < 1e-3,
        "test2_noop_ok": t2_noop_ok,
    }
    write_text(results_dir / "07f_decision.json", json.dumps(summary, indent=2) + "\n")
    md = "\n".join(
        [
            "# 07f causal-mechanism validation",
            "",
            "**Official 07d label: NO_GO (unchanged).**",
            "",
            f"**07f decision: {letter} — {title}**",
            "",
            explanation,
            "",
            "## Checkpoint",
            "",
            f"- step 9080, site `{SITE_ID}`, hook `{HOOK_MLP_POST}`, position {POSITION}",
            f"- Fourier decode acc {t1['decode_acc_test']:.4f} (shuffled {t1['decode_acc_shuffled']:.4f}, "
            f"random-subspace {t1['decode_acc_random_subspace']:.4f}, chance {t1['chance_decode_acc']:.4f})",
            f"- R² {t1['r2_test']:.4f}, effective rank {t1['effective_rank']}/10, "
            f"energy fraction in germ {t1['test_energy_fraction_in_germ']:.4f}",
            f"- original probe after ablation decode {t1['decode_acc_original_probe_after_ablation']:.4f} "
            f"(linear residual rel-norm {t1['linear_probe_residual_relnorm_after_ablation']:.3e})",
            f"- new train-only probe on complement decode {t1['decode_acc_residual_probe']:.4f}",
            "",
            "## These tests do not validate obstruction",
            "",
            "They ask whether a representation at one hook is a causal modular register.",
            "They are not a test of O_y, transports, or the monitoring hypothesis.",
            "",
            "## Recommendation",
            "",
            rec,
            "",
        ]
    )
    write_text(results_dir / "07f_decision_summary.md", md)
    print("\n" + md)
    print(f"saved tables under {results_dir}")
    print(f"saved figures under {figures_dir}")


if __name__ == "__main__":
    main()
