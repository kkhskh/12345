"""Experiment 0B: finish what 07g asked — modular structure and remaining mix sites.

Frozen checkpoint 9080. Does not retrain. Official 07d remains NO_GO.

07g Parts A/B at hook_post already identified the unembed channel T (decision B).
07h showed resid_mid and mlp.hook_pre can be overwritten, still not a +Δ operator.

This script does the two things the 07g prompt still required:

  Block 1 — Part C as written, at hook_post: hold out operands, test G consistency,
  G_0 ≈ I, and G_a G_b ≈ G_{a+b}. Also whether T's right singular vectors are Fourier
  on Z/113 (a property of T, not an adder claim).

  Block 2 — Part D as a measurement: resid_pre at pos 0/1/2, hook_z per head and
  all heads, attention patterns. Full donor patch first. Operand-copy next.
  Rank / G_Δ only if the full-patch gate passes.

No obstruction. No compatibility maps unless the decision is a real A.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.toy_modular import (
    fit_ridge,
    key_frequencies,
    latent_targets,
    load_checkpoint,
    make_dataset,
    match_norm_in_subspace,
    pca_subspace,
    random_subspace,
    split_dataset,
)

from run_07f_modular_causal_sanity_validation import (
    cache_mlp_post,
    ensure_dir,
    pair_metrics,
    project_onto,
    write_text,
)

from run_07g_modular_causal_subspace import (
    COMPOSE_PAIRS,
    RANKS,
    SHIFT_DELTAS,
    apply_G,
    collect_train_diffs,
    eval_delta_patch,
    fit_G,
    gather_pairs,
    orthonormalize,
    output_map_T,
    split_pool_indices,
    stratified,
    subspace_overlap,
    svd_map,
    take_pool,
    write_table,
)

from run_07h_modular_earlier_sites import (
    COMPACT_RECOV,
    COMPOSE_FRO,
    FULL_PATCH_DP,
    G_MODULAR_DP,
    eval_site_patch,
    logits_with_site,
    site_full_patch_ok,
)

G_RANKS = (16, 32, 128)
N_HOLDOUT = 20
OPERAND_SHIFTS = ((1, 0), (0, 1), (1, 1))
MIN_PAIRS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="07i finish 07g Part C/D at step 9080.")
    parser.add_argument("--checkpoint", default="artifacts/toy_modular/checkpoints/step_0009080.pt")
    parser.add_argument("--results-dir", default="artifacts/results/toy_modular")
    parser.add_argument("--figures-dir", default="artifacts/figures/toy_modular")
    parser.add_argument("--n-freqs", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--g-ridges", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.3)
    parser.add_argument("--n-recipients", type=int, default=96)
    parser.add_argument("--n-donors-each", type=int, default=2)
    parser.add_argument("--n-train-diff-recipients", type=int, default=80)
    parser.add_argument("--n-holdout", type=int, default=N_HOLDOUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def residue_in(values, holdout: set[int], p: int):
    import torch

    table = torch.zeros(p, dtype=torch.bool)
    for v in holdout:
        table[int(v) % p] = True
    idx = values.detach().cpu().long() % p
    return table[idx]


def holdout_set(p: int, n: int, seed: int, offset: int) -> set[int]:
    import torch

    n = min(n, p // 4)
    perm = torch.randperm(p, generator=torch.Generator().manual_seed(seed + offset))
    return {int(v) for v in perm[:n].tolist()}


def filter_pool(pool, mask):
    tokens, labels, correct, h = pool
    mask = mask.to(device=tokens.device)
    return tokens[mask], labels[mask], correct[mask], h[mask.detach().cpu()]


def cache_hook(model, tokens, hook_name: str, batch_size: int = 2048):
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start : start + batch_size]
            _, cache = model.run_with_cache(batch, names_filter=lambda n, h=hook_name: n == h)
            chunks.append(cache[hook_name].detach().cpu())
    return torch.cat(chunks, dim=0)


def logit_fourier_basis(p: int, n_freqs: int):
    import torch

    s = torch.arange(p).float()
    cols = []
    for k in range(1, n_freqs + 1):
        theta = 2.0 * math.pi * k * s / p
        cols.append(torch.cos(theta))
        cols.append(torch.sin(theta))
    return orthonormalize(torch.stack(cols, dim=1))


def t_right_sv_fourier(T_info, p: int, n_freqs: int = 16):
    import torch

    Vh = T_info["Vh"].float()
    s = T_info["s"].float()
    F = logit_fourier_basis(p, n_freqs)
    rows = []
    for i in range(min(16, Vh.shape[0])):
        v = Vh[i]
        dots = (F.T @ v)
        best = int(dots.abs().argmax())
        freq = best // 2 + 1
        rows.append(
            {
                "sv_index": i + 1,
                "singular_value": float(s[i]),
                "best_fourier_freq": freq,
                "best_abs_cos": float(dots.abs()[best]),
                "energy_frac": float((s[i] * s[i]) / (T_info["energy_total"] + 1e-12)),
            }
        )
    overlap_rows = []
    for k_sv, n_f in ((8, 4), (16, 8), (32, 16)):
        k_sv = min(k_sv, Vh.shape[0])
        Bv = orthonormalize(Vh[:k_sv].T)
        Bf = logit_fourier_basis(p, n_f)
        overlap_rows.append(
            {
                "n_right_sv": k_sv,
                "n_fourier_freqs": n_f,
                "fourier_dim": int(Bf.shape[1]),
                "subspace_overlap": subspace_overlap(Bv, Bf),
            }
        )
    return rows, overlap_rows


def select_ridge(model, train_pool, val_bundles, B, p, args, eval_fn, spec):
    best_ridge, best = args.g_ridges[0], -1e9
    for ridge in args.g_ridges:
        scores = []
        for delta in SHIFT_DELTAS:
            tr = gather_pairs(train_pool, delta, args.n_recipients, args.n_donors_each, args.seed + 40 * delta, p)
            va = val_bundles.get(delta)
            if tr is None or va is None:
                continue
            G = fit_G(tr["h_rec"].float() @ B, tr["h_don"].float() @ B, ridge)
            metrics, _, _ = eval_fn(model, va, apply_G(va["h_rec"], B, G) - va["h_rec"].float(), spec, args.batch_size)
            scores.append(metrics["mean_delta_p_donor"])
        if scores and sum(scores) / len(scores) > best:
            best, best_ridge = sum(scores) / len(scores), ridge
    return best_ridge


def g_on_bundle(model, bundle, B, G, eval_fn, spec, batch_size: int):
    metrics, _, _ = eval_fn(
        model, bundle, apply_G(bundle["h_rec"], B, G) - bundle["h_rec"].float(), spec, batch_size
    )
    C_r = bundle["h_rec"].float() @ B
    C_d = bundle["h_don"].float() @ B
    rel = float(((C_r @ G) - C_d).pow(2).mean().sqrt() / (C_d.pow(2).mean().sqrt() + 1e-12))
    return metrics, rel


def eval_hook_post(model, bundle, delta_h, spec, batch_size: int):
    return eval_delta_patch(model, bundle, delta_h, batch_size)


def eval_resid(model, bundle, delta_h, spec, batch_size: int):
    return eval_site_patch(model, bundle, delta_h, spec["hook_name"], spec["position"], batch_size)


def logits_with_z(model, tokens, replacement, position: int, heads, batch_size: int):
    import torch

    n_heads = int(model.cfg.n_heads)
    d_head = int(model.cfg.d_head)
    n_patch = n_heads if heads is None else len(heads)
    if replacement.ndim == 2:
        replacement = replacement.reshape(replacement.shape[0], n_patch, d_head)
    outs = []
    n = tokens.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = tokens[start:end]
            h_new = replacement[start:end]

            def hook(acts, hook, h=h_new, pos=position, hd=heads):  # noqa: ARG001
                hh = h.to(device=acts.device, dtype=acts.dtype)
                if hd is None:
                    acts[:, pos, :, :] = hh
                else:
                    for i, head in enumerate(hd):
                        acts[:, pos, head, :] = hh[:, i, :]
                return acts

            with model.hooks(fwd_hooks=[("blocks.0.attn.hook_z", hook)]):
                outs.append(model(batch)[:, -1, :].detach())
    return torch.cat(outs, dim=0)


def eval_z(model, bundle, delta_h, spec, batch_size: int):
    import torch

    with torch.no_grad():
        logits_b = model(bundle["tokens"])[:, -1, :]
    h_new = bundle["h_rec"].float() + delta_h.float()
    logits_a = logits_with_z(model, bundle["tokens"], h_new, spec["position"], spec["heads"], batch_size)
    row = stratified(
        logits_b,
        logits_a,
        bundle["labels"],
        bundle["donor_sums"],
        bundle["h_rec"],
        h_new,
        bundle["recip_correct"],
    )
    row["mean_applied_delta_norm"] = float(delta_h.float().norm(dim=-1).mean())
    return row, logits_b, logits_a


def logits_with_resid_multi(model, tokens, pos_to_h: dict[int, object], batch_size: int):
    import torch

    outs = []
    n = tokens.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = tokens[start:end]
            chunk = {pos: h[start:end] for pos, h in pos_to_h.items()}

            def hook(acts, hook, repl=chunk):  # noqa: ARG001
                for pos, h in repl.items():
                    acts[:, pos, :] = h.to(device=acts.device, dtype=acts.dtype)
                return acts

            with model.hooks(fwd_hooks=[("blocks.0.hook_resid_pre", hook)]):
                outs.append(model(batch)[:, -1, :].detach())
    return torch.cat(outs, dim=0)


def sample_ab_shift(tokens, labels, correct, da: int, db: int, n_recipients: int, seed: int, p: int):
    import torch

    generator = torch.Generator().manual_seed(seed + 31 * da + 17 * db)
    tokens_cpu = tokens.detach().cpu()
    correct_cpu = correct.detach().cpu()
    key_to_idx = {}
    for i in range(tokens_cpu.shape[0]):
        key_to_idx[(int(tokens_cpu[i, 0]), int(tokens_cpu[i, 1]))] = i
    cand_r, cand_d = [], []
    for i in range(tokens_cpu.shape[0]):
        a, b = int(tokens_cpu[i, 0]), int(tokens_cpu[i, 1])
        j = key_to_idx.get(((a + da) % p, (b + db) % p))
        if j is None or j == i:
            continue
        if bool(correct_cpu[j]):
            cand_r.append(i)
            cand_d.append(j)
    if not cand_r:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    perm = torch.randperm(len(cand_r), generator=generator)
    take = min(n_recipients, len(cand_r))
    rec = torch.tensor([cand_r[int(perm[k])] for k in range(take)], dtype=torch.long)
    don = torch.tensor([cand_d[int(perm[k])] for k in range(take)], dtype=torch.long)
    return rec, don


def sample_same_operand(tokens, labels, correct, axis: str, n_recipients: int, n_donors: int, seed: int, p: int):
    """axis='a': same b, different a. axis='b': same a, different b. Donors must be correct."""
    import torch

    generator = torch.Generator().manual_seed(seed + (3 if axis == "a" else 7))
    tokens_cpu = tokens.detach().cpu()
    correct_cpu = correct.detach().cpu()
    by_key: dict[int, list[int]] = {}
    col = 1 if axis == "a" else 0
    other = 0 if axis == "a" else 1
    for i in range(tokens_cpu.shape[0]):
        by_key.setdefault(int(tokens_cpu[i, col]), []).append(i)
    candidates = []
    for i in range(tokens_cpu.shape[0]):
        pool = [
            j
            for j in by_key.get(int(tokens_cpu[i, col]), [])
            if j != i and int(tokens_cpu[j, other]) != int(tokens_cpu[i, other]) and bool(correct_cpu[j])
        ]
        if pool:
            candidates.append(i)
    if not candidates:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    perm = torch.randperm(len(candidates), generator=generator)
    take_r = min(n_recipients, len(candidates))
    rec_idx, don_idx = [], []
    for k in range(take_r):
        r = candidates[int(perm[k])]
        pool = [
            j
            for j in by_key.get(int(tokens_cpu[r, col]), [])
            if j != r and int(tokens_cpu[j, other]) != int(tokens_cpu[r, other]) and bool(correct_cpu[j])
        ]
        if not pool:
            continue
        dperm = torch.randperm(len(pool), generator=generator)
        for t in range(min(n_donors, len(pool))):
            rec_idx.append(r)
            don_idx.append(pool[int(dperm[t])])
    return torch.tensor(rec_idx, dtype=torch.long), torch.tensor(don_idx, dtype=torch.long)


def bundle_from_idx(pool, rec, don, tag: int):
    if rec.numel() == 0:
        return None
    tokens, labels, correct, h = pool
    return {
        "delta": tag,
        "rec": rec,
        "don": don,
        "tokens": tokens[rec],
        "labels": labels[rec],
        "donor_sums": labels[don],
        "donor_tokens": tokens[don],
        "h_rec": h[rec.cpu()],
        "h_don": h[don.cpu()],
        "recip_correct": correct[rec].detach().cpu(),
    }


def split_bundle_halves(bundle, seed: int):
    import torch

    if bundle is None:
        return None, None
    n = int(bundle["rec"].numel())
    if n < 2 * MIN_PAIRS:
        return None, None
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    mid = n // 2

    def take(idx):
        out = {"delta": bundle["delta"]}
        for key, val in bundle.items():
            if key == "delta":
                continue
            if hasattr(val, "shape") and val.shape[0] == n:
                out[key] = val[idx]
            else:
                out[key] = val
        return out

    return take(perm[:mid]), take(perm[mid:])


def fro_rel(A, B):
    return float((A - B).pow(2).mean().sqrt() / (B.pow(2).mean().sqrt() + 1e-12))


def build_post_bases(train_h, train_tokens, train_diffs, T_info, p, freqs, ridge, seed, k: int):
    import torch

    d = train_h.shape[1]
    U_T = T_info["U"]
    _, _, Vh_diff = torch.linalg.svd(train_diffs.float(), full_matrices=False)
    y = latent_targets(train_tokens, p, freqs).cpu()
    W = fit_ridge(train_h, y, ridge)
    U_f, _, _ = torch.linalg.svd(W[:-1].float(), full_matrices=False)
    return {
        "T_top": U_T[:, : min(k, U_T.shape[1])],
        "pca_acts": pca_subspace(train_h, min(k, d)),
        "svd_donor_diff": Vh_diff[: min(k, Vh_diff.shape[0])].T,
        "random": random_subspace(d, min(k, d), seed),
        "fourier_probe": U_f[:, : min(k, int(U_f.shape[1]))],
    }


def build_site_bases(train_h, train_tokens, train_diffs, p, freqs, ridge, seed, k: int):
    import torch

    d = train_h.shape[1]
    _, _, Vh = torch.linalg.svd(train_diffs.float(), full_matrices=False)
    y = latent_targets(train_tokens, p, freqs).cpu()
    W = fit_ridge(train_h, y, ridge)
    U_f, _, _ = torch.linalg.svd(W[:-1].float(), full_matrices=False)
    return {
        "pca_acts": pca_subspace(train_h, min(k, d)),
        "svd_donor_diff": Vh[: min(k, Vh.shape[0])].T,
        "random": random_subspace(d, min(k, d), seed),
        "fourier_probe": U_f[:, : min(k, int(U_f.shape[1]))],
    }


# ----------------------------------------------------------------------------------
# Block 1
# ----------------------------------------------------------------------------------


def run_t_identity_check(model, T, test_bundle, batch_size: int) -> dict:
    if test_bundle is None:
        return {"max_rel_error_z": float("nan"), "n_pairs": 0}
    delta = test_bundle["h_don"].float() - test_bundle["h_rec"].float()
    _, logits_b, logits_a = eval_delta_patch(model, test_bundle, delta, batch_size)
    emp = (logits_a - logits_b).float().cpu()
    pred = delta @ T.float()
    abs_err = (emp - pred).abs()
    return {
        "n_pairs": int(delta.shape[0]),
        "max_abs_error_z": float(abs_err.max()),
        "max_rel_error_z": float(abs_err.max() / (emp.abs().max() + 1e-12)),
    }


def run_holdout_G(args, model, cfg, train_pool, val_pool, eval_pool, bases, ridge, eval_fn, spec):
    p = cfg.p
    rows = []
    axes = (("a", 11), ("b", 13), ("sum", 17))
    for axis, offset in axes:
        held = holdout_set(p, args.n_holdout, args.seed, offset)
        if axis == "a":
            fit_mask = ~residue_in(train_pool[0][:, 0], held, p)
            test_mask = residue_in(eval_pool[0][:, 0], held, p)
            val_mask = residue_in(val_pool[0][:, 0], held, p)
        elif axis == "b":
            fit_mask = ~residue_in(train_pool[0][:, 1], held, p)
            test_mask = residue_in(eval_pool[0][:, 1], held, p)
            val_mask = residue_in(val_pool[0][:, 1], held, p)
        else:
            fit_mask = ~residue_in(train_pool[1], held, p)
            test_mask = residue_in(eval_pool[1], held, p)
            val_mask = residue_in(val_pool[1], held, p)
        fit_pool = filter_pool(train_pool, fit_mask)
        test_pool = filter_pool(eval_pool, test_mask)
        held_val = filter_pool(val_pool, val_mask)
        print(f"  holdout {axis}: fit n={fit_pool[0].shape[0]} test n={test_pool[0].shape[0]} held={sorted(held)[:8]}...")
        for method, B in bases.items():
            for delta in SHIFT_DELTAS:
                tr = gather_pairs(fit_pool, delta, args.n_recipients, args.n_donors_each, args.seed + 50 * delta, p)
                te = gather_pairs(test_pool, delta, args.n_recipients, args.n_donors_each, args.seed + 200 + 9 * delta, p)
                va = gather_pairs(held_val, delta, min(48, args.n_recipients), args.n_donors_each, args.seed + 100 + 9 * delta, p)
                if tr is None or te is None:
                    continue
                G = fit_G(tr["h_rec"].float() @ B, tr["h_don"].float() @ B, ridge)
                for split, bundle in (("val", va), ("test", te)):
                    if bundle is None or bundle["rec"].numel() < MIN_PAIRS:
                        continue
                    metrics, rel = g_on_bundle(model, bundle, B, G, eval_fn, spec, args.batch_size)
                    row = {
                        "test": "holdout",
                        "holdout_axis": axis,
                        "n_holdout": len(held),
                        "method": method,
                        "rank": int(B.shape[1]),
                        "delta": delta,
                        "split": split,
                        "ridge": ridge,
                        "n_train_pairs": int(tr["rec"].numel()),
                        "repr_rel_rmse": rel,
                    }
                    row.update(metrics)
                    rows.append(row)
                    if split == "test" and method in {"T_top", "svd_donor_diff", "fourier_probe"}:
                        print(
                            f"  holdout-{axis} {method} k={B.shape[1]} Δ={delta} "
                            f"ΔP={metrics['mean_delta_p_donor']:+.3f} "
                            f"pred {metrics['frac_pred_donor_before']:.3f}->{metrics['frac_pred_donor_after']:.3f}"
                        )
    return rows


def run_consistency_G(args, model, cfg, train_pool, bases, ridge, eval_fn, spec):
    rows = []
    p = cfg.p
    for method, B in bases.items():
        for delta in SHIFT_DELTAS:
            full = gather_pairs(train_pool, delta, args.n_recipients * 2, args.n_donors_each, args.seed + 70 * delta, p)
            s1, s2 = split_bundle_halves(full, args.seed + 3 * delta)
            if s1 is None or s2 is None:
                continue
            G = fit_G(s1["h_rec"].float() @ B, s1["h_don"].float() @ B, ridge)
            for split, bundle in (("S1", s1), ("S2", s2)):
                metrics, rel = g_on_bundle(model, bundle, B, G, eval_fn, spec, args.batch_size)
                row = {
                    "test": "consistency",
                    "method": method,
                    "rank": int(B.shape[1]),
                    "delta": delta,
                    "split": split,
                    "ridge": ridge,
                    "repr_rel_rmse": rel,
                }
                row.update(metrics)
                rows.append(row)
                if split == "S2" and method in {"T_top", "svd_donor_diff", "fourier_probe"}:
                    print(
                        f"  S1→S2 {method} Δ={delta} ΔP={metrics['mean_delta_p_donor']:+.3f} "
                        f"repr_rel={rel:.3f}"
                    )
    return rows


def run_G0(args, model, cfg, train_pool, test_bundles, bases, ridge, eval_fn, spec):
    import torch

    rows = []
    p = cfg.p
    tr = gather_pairs(train_pool, 0, args.n_recipients, args.n_donors_each, args.seed + 3, p)
    if tr is None:
        return rows
    for method, B in bases.items():
        G = fit_G(tr["h_rec"].float() @ B, tr["h_don"].float() @ B, ridge)
        ident = torch.eye(B.shape[1], dtype=G.dtype)
        ident_err = fro_rel(G, ident)
        for delta, bundle in test_bundles.items():
            if bundle is None:
                continue
            metrics, rel = g_on_bundle(model, bundle, B, G, eval_fn, spec, args.batch_size)
            row = {
                "test": "G0",
                "method": method,
                "rank": int(B.shape[1]),
                "delta": delta,
                "split": "test",
                "ridge": ridge,
                "fro_to_identity": ident_err,
                "repr_rel_rmse": rel,
            }
            row.update(metrics)
            rows.append(row)
        print(f"  G_0 {method} k={B.shape[1]} ||G-I||_rel={ident_err:.3f}")
    return rows


def run_operand_composition(args, model, cfg, train_pool, val_pool, eval_pool, bases, ridge, eval_fn, spec):
    p = cfg.p
    rows = []
    compose_rows = []
    for method, B in bases.items():
        G_by = {}
        for da, db in OPERAND_SHIFTS:
            tag = 1000 * da + db
            tr_rec, tr_don = sample_ab_shift(train_pool[0], train_pool[1], train_pool[2], da, db, args.n_recipients, args.seed + da + 5 * db, p)
            tr = bundle_from_idx(train_pool, tr_rec, tr_don, tag)
            if tr is None:
                continue
            G = fit_G(tr["h_rec"].float() @ B, tr["h_don"].float() @ B, ridge)
            G_by[(da, db)] = G
            for split, pool, extra in (
                ("val", val_pool, 100),
                ("test", eval_pool, 200),
            ):
                rec, don = sample_ab_shift(pool[0], pool[1], pool[2], da, db, args.n_recipients, args.seed + extra + da + 5 * db, p)
                bundle = bundle_from_idx(pool, rec, don, tag)
                if bundle is None or bundle["rec"].numel() < MIN_PAIRS:
                    continue
                metrics, rel = g_on_bundle(model, bundle, B, G, eval_fn, spec, args.batch_size)
                row = {
                    "test": "operand_shift",
                    "method": method,
                    "rank": int(B.shape[1]),
                    "da": da,
                    "db": db,
                    "split": split,
                    "ridge": ridge,
                    "repr_rel_rmse": rel,
                    "n_train_pairs": int(tr["rec"].numel()),
                }
                row.update(metrics)
                rows.append(row)
                if split == "test" and method in {"T_top", "svd_donor_diff", "fourier_probe"}:
                    print(
                        f"  G_a={da} G_b={db} {method} ΔP={metrics['mean_delta_p_donor']:+.3f} "
                        f"pred {metrics['frac_pred_donor_before']:.3f}->{metrics['frac_pred_donor_after']:.3f}"
                    )
        if (1, 0) in G_by and (0, 1) in G_by and (1, 1) in G_by:
            Ga, Gb, Gab = G_by[(1, 0)], G_by[(0, 1)], G_by[(1, 1)]
            composed = Ga @ Gb
            f_err = fro_rel(composed, Gab)
            rec, don = sample_ab_shift(eval_pool[0], eval_pool[1], eval_pool[2], 1, 1, args.n_recipients, args.seed + 211, p)
            bundle = bundle_from_idx(eval_pool, rec, don, 1001)
            if bundle is not None and bundle["rec"].numel() >= MIN_PAIRS:
                met_c, _, _ = eval_fn(
                    model,
                    bundle,
                    apply_G(apply_G(bundle["h_rec"], B, Ga), B, Gb) - bundle["h_rec"].float(),
                    spec,
                    args.batch_size,
                )
                met_d, _, _ = eval_fn(
                    model,
                    bundle,
                    apply_G(bundle["h_rec"], B, Gab) - bundle["h_rec"].float(),
                    spec,
                    args.batch_size,
                )
                compose_rows.append(
                    {
                        "test": "operand_compose",
                        "method": method,
                        "rank": int(B.shape[1]),
                        "delta_a": "a+1",
                        "delta_b": "b+1",
                        "delta_c": "a+1,b+1",
                        "fro_rel_error": f_err,
                        "compose_mean_delta_p_donor": met_c["mean_delta_p_donor"],
                        "direct_mean_delta_p_donor": met_d["mean_delta_p_donor"],
                        "compose_frac_pred_donor_after": met_c["frac_pred_donor_after"],
                        "direct_frac_pred_donor_after": met_d["frac_pred_donor_after"],
                    }
                )
                print(
                    f"  G_a G_b vs G_{{a+b}} {method} fro={f_err:.3f} "
                    f"compose ΔP={met_c['mean_delta_p_donor']:+.3f} direct ΔP={met_d['mean_delta_p_donor']:+.3f}"
                )
    return rows, compose_rows


def run_sum_composition(args, model, cfg, train_pool, test_bundles, bases, ridge, eval_fn, spec):
    rows = []
    p = cfg.p
    for method, B in bases.items():
        G_by = {}
        for delta in SHIFT_DELTAS:
            tr = gather_pairs(train_pool, delta, args.n_recipients, args.n_donors_each, args.seed + 40 * delta, p)
            if tr is None:
                continue
            G_by[delta] = fit_G(tr["h_rec"].float() @ B, tr["h_don"].float() @ B, ridge)
        for a, b, c in COMPOSE_PAIRS:
            if a not in G_by or b not in G_by or c not in G_by:
                continue
            bundle = test_bundles.get(c)
            if bundle is None:
                continue
            Ga, Gb, Gc = G_by[a], G_by[b], G_by[c]
            f_err = fro_rel(Ga @ Gb, Gc)
            met_c, _, _ = eval_fn(
                model,
                bundle,
                apply_G(apply_G(bundle["h_rec"], B, Ga), B, Gb) - bundle["h_rec"].float(),
                spec,
                args.batch_size,
            )
            met_d, _, _ = eval_fn(
                model,
                bundle,
                apply_G(bundle["h_rec"], B, Gc) - bundle["h_rec"].float(),
                spec,
                args.batch_size,
            )
            rows.append(
                {
                    "test": "sum_compose",
                    "method": method,
                    "rank": int(B.shape[1]),
                    "delta_a": a,
                    "delta_b": b,
                    "delta_c": c,
                    "fro_rel_error": f_err,
                    "compose_mean_delta_p_donor": met_c["mean_delta_p_donor"],
                    "direct_mean_delta_p_donor": met_d["mean_delta_p_donor"],
                }
            )
    return rows


# ----------------------------------------------------------------------------------
# Block 2
# ----------------------------------------------------------------------------------


def attn_pattern_rows(patterns) -> list[dict]:
    rows = []
    # patterns: [n, n_heads, dest, src]
    dest = 2
    mean = patterns[:, :, dest, :].float().mean(dim=0)
    for head in range(mean.shape[0]):
        row = {
            "head": head,
            "dest": dest,
            "w_pos0_a": float(mean[head, 0]),
            "w_pos1_b": float(mean[head, 1]),
            "w_pos2_eq": float(mean[head, 2]),
        }
        rows.append(row)
        print(
            f"  head {head} '=' → a={row['w_pos0_a']:.3f} b={row['w_pos1_b']:.3f} "
            f"equals={row['w_pos2_eq']:.3f}"
        )
    return rows


def run_full_patches(model, spec, bundles, batch_size: int, split: str, eval_fn, site_id: str):
    rows = []
    for delta, bundle in bundles.items():
        if bundle is None:
            continue
        delta_h = bundle["h_don"].float() - bundle["h_rec"].float()
        metrics, _, _ = eval_fn(model, bundle, delta_h, spec, batch_size)
        metrics.update(
            {
                "site_id": site_id,
                "split": split,
                "delta": bundle["delta"],
                "method": "full_patch",
                "variant": "full",
            }
        )
        rows.append(metrics)
        print(
            f"  {site_id} {split} Δ={delta} n={metrics['n_pairs']} "
            f"ΔP={metrics['mean_delta_p_donor']:+.3f} "
            f"pred {metrics['frac_pred_donor_before']:.3f}->{metrics['frac_pred_donor_after']:.3f}"
        )
    return rows


def run_rank_and_G_site(args, model, cfg, spec, site_id, freqs, train_pool, val_bundles, test_bundles, full_cache, eval_fn):
    train_tokens, _, _, train_h = train_pool
    d = train_h.shape[1]
    train_diffs = collect_train_diffs(train_pool, cfg.p, args.n_train_diff_recipients, args.n_donors_each, args.seed)
    print(f"  {site_id} train diffs {tuple(train_diffs.shape)} d={d}")
    b_rows = []
    ranks = tuple(k for k in RANKS if k <= d)
    for k in ranks:
        bases = build_site_bases(train_h, train_tokens, train_diffs, cfg.p, freqs, args.ridge, args.seed, k)
        for split, bundles in (("val", val_bundles), ("test", test_bundles)):
            for delta, bundle in bundles.items():
                if bundle is None:
                    continue
                full_row = full_cache[(split, delta)]
                dh = bundle["h_don"].float() - bundle["h_rec"].float()
                for method, B in bases.items():
                    for variant, applied in (
                        ("proj", project_onto(dh, B)),
                        ("norm_matched_full", match_norm_in_subspace(dh, B)),
                    ):
                        metrics, _, _ = eval_fn(model, bundle, applied, spec, args.batch_size)
                        recov = (
                            metrics["mean_delta_p_donor"] / full_row["mean_delta_p_donor"]
                            if abs(full_row["mean_delta_p_donor"]) > 1e-6
                            else float("nan")
                        )
                        row = {
                            "site_id": site_id,
                            "split": split,
                            "delta": delta,
                            "method": method,
                            "rank": k,
                            "variant": variant,
                            "basis_dim": int(B.shape[1]),
                            "frac_full_dP_recovered": recov,
                        }
                        row.update(metrics)
                        b_rows.append(row)

    selected = {}
    for method in sorted({r["method"] for r in b_rows}):
        scores = []
        for k in ranks:
            vals = [
                float(r["mean_delta_p_donor"])
                for r in b_rows
                if r["method"] == method
                and r["split"] == "val"
                and r["variant"] == "proj"
                and int(r["rank"]) == k
                and int(r["delta"]) != 0
            ]
            if vals:
                scores.append((sum(vals) / len(vals), k))
        if scores:
            selected[method] = max(scores)[1]
            print(f"  {site_id} val rank {method}={selected[method]}")

    bases_c = {}
    for method, k in selected.items():
        bases_c[method] = build_site_bases(
            train_h, train_tokens, train_diffs, cfg.p, freqs, args.ridge, args.seed, k
        )[method]
    ridge = args.g_ridges[0]
    if bases_c:
        probe = "svd_donor_diff" if "svd_donor_diff" in bases_c else next(iter(bases_c))
        ridge = select_ridge(model, train_pool, val_bundles, bases_c[probe], cfg.p, args, eval_fn, spec)
        print(f"  {site_id} G ridge={ridge}")
    c_rows, compose_rows = [], []
    for method, B in bases_c.items():
        G_by = {}
        for delta in SHIFT_DELTAS:
            tr = gather_pairs(train_pool, delta, args.n_recipients, args.n_donors_each, args.seed + 40 * delta, cfg.p)
            if tr is None:
                continue
            G = fit_G(tr["h_rec"].float() @ B, tr["h_don"].float() @ B, ridge)
            G_by[delta] = G
            for split, bundle in (("val", val_bundles.get(delta)), ("test", test_bundles.get(delta))):
                if bundle is None:
                    continue
                metrics, rel = g_on_bundle(model, bundle, B, G, eval_fn, spec, args.batch_size)
                row = {
                    "site_id": site_id,
                    "method": method,
                    "rank": int(B.shape[1]),
                    "delta": delta,
                    "split": split,
                    "ridge": ridge,
                    "repr_rel_rmse": rel,
                }
                row.update(metrics)
                c_rows.append(row)
                if split == "test" and method in {"svd_donor_diff", "fourier_probe", "pca_acts"}:
                    print(
                        f"  {site_id} {method} Δ={delta} G_ΔP={metrics['mean_delta_p_donor']:+.3f} "
                        f"pred {metrics['frac_pred_donor_before']:.3f}->{metrics['frac_pred_donor_after']:.3f}"
                    )
        for a, b, c in COMPOSE_PAIRS:
            if a not in G_by or b not in G_by or c not in G_by:
                continue
            bundle = test_bundles.get(c)
            if bundle is None:
                continue
            Ga, Gb, Gc = G_by[a], G_by[b], G_by[c]
            met_c, _, _ = eval_fn(
                model, bundle, apply_G(apply_G(bundle["h_rec"], B, Ga), B, Gb) - bundle["h_rec"].float(), spec, args.batch_size
            )
            met_d, _, _ = eval_fn(
                model, bundle, apply_G(bundle["h_rec"], B, Gc) - bundle["h_rec"].float(), spec, args.batch_size
            )
            compose_rows.append(
                {
                    "site_id": site_id,
                    "method": method,
                    "rank": int(B.shape[1]),
                    "delta_a": a,
                    "delta_b": b,
                    "delta_c": c,
                    "fro_rel_error": fro_rel(Ga @ Gb, Gc),
                    "compose_mean_delta_p_donor": met_c["mean_delta_p_donor"],
                    "direct_mean_delta_p_donor": met_d["mean_delta_p_donor"],
                }
            )
    return b_rows, c_rows, compose_rows, selected


def operand_copy_ok(rows: list[dict]) -> bool:
    if not rows:
        return False
    dP = sum(float(r["mean_delta_p_donor"]) for r in rows) / len(rows)
    pred = sum(float(r["frac_pred_donor_after"]) - float(r["frac_pred_donor_before"]) for r in rows) / len(rows)
    return dP >= FULL_PATCH_DP and pred >= 0.10


def run_operand_copy(args, model, cfg, eval_pool, resid_pre_eval, z_eval, batch_size: int):
    """resid_pre pos0/1 and both; hook_z per head and all heads, a-swap and b-swap."""
    import torch

    p = cfg.p
    tokens, labels, correct, _ = eval_pool
    rows = []

    def eval_logits(logits_b, logits_a, rec, donor_sums, h_r, h_d, site_id, swap):
        row = stratified(
            logits_b,
            logits_a,
            labels[rec],
            donor_sums,
            h_r,
            h_d,
            correct[rec].detach().cpu(),
        )
        row.update({"site_id": site_id, "swap": swap, "split": "test"})
        return row

    for axis, site_id, pos in (("a", "L0_resid_pre_pos0", 0), ("b", "L0_resid_pre_pos1", 1)):
        rec, don = sample_same_operand(tokens, labels, correct, axis, args.n_recipients, args.n_donors_each, args.seed + 300, p)
        if rec.numel() < MIN_PAIRS:
            continue
        h_rec = resid_pre_eval[rec.cpu(), pos, :]
        h_don = resid_pre_eval[don.cpu(), pos, :]
        with torch.no_grad():
            logits_b = model(tokens[rec])[:, -1, :]
        logits_a = logits_with_site(model, tokens[rec], "blocks.0.hook_resid_pre", pos, h_don, batch_size)
        row = eval_logits(logits_b, logits_a, rec, labels[don], h_rec, h_don, site_id, f"{axis}_swap")
        rows.append(row)
        print(
            f"  operand-copy {site_id} {axis}_swap n={row['n_pairs']} "
            f"ΔP={row['mean_delta_p_donor']:+.3f} "
            f"pred {row['frac_pred_donor_before']:.3f}->{row['frac_pred_donor_after']:.3f}"
        )

    rec_a, don_a = sample_same_operand(tokens, labels, correct, "a", args.n_recipients, 1, args.seed + 310, p)
    if rec_a.numel() >= MIN_PAIRS:
        rec = rec_a
        # for each recipient, find a b-swap donor (same a, different b)
        rec_b, don_b = [], []
        tokens_cpu = tokens.detach().cpu()
        correct_cpu = correct.detach().cpu()
        by_a: dict[int, list[int]] = {}
        for i in range(tokens_cpu.shape[0]):
            by_a.setdefault(int(tokens_cpu[i, 0]), []).append(i)
        keep_rec, keep_da, keep_db = [], [], []
        for k in range(rec.numel()):
            r = int(rec[k].item())
            a = int(tokens_cpu[r, 0])
            b = int(tokens_cpu[r, 1])
            pool_b = [j for j in by_a.get(a, []) if j != r and int(tokens_cpu[j, 1]) != b and bool(correct_cpu[j])]
            if not pool_b:
                continue
            keep_rec.append(r)
            keep_da.append(int(don_a[k].item()))
            keep_db.append(pool_b[0])
        if len(keep_rec) >= MIN_PAIRS:
            rec_t = torch.tensor(keep_rec, dtype=torch.long)
            da_t = torch.tensor(keep_da, dtype=torch.long)
            db_t = torch.tensor(keep_db, dtype=torch.long)
            target = (tokens[da_t, 0] + tokens[db_t, 1]) % p
            h0 = resid_pre_eval[da_t.cpu(), 0, :]
            h1 = resid_pre_eval[db_t.cpu(), 1, :]
            with torch.no_grad():
                logits_b = model(tokens[rec_t])[:, -1, :]
            logits_a = logits_with_resid_multi(model, tokens[rec_t], {0: h0, 1: h1}, batch_size)
            h_r = resid_pre_eval[rec_t.cpu(), 0, :]
            row = stratified(
                logits_b,
                logits_a,
                labels[rec_t],
                target,
                h_r,
                h0,
                correct[rec_t].detach().cpu(),
            )
            row.update({"site_id": "L0_resid_pre_pos0+pos1", "swap": "both", "split": "test"})
            rows.append(row)
            print(
                f"  operand-copy resid_pre both n={row['n_pairs']} "
                f"ΔP={row['mean_delta_p_donor']:+.3f} "
                f"pred {row['frac_pred_donor_before']:.3f}->{row['frac_pred_donor_after']:.3f}"
            )

    n_heads = int(z_eval.shape[1])
    head_lists = [((h,), f"L0_H{h}") for h in range(n_heads)] + [(None, "L0_z_all")]
    for axis in ("a", "b"):
        rec, don = sample_same_operand(tokens, labels, correct, axis, args.n_recipients, args.n_donors_each, args.seed + 320, p)
        if rec.numel() < MIN_PAIRS:
            continue
        for heads, site_id in head_lists:
            if heads is None:
                h_rec = z_eval[rec.cpu()].reshape(rec.numel(), -1)
                h_don = z_eval[don.cpu()].reshape(don.numel(), -1)
            else:
                h_rec = z_eval[rec.cpu()][:, heads[0], :]
                h_don = z_eval[don.cpu()][:, heads[0], :]
            with torch.no_grad():
                logits_b = model(tokens[rec])[:, -1, :]
            logits_a = logits_with_z(model, tokens[rec], h_don, 2, heads, batch_size)
            row = eval_logits(logits_b, logits_a, rec, labels[don], h_rec, h_don, site_id, f"{axis}_swap")
            rows.append(row)
            print(
                f"  operand-copy {site_id} {axis}_swap n={row['n_pairs']} "
                f"ΔP={row['mean_delta_p_donor']:+.3f} "
                f"pred {row['frac_pred_donor_before']:.3f}->{row['frac_pred_donor_after']:.3f}"
            )
    return rows


def decide_site(site_id: str, full_ok: bool, b_rows, c_rows, compose_rows) -> tuple[str, str]:
    if not full_ok:
        return (
            "SITE_FAIL",
            f"{site_id}: full donor patch does not move the recipient to the donor sum.",
        )
    test = [
        r
        for r in b_rows
        if r.get("split") == "test" and r.get("variant") == "proj" and int(r.get("delta", 0)) != 0
    ]

    def recov(method, k):
        vals = [float(r["frac_full_dP_recovered"]) for r in test if r["method"] == method and int(r["rank"]) == k]
        return sum(vals) / len(vals) if vals else float("nan")

    d32, f8, r16 = recov("svd_donor_diff", 32), recov("fourier_probe", 8), recov("random", 16)
    g_test = [r for r in c_rows if r.get("split") == "test"]

    def gmean(method):
        vals = [float(r["mean_delta_p_donor"]) for r in g_test if r["method"] == method]
        return sum(vals) / len(vals) if vals else float("nan")

    g_d, g_r = gmean("svd_donor_diff"), gmean("random")
    compose = [r for r in compose_rows if r["method"] in {"svd_donor_diff", "pca_acts"}]
    fro = sum(float(r["fro_rel_error"]) for r in compose) / len(compose) if compose else 1e9
    compact = d32 == d32 and d32 >= COMPACT_RECOV and d32 > (r16 if r16 == r16 else 0) + 0.15
    modular = g_d == g_d and g_d >= G_MODULAR_DP and g_d > (g_r if g_r == g_r else 0) + 0.15 and fro < COMPOSE_FRO
    if compact and modular:
        return (
            "A",
            f"{site_id}: compact subspace recovers interchange and G composes "
            f"(diff32={d32:.3f}, G_Δ={g_d:.3f}, fro={fro:.3f}).",
        )
    return (
        "B",
        f"{site_id}: full patch works, but this is not a modular shift operator "
        f"(diff32={d32:.3f}, Fourier8={f8:.3f}, G_Δ={g_d:.3f}, fro={fro:.3f}).",
    )


def mean_or_nan(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def decide_all(t_id, holdout_rows, cons_rows, g0_rows, op_rows, op_comp, site_verdicts, copy_rows) -> tuple[str, str, str]:
    if t_id.get("max_rel_error_z", 1.0) != t_id.get("max_rel_error_z", 1.0) or t_id.get("max_rel_error_z", 1.0) > 1e-2:
        return "D", "Inconclusive", f"T identity failed or missing (max rel {t_id.get('max_rel_error_z')})."

    def g_hold(method: str) -> float:
        vals = [
            float(r["mean_delta_p_donor"])
            for r in holdout_rows
            if r.get("split") == "test" and r["method"] == method and int(r.get("rank", 99)) <= 32
        ]
        return mean_or_nan(vals)

    g_hold_t, g_hold_d = g_hold("T_top"), g_hold("svd_donor_diff")
    op_comp_compact = [r for r in op_comp if r["method"] in {"T_top", "svd_donor_diff"} and int(r.get("rank", 99)) <= 32]
    fro_op = mean_or_nan([float(r["fro_rel_error"]) for r in op_comp_compact])
    g_ab = mean_or_nan([float(r["compose_mean_delta_p_donor"]) for r in op_comp_compact])
    compact_modular = (
        (g_hold_t >= G_MODULAR_DP or g_hold_d >= G_MODULAR_DP)
        and fro_op == fro_op
        and fro_op < COMPOSE_FRO
        and g_ab == g_ab
        and g_ab >= G_MODULAR_DP
    )
    if compact_modular:
        return (
            "A",
            "Compact modular causal representation",
            f"Held-out G_Δ and G_a G_b cleared the bar on a rank≤32 basis "
            f"(holdout T={g_hold_t:.3f} diff={g_hold_d:.3f}, compose fro={fro_op:.3f}, G_aG_b ΔP={g_ab:.3f}).",
        )
    if any(v[0] == "A" for v in site_verdicts):
        text = " ".join(v[1] for v in site_verdicts if v[0] == "A")
        return "A", "Compact modular causal representation", text

    copy_ok = operand_copy_ok(copy_rows)
    any_write = any(v[0] == "B" for v in site_verdicts) or copy_ok
    all_fail = site_verdicts and all(v[0] == "SITE_FAIL" for v in site_verdicts) and not copy_ok
    if all_fail:
        return (
            "C",
            "Distributed or no replaceable mix site",
            "Full patches at resid_pre and hook_z failed, and operand-copy did not move the answer. "
            f"Held-out G stayed weak (T={g_hold_t:.3f}, diff={g_hold_d:.3f}, G_aG_b fro={fro_op:.3f}).",
        )
    if any_write:
        extra = " ".join(v[1] for v in site_verdicts)
        return (
            "B",
            "Causal answer channel or operand copy, not a +Δ / addition operator",
            f"Held-out G_Δ T={g_hold_t:.3f} diff={g_hold_d:.3f}; G_a G_b fro={fro_op:.3f} ΔP={g_ab:.3f} "
            f"(bar was ΔP≥{G_MODULAR_DP} and fro<{COMPOSE_FRO}). "
            f"Operand-copy gate={'PASS' if copy_ok else 'FAIL'}. {extra}",
        )
    return (
        "D",
        "Inconclusive",
        "Pair counts or controls do not separate A/B/C "
        f"(holdout T={g_hold_t:.3f}, sites={ [v[0] for v in site_verdicts] }).",
    )


def recommend(letter: str) -> str:
    if letter == "A":
        return (
            "A compact modular object appeared with held-out operand composition. "
            "Next: obstruction on that subspace and the real DAG, still at 9080. "
            "Do not use the 10-D Fourier germ."
        )
    if letter == "B":
        return (
            "The remaining mix sites can copy or overwrite an answer, but there is still "
            "no held-out +Δ / G_a G_b operator. Do not rebuild O on T, Fourier, or donor-SVD. "
            "Compatibility maps are not justified."
        )
    if letter == "C":
        return (
            "The sum is not a replaceable activation at resid_pre or hook_z either. "
            "A local residual obstruction at 9080 is not justified."
        )
    return "Controls were inconclusive. Enlarge pairs or fix a hook identity before any new O claim."


def make_figures(figures_dir: Path, sv_rows, holdout_rows, pattern_rows, full_rows, copy_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if sv_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["sv_index"] for r in sv_rows], [r["best_abs_cos"] for r in sv_rows], marker="o")
        ax.set_xlabel("right singular vector of T")
        ax.set_ylabel("max |cos| with a Fourier mode on Z/113")
        ax.set_title("07i — T right-SV vs Fourier (not an adder claim)")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(figures_dir / "07i_T_rightSV_fourier.png", dpi=150)
        plt.close(fig)

    test_h = [r for r in holdout_rows if r.get("split") == "test" and int(r.get("rank", 99)) <= 32]
    if test_h:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in sorted({r["method"] for r in test_h}):
            sub = [r for r in test_h if r["method"] == method]
            xs = list(range(len(sub)))
            ax.plot(xs, [float(r["mean_delta_p_donor"]) for r in sub], marker="o", label=method)
        ax.axhline(G_MODULAR_DP, color="black", ls="--", lw=0.8, label="G bar")
        ax.set_ylabel("held-out G ΔP(donor)")
        ax.set_title("07i Block 1 — operand/sum holdout G")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "07i_holdout_G.png", dpi=150)
        plt.close(fig)

    if pattern_rows:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        heads = [r["head"] for r in pattern_rows]
        w = [[r["w_pos0_a"], r["w_pos1_b"], r["w_pos2_eq"]] for r in pattern_rows]
        im = ax.imshow(w, aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(len(heads)), [f"H{h}" for h in heads])
        ax.set_xticks([0, 1, 2], ["a", "b", "="])
        ax.set_title("07i — mean attention at '=' (descriptive)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(figures_dir / "07i_attn_pattern.png", dpi=150)
        plt.close(fig)

    test_full = [r for r in full_rows if r.get("split") == "test" and int(r.get("delta", 0)) != 0]
    if test_full:
        fig, ax = plt.subplots(figsize=(8, 4))
        sites = sorted({r["site_id"] for r in test_full})
        for site in sites:
            sub = [r for r in test_full if r["site_id"] == site]
            xs = [int(r["delta"]) for r in sub]
            ys = [float(r["mean_delta_p_donor"]) for r in sub]
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order], marker="o", label=site)
        ax.axhline(FULL_PATCH_DP, color="black", ls="--", lw=0.8)
        ax.set_xlabel("Δ")
        ax.set_ylabel("full-patch ΔP")
        ax.set_title("07i Block 2 — full donor patch")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures_dir / "07i_full_patch.png", dpi=150)
        plt.close(fig)

    if copy_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"{r['site_id']}\n{r.get('swap','')}" for r in copy_rows]
        ax.bar(range(len(copy_rows)), [float(r["mean_delta_p_donor"]) for r in copy_rows])
        ax.axhline(FULL_PATCH_DP, color="black", ls="--", lw=0.8)
        ax.set_xticks(range(len(copy_rows)), labels, fontsize=6, rotation=30, ha="right")
        ax.set_ylabel("ΔP toward intended sum")
        ax.set_title("07i — operand-copy")
        fig.tight_layout()
        fig.savefig(figures_dir / "07i_operand_copy.png", dpi=150)
        plt.close(fig)


def make_sum_bundles(pool, args, p, tag: int):
    out = {}
    for delta in (0,) + SHIFT_DELTAS:
        out[delta] = gather_pairs(pool, delta, args.n_recipients, args.n_donors_each, args.seed + tag + 9 * delta, p)
    return out


def main() -> None:
    import torch

    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    figures_dir = ensure_dir(args.figures_dir)
    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Pass step_0009080.pt.")

    model, cfg, step = load_checkpoint(path, device=args.device)
    if step != 9080:
        raise RuntimeError(f"Refusing step {step}; 07i is frozen at 9080.")
    if cfg.n_layers != 1:
        raise RuntimeError("07i assumes the audited one-layer model.")
    model.eval()
    device = next(model.parameters()).device
    print(f"07i checkpoint step {step} device={device}")
    print("07d remains NO_GO. 07g A/B at hook_post already done (B). 07h resid_mid/hook_pre already B.")

    tokens, labels = make_dataset(cfg)
    (train_tokens, train_labels), (test_tokens, test_labels) = split_dataset(cfg, tokens, labels)
    train_tokens, train_labels = train_tokens.to(device), train_labels.to(device)
    test_tokens, test_labels = test_tokens.to(device), test_labels.to(device)
    freqs = key_frequencies(model, cfg.p, top_k=args.n_freqs)
    with torch.no_grad():
        train_ok = model(train_tokens)[:, -1, :].argmax(-1) == train_labels
        test_ok = model(test_tokens)[:, -1, :].argmax(-1) == test_labels
    print(f"train_acc={float(train_ok.float().mean()):.4f} test_acc={float(test_ok.float().mean()):.4f}")

    val_idx, eval_idx = split_pool_indices(test_tokens.shape[0], args.val_frac, args.val_seed)
    train_h = cache_mlp_post(model, train_tokens)
    test_h = cache_mlp_post(model, test_tokens)
    train_pool = (train_tokens, train_labels, train_ok, train_h)
    val_pool = take_pool(test_tokens, test_labels, test_ok, test_h, val_idx)
    eval_pool = take_pool(test_tokens, test_labels, test_ok, test_h, eval_idx)

    T = output_map_T(model)
    T_info = svd_map(T, 1e-6)
    print(f"T shape={tuple(T.shape)} rank={T_info['numerical_rank']} PR={T_info['participation_ratio']:.2f}")

    print("\n=== Block 1 — T right-SV vs Fourier on Z/113 ===")
    sv_rows, sv_overlap = t_right_sv_fourier(T_info, cfg.p)
    for r in sv_rows[:5]:
        print(f"  SV{r['sv_index']} s={r['singular_value']:.4f} best_k={r['best_fourier_freq']} |cos|={r['best_abs_cos']:.3f}")
    for r in sv_overlap:
        print(f"  overlap top-{r['n_right_sv']} V vs {r['n_fourier_freqs']} freqs: {r['subspace_overlap']:.3f}")
    write_table(results_dir / "07i_T_rightSV_fourier.csv", sv_rows)
    write_table(results_dir / "07i_T_rightSV_fourier_overlap.csv", sv_overlap)

    val_b = make_sum_bundles(val_pool, args, cfg.p, 100)
    test_b = make_sum_bundles(eval_pool, args, cfg.p, 200)
    t_id = run_t_identity_check(model, T, test_b.get(1), args.batch_size)
    print(f"  T identity Δ=1 max_rel={t_id.get('max_rel_error_z', float('nan')):.3e}")
    write_table(results_dir / "07i_T_identity.csv", [t_id])

    train_diffs = collect_train_diffs(train_pool, cfg.p, args.n_train_diff_recipients, args.n_donors_each, args.seed)
    spec_post = {"kind": "mlp_post"}
    holdout_rows, cons_rows, g0_rows, op_rows, op_comp, sum_comp = [], [], [], [], [], []
    print("\n=== Block 1 — holdout / consistency / G_0 / G_a G_b ===")
    for k in G_RANKS:
        if k > train_h.shape[1]:
            continue
        bases = build_post_bases(train_h, train_tokens, train_diffs, T_info, cfg.p, freqs, args.ridge, args.seed, k)
        probe = bases["T_top"]
        ridge = select_ridge(model, train_pool, val_b, probe, cfg.p, args, eval_hook_post, spec_post)
        print(f"  rank {k} G ridge={ridge}")
        holdout_rows.extend(run_holdout_G(args, model, cfg, train_pool, val_pool, eval_pool, bases, ridge, eval_hook_post, spec_post))
        cons_rows.extend(run_consistency_G(args, model, cfg, train_pool, bases, ridge, eval_hook_post, spec_post))
        g0_rows.extend(run_G0(args, model, cfg, train_pool, test_b, bases, ridge, eval_hook_post, spec_post))
        op_r, op_c = run_operand_composition(args, model, cfg, train_pool, val_pool, eval_pool, bases, ridge, eval_hook_post, spec_post)
        op_rows.extend(op_r)
        op_comp.extend(op_c)
        sum_comp.extend(run_sum_composition(args, model, cfg, train_pool, test_b, bases, ridge, eval_hook_post, spec_post))

    write_table(results_dir / "07i_holdout_G.csv", holdout_rows)
    write_table(results_dir / "07i_consistency_G.csv", cons_rows)
    write_table(results_dir / "07i_G0.csv", g0_rows)
    write_table(results_dir / "07i_operand_shift_G.csv", op_rows)
    write_table(results_dir / "07i_operand_composition.csv", op_comp)
    write_table(results_dir / "07i_sum_composition.csv", sum_comp)

    print("\n=== Block 2 — attention patterns (descriptive) ===")
    eval_tokens = eval_pool[0]
    patterns = cache_hook(model, eval_tokens, "blocks.0.attn.hook_pattern")
    pattern_rows = attn_pattern_rows(patterns)
    write_table(results_dir / "07i_attn_pattern.csv", pattern_rows)

    print("\n=== Block 2 — cache resid_pre and hook_z ===")
    train_resid = cache_hook(model, train_tokens, "blocks.0.hook_resid_pre")
    test_resid = cache_hook(model, test_tokens, "blocks.0.hook_resid_pre")
    train_z = cache_hook(model, train_tokens, "blocks.0.attn.hook_z")[:, 2, :, :]
    test_z = cache_hook(model, test_tokens, "blocks.0.attn.hook_z")[:, 2, :, :]
    resid_pre_eval = test_resid[eval_idx.cpu()]
    z_eval = test_z[eval_idx.cpu()]

    site_specs = []
    for pos, name in ((0, "L0_resid_pre_pos0"), (1, "L0_resid_pre_pos1"), (2, "L0_resid_pre_pos2")):
        site_specs.append(
            {
                "site_id": name,
                "kind": "resid",
                "hook_name": "blocks.0.hook_resid_pre",
                "position": pos,
                "heads": None,
                "eval_fn": eval_resid,
                "slice_train": train_resid[:, pos, :],
                "slice_test": test_resid[:, pos, :],
            }
        )
    n_heads = int(train_z.shape[1])
    for h in range(n_heads):
        site_specs.append(
            {
                "site_id": f"L0_H{h}",
                "kind": "z",
                "hook_name": "blocks.0.attn.hook_z",
                "position": 2,
                "heads": (h,),
                "eval_fn": eval_z,
                "slice_train": train_z[:, h, :],
                "slice_test": test_z[:, h, :],
            }
        )
    site_specs.append(
        {
            "site_id": "L0_z_all",
            "kind": "z",
            "hook_name": "blocks.0.attn.hook_z",
            "position": 2,
            "heads": None,
            "eval_fn": eval_z,
            "slice_train": train_z.reshape(train_z.shape[0], -1),
            "slice_test": test_z.reshape(test_z.shape[0], -1),
        }
    )

    all_full, all_b, all_c, all_comp = [], [], [], []
    site_verdicts = []
    for spec in site_specs:
        site_id = spec["site_id"]
        print(f"\n=== {site_id} ===")
        t_pool = (train_tokens, train_labels, train_ok, spec["slice_train"])
        v_pool = take_pool(test_tokens, test_labels, test_ok, spec["slice_test"], val_idx)
        e_pool = take_pool(test_tokens, test_labels, test_ok, spec["slice_test"], eval_idx)
        v_b, e_b = make_sum_bundles(v_pool, args, cfg.p, 100), make_sum_bundles(e_pool, args, cfg.p, 200)
        patch_spec = {k: spec[k] for k in spec if k not in {"slice_train", "slice_test", "eval_fn", "site_id"}}
        eval_fn = spec["eval_fn"]
        full_val = run_full_patches(model, patch_spec, v_b, args.batch_size, "val", eval_fn, site_id)
        full_test = run_full_patches(model, patch_spec, e_b, args.batch_size, "test", eval_fn, site_id)
        all_full.extend(full_val + full_test)
        ok = site_full_patch_ok(full_test)
        print(f"  full-patch gate: {'PASS' if ok else 'FAIL'}")
        if not ok:
            site_verdicts.append(decide_site(site_id, False, [], [], []))
            continue
        full_cache = {(r["split"], r["delta"]): r for r in full_val + full_test}
        b_rows, c_rows, comp, _ = run_rank_and_G_site(
            args, model, cfg, patch_spec, site_id, freqs, t_pool, v_b, e_b, full_cache, eval_fn
        )
        all_b.extend(b_rows)
        all_c.extend(c_rows)
        all_comp.extend(comp)
        site_verdicts.append(decide_site(site_id, True, b_rows, c_rows, comp))
        print(f"  decision {site_verdicts[-1][0]}: {site_verdicts[-1][1]}")

    print("\n=== Block 2 — operand-copy ===")
    eval_pool_tokens = (eval_tokens, eval_pool[1], eval_pool[2], eval_pool[3])
    copy_rows = run_operand_copy(args, model, cfg, eval_pool_tokens, resid_pre_eval, z_eval, args.batch_size)

    write_table(results_dir / "07i_full_patch.csv", all_full)
    write_table(results_dir / "07i_rank_limited.csv", all_b)
    write_table(results_dir / "07i_shift_operators.csv", all_c)
    write_table(results_dir / "07i_site_composition.csv", all_comp)
    write_table(results_dir / "07i_operand_copy.csv", copy_rows)
    make_figures(figures_dir, sv_rows, holdout_rows, pattern_rows, all_full, copy_rows)

    letter, title, explanation = decide_all(t_id, holdout_rows, cons_rows, g0_rows, op_rows, op_comp, site_verdicts, copy_rows)
    rec = recommend(letter)
    md = "\n".join(
        [
            "# 07i operand composition and remaining mix sites",
            "",
            "**Official 07d label: NO_GO (unchanged).**",
            "",
            f"**07i decision: {letter} — {title}**",
            "",
            explanation,
            "",
            "Block 1 is the 07g Part C test with unseen operands and G_a G_b. "
            "Block 2 measures resid_pre and hook_z. Rank / G_Δ ran only where the full-patch gate passed. "
            "Attention patterns are descriptive. Linear maps through attention/ReLU are not identities.",
            "This is not an obstruction test.",
            "",
            "## Recommendation",
            "",
            rec,
            "",
        ]
    )
    write_text(results_dir / "07i_decision_summary.md", md)
    write_text(
        results_dir / "07i_decision.json",
        json.dumps(
            {
                "official_07d_label": "NO_GO",
                "checkpoint_step": 9080,
                "T_identity": t_id,
                "decision": letter,
                "decision_title": title,
                "explanation": explanation,
                "site_verdicts": [{"letter": a, "text": b} for a, b in site_verdicts],
                "operand_copy_gate": operand_copy_ok(copy_rows),
                "recommendation": rec,
            },
            indent=2,
        )
        + "\n",
    )
    print("\n" + md)
    print(f"saved {results_dir}")
    print(f"saved {figures_dir}")


if __name__ == "__main__":
    main()
