"""Experiment 07j: localize where modular addition is computed.

Frozen checkpoint 9080. Does not retrain. Does not rebuild obstruction.
Official 07d remains NO_GO. 07g–07i remain B.

Question: where do a and b become (a+b) mod p — attention (H1), MLP/ReLU (H2),
or both (H3)? Decoding is information, not computation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.toy_modular import (
    apply_ridge,
    fit_ridge,
    load_checkpoint,
    make_dataset,
    split_dataset,
)

from run_07f_modular_causal_sanity_validation import ensure_dir, write_text
from run_07g_modular_causal_subspace import split_pool_indices, write_table
from run_07i_modular_operand_composition import holdout_set, residue_in

POS = 2
HOOKS = (
    "blocks.0.hook_resid_pre",
    "blocks.0.attn.hook_pattern",
    "blocks.0.attn.hook_z",
    "blocks.0.hook_attn_out",
    "blocks.0.hook_resid_mid",
    "blocks.0.mlp.hook_pre",
    "blocks.0.mlp.hook_post",
    "blocks.0.hook_mlp_out",
    "blocks.0.hook_resid_post",
)
INFO_SITES = (
    "resid_pre",
    "attn_out",
    "resid_mid",
    "mlp_pre",
    "mlp_post",
    "mlp_out",
    "resid_post",
)
SITE_HOOK = {
    "resid_pre": "blocks.0.hook_resid_pre",
    "attn_out": "blocks.0.hook_attn_out",
    "resid_mid": "blocks.0.hook_resid_mid",
    "mlp_pre": "blocks.0.mlp.hook_pre",
    "mlp_post": "blocks.0.mlp.hook_post",
    "mlp_out": "blocks.0.hook_mlp_out",
    "resid_post": "blocks.0.hook_resid_post",
}
IDENTITY_TOL = 1e-4
N_BOOT = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="07j localize modular addition at step 9080.")
    parser.add_argument("--checkpoint", default="artifacts/toy_modular/checkpoints/step_0009080.pt")
    parser.add_argument("--results-dir", default="artifacts/results/toy_modular")
    parser.add_argument("--figures-dir", default="artifacts/figures/toy_modular")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.3)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--probe-hidden", type=int, default=32)
    parser.add_argument("--n-holdout", type=int, default=20)
    parser.add_argument("--n-quad-val", type=int, default=40)
    parser.add_argument("--n-quad-test", type=int, default=80)
    parser.add_argument("--n-causal", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cache-batch", type=int, default=1024)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def one_hot(labels, p: int):
    import torch

    lab = labels.detach().cpu().long()
    y = torch.zeros(lab.shape[0], p)
    y[range(lab.shape[0]), lab] = 1.0
    return y


def bootstrap_mean(values, n_boot: int, seed: int) -> tuple[float, float, float]:
    import torch

    v = values.detach().float().cpu().reshape(-1)
    if v.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    g = torch.Generator().manual_seed(seed)
    n = int(v.numel())
    means = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        means.append(float(v[idx].mean()))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return float(v.mean()), lo, hi


def softmax(logits):
    import torch

    z = logits.float()
    z = z - z.amax(dim=-1, keepdim=True)
    e = torch.exp(z)
    return e / e.sum(dim=-1, keepdim=True)


def class_prob(probs, labels):
    lab = labels.detach().cpu().long()
    return probs.detach().cpu()[range(lab.shape[0]), lab]


def true_margin(logits, labels):
    import torch

    z = logits.float().cpu()
    lab = labels.detach().cpu().long()
    correct = z[range(lab.shape[0]), lab]
    z2 = z.clone()
    z2[range(lab.shape[0]), lab] = -1e9
    return correct - z2.max(dim=-1).values


def kl_div(p_logits, q_logits):
    import torch

    p = softmax(p_logits).clamp_min(1e-12)
    q = softmax(q_logits).clamp_min(1e-12)
    return (p * (p.log() - q.log())).sum(dim=-1)


def weight_maps(model):
    """Return maps so r @ Win + bin = pre, relu(pre) @ Wout + bout = mlp_out, r @ WU + bU = logits."""
    d_model = int(model.cfg.d_model)
    d_mlp = int(model.cfg.d_mlp)
    p = int(model.cfg.d_vocab_out)
    W_in = model.blocks[0].mlp.W_in.detach().float().cpu()
    W_out = model.blocks[0].mlp.W_out.detach().float().cpu()
    W_U = model.W_U.detach().float().cpu()
    if tuple(W_in.shape) == (d_model, d_mlp):
        Win = W_in
    elif tuple(W_in.shape) == (d_mlp, d_model):
        Win = W_in.T
    else:
        raise RuntimeError(f"Unexpected W_in {tuple(W_in.shape)}")
    if tuple(W_out.shape) == (d_mlp, d_model):
        Wout = W_out
    elif tuple(W_out.shape) == (d_model, d_mlp):
        Wout = W_out.T
    else:
        raise RuntimeError(f"Unexpected W_out {tuple(W_out.shape)}")
    if tuple(W_U.shape) == (d_model, p):
        WU = W_U
    elif tuple(W_U.shape) == (p, d_model):
        WU = W_U.T
    else:
        raise RuntimeError(f"Unexpected W_U {tuple(W_U.shape)}")
    b_in = model.blocks[0].mlp.b_in.detach().float().cpu()
    b_out = model.blocks[0].mlp.b_out.detach().float().cpu()
    b_U = model.b_U.detach().float().cpu() if hasattr(model, "b_U") else None
    return {
        "Win": Win,
        "Wout": Wout,
        "WU": WU,
        "b_in": b_in,
        "b_out": b_out,
        "b_U": b_U,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "p": p,
        "W_in_raw": tuple(W_in.shape),
        "W_out_raw": tuple(W_out.shape),
        "W_U_raw": tuple(W_U.shape),
    }


def mlp_pre(r, w):
    return r.float() @ w["Win"] + w["b_in"]


def mlp_from_pre(u, w, gates=None):
    import torch

    g = (u > 0).float() if gates is None else gates.float()
    return (u * g) @ w["Wout"] + w["b_out"]


def logits_from_resid(r, w):
    z = r.float() @ w["WU"]
    if w["b_U"] is not None:
        z = z + w["b_U"]
    return z


def cache_hooks(model, tokens, batch_size: int):
    import torch

    buckets = {name: [] for name in HOOKS}
    logits = []
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start : start + batch_size]
            out, cache = model.run_with_cache(batch, names_filter=lambda n: n in buckets)
            logits.append(out[:, -1, :].detach().cpu())
            for name in HOOKS:
                buckets[name].append(cache[name].detach().cpu())
    packed = {name: torch.cat(chunks, dim=0) for name, chunks in buckets.items()}
    packed["logits"] = torch.cat(logits, dim=0)
    return packed


def site_vec(cache, site: str, position: int = POS):
    if site == "logits":
        return cache["logits"].float()
    return cache[SITE_HOOK[site]][:, position, :].float()


def verify_forward(cache, w, tokens) -> list[dict]:
    import torch

    pos = POS
    r_pre = cache["blocks.0.hook_resid_pre"][:, pos, :].float()
    a_out = cache["blocks.0.hook_attn_out"][:, pos, :].float()
    r_mid = cache["blocks.0.hook_resid_mid"][:, pos, :].float()
    u = cache["blocks.0.mlp.hook_pre"][:, pos, :].float()
    g = cache["blocks.0.mlp.hook_post"][:, pos, :].float()
    m_out = cache["blocks.0.hook_mlp_out"][:, pos, :].float()
    r_post = cache["blocks.0.hook_resid_post"][:, pos, :].float()
    logits = cache["logits"].float()

    u_hat = mlp_pre(r_mid, w)
    g_hat = torch.relu(u_hat)
    m_hat = g_hat @ w["Wout"] + w["b_out"]
    mid_hat = r_pre + a_out
    post_hat = r_mid + m_out
    z_hat = logits_from_resid(r_post, w)

    checks = [
        ("resid_mid = resid_pre + attn_out", r_mid, mid_hat),
        ("mlp_pre = resid_mid @ Win + bin", u, u_hat),
        ("mlp_post = ReLU(mlp_pre)", g, g_hat),
        ("mlp_out = ReLU(u) @ Wout + bout", m_out, m_hat),
        ("resid_post = resid_mid + mlp_out", r_post, post_hat),
        ("logits = resid_post @ WU + bU", logits, z_hat),
    ]
    rows = []
    worst = 0.0
    for name, emp, pred in checks:
        err = (emp - pred).abs()
        rel = float(err.max() / (emp.abs().max() + 1e-12))
        worst = max(worst, rel)
        rows.append(
            {
                "identity": name,
                "emp_shape": str(tuple(emp.shape)),
                "max_abs_error": float(err.max()),
                "mean_abs_error": float(err.mean()),
                "max_rel_error": rel,
                "n": int(emp.shape[0]),
            }
        )
        print(f"  {name}: max_rel={rel:.3e} max_abs={float(err.max()):.3e}")
    rows.append(
        {
            "identity": "weight_orientation",
            "emp_shape": f"Win{tuple(w['Win'].shape)} Wout{tuple(w['Wout'].shape)} WU{tuple(w['WU'].shape)}",
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "n": int(tokens.shape[0]),
            "W_in_raw": str(w["W_in_raw"]),
            "W_out_raw": str(w["W_out_raw"]),
            "W_U_raw": str(w["W_U_raw"]),
        }
    )
    if worst > IDENTITY_TOL:
        raise RuntimeError(f"Forward identities failed (max rel {worst:.3e}). Stopping.")
    for name in HOOKS:
        rows.append(
            {
                "identity": f"hook_present:{name}",
                "emp_shape": str(tuple(cache[name].shape)),
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "max_rel_error": 0.0,
                "n": int(cache[name].shape[0]),
            }
        )
    return rows


def pair_index(tokens):
    table = {}
    tok = tokens.detach().cpu()
    for i in range(tok.shape[0]):
        table[(int(tok[i, 0]), int(tok[i, 1]))] = i
    return table


def sample_quadruples(tokens, n: int, seed: int, p: int):
    import torch

    table = pair_index(tokens)
    keys = list(table.keys())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(keys), generator=g)
    quads = []
    seen = set()
    for k in perm.tolist():
        a, b = keys[k]
        if len(quads) >= n:
            break
        for _ in range(8):
            a2 = int(torch.randint(0, p, (1,), generator=g).item())
            b2 = int(torch.randint(0, p, (1,), generator=g).item())
            if a2 == a or b2 == b:
                continue
            corners = ((a, b), (a2, b), (a, b2), (a2, b2))
            if any(c not in table for c in corners):
                continue
            sig = tuple(sorted(corners))
            if sig in seen:
                continue
            seen.add(sig)
            quads.append({name: table[c] for name, c in zip(("ab", "apb", "abp", "apbp"), corners)})
            quads[-1].update({"a": a, "ap": a2, "b": b, "bp": b2})
            break
    return quads


def coverage_row(tokens, labels, p: int, tag: str) -> dict:
    tok = tokens.detach().cpu()
    lab = labels.detach().cpu()
    return {
        "split": tag,
        "n": int(tok.shape[0]),
        "n_unique_a": int(tok[:, 0].unique().numel()),
        "n_unique_b": int(tok[:, 1].unique().numel()),
        "n_unique_sum": int(lab.unique().numel()),
        "p": p,
    }


def fit_linear_probe(X, labels, p: int, ridge: float):
    return fit_ridge(X.float().cpu(), one_hot(labels, p), ridge)


def fit_nonlinear_probe(X, labels, p: int, ridge: float, hidden: int, seed: int):
    import torch

    g = torch.Generator().manual_seed(seed)
    R = torch.randn(X.shape[1], hidden, generator=g)
    R = R / (R.norm(dim=0, keepdim=True) + 1e-12)
    b = torch.randn(hidden, generator=g) * 0.1
    feat = torch.relu(X.float().cpu() @ R + b)
    W = fit_ridge(feat, one_hot(labels, p), ridge)
    return {"R": R, "b": b, "W": W}


def apply_nonlinear(probe, X):
    import torch

    feat = torch.relu(X.float().cpu() @ probe["R"] + probe["b"])
    return apply_ridge(probe["W"], feat)


def decode_acc(scores, labels) -> float:
    pred = scores.argmax(dim=-1).cpu()
    return float((pred == labels.detach().cpu().long()).float().mean())


def run_information(args, p, train_cache, val_cache, eval_cache, train_tok, val_tok, eval_tok, train_lab, val_lab, eval_lab):
    import torch

    rows = []
    held_a = holdout_set(p, args.n_holdout, args.seed, 11)
    targets = {
        "sum": (train_lab, val_lab, eval_lab),
        "a": (train_tok[:, 0], val_tok[:, 0], eval_tok[:, 0]),
        "b": (train_tok[:, 1], val_tok[:, 1], eval_tok[:, 1]),
    }
    for site in INFO_SITES:
        Xtr, Xva, Xte = site_vec(train_cache, site), site_vec(val_cache, site), site_vec(eval_cache, site)
        for target, (ytr, yva, yte) in targets.items():
            W = fit_linear_probe(Xtr, ytr, p, args.ridge)
            acc_va = decode_acc(apply_ridge(W, Xva), yva)
            acc_te = decode_acc(apply_ridge(W, Xte), yte)
            perm = torch.randperm(ytr.shape[0], generator=torch.Generator().manual_seed(args.seed + 3))
            Wsh = fit_linear_probe(Xtr, ytr.cpu()[perm], p, args.ridge)
            acc_sh = decode_acc(apply_ridge(Wsh, Xte), yte)
            nl = fit_nonlinear_probe(Xtr, ytr, p, args.ridge, args.probe_hidden, args.seed + 5)
            acc_nl = decode_acc(apply_nonlinear(nl, Xte), yte)
            fit_mask = ~residue_in(train_tok[:, 0], held_a, p)
            te_mask = residue_in(eval_tok[:, 0], held_a, p)
            acc_hold = float("nan")
            if int(fit_mask.sum()) >= 64 and int(te_mask.sum()) >= 32:
                Wh = fit_linear_probe(Xtr[fit_mask.cpu()], ytr.cpu()[fit_mask.cpu()], p, args.ridge)
                acc_hold = decode_acc(apply_ridge(Wh, Xte[te_mask.cpu()]), yte.cpu()[te_mask.cpu()])
            row = {
                "site": site,
                "target": target,
                "linear_acc_val": acc_va,
                "linear_acc_test": acc_te,
                "linear_acc_shuffled": acc_sh,
                "nonlinear_acc_test": acc_nl,
                "linear_acc_holdout_a": acc_hold,
                "n_train": int(Xtr.shape[0]),
                "n_test": int(Xte.shape[0]),
                "dim": int(Xtr.shape[1]),
                "n_holdout_a": len(held_a),
            }
            rows.append(row)
            if target == "sum":
                print(
                    f"  {site} sum linear={acc_te:.3f} nl={acc_nl:.3f} shuffle={acc_sh:.3f} "
                    f"holdout_a={acc_hold:.3f}"
                )
    return rows


def intervention_metrics(logits_b, logits_a, true_labels, target_labels, h_b, h_a, recip_correct):
    import torch

    pb, pa = softmax(logits_b), softmax(logits_a)
    pred_b = logits_b.argmax(dim=-1).cpu()
    pred_a = logits_a.argmax(dim=-1).cpu()
    tgt = target_labels.detach().cpu().long()
    true = true_labels.detach().cpu().long()
    ok = recip_correct.detach().cpu().bool()
    row = {
        "n": int(tgt.shape[0]),
        "frac_recip_correct": float(ok.float().mean()) if ok.numel() else float("nan"),
        "acc_true_before": float((pred_b == true).float().mean()),
        "acc_true_after": float((pred_a == true).float().mean()),
        "acc_target_before": float((pred_b == tgt).float().mean()),
        "acc_target_after": float((pred_a == tgt).float().mean()),
        "mean_p_target_before": float(class_prob(pb, tgt).mean()),
        "mean_p_target_after": float(class_prob(pa, tgt).mean()),
        "mean_delta_p_target": float((class_prob(pa, tgt) - class_prob(pb, tgt)).mean()),
        "mean_margin_target_before": float(true_margin(logits_b, tgt).mean()),
        "mean_margin_target_after": float(true_margin(logits_a, tgt).mean()),
        "mean_kl": float(kl_div(logits_b, logits_a).mean()),
        "mean_pert_norm": float((h_a.float() - h_b.float()).reshape(h_b.shape[0], -1).norm(dim=-1).mean()),
    }
    if int(ok.sum()) >= 8:
        row["correct_acc_target_after"] = float((pred_a[ok] == tgt[ok]).float().mean())
        row["correct_delta_p_target"] = float((class_prob(pa, tgt)[ok] - class_prob(pb, tgt)[ok]).mean())
    if int((~ok).sum()) >= 8:
        row["incorrect_acc_target_after"] = float((pred_a[~ok] == tgt[~ok]).float().mean())
        row["incorrect_delta_p_target"] = float((class_prob(pa, tgt)[~ok] - class_prob(pb, tgt)[~ok]).mean())
    _ = torch
    return row


def forward_replace(model, tokens, repls: list[tuple[str, int, object]], batch_size: int):
    """repls: (hook_name, position, tensor [n, ...])."""
    import torch

    outs = []
    n = tokens.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = tokens[start:end]
            chunk = [(name, pos, h[start:end]) for name, pos, h in repls]

            def make_hook(h, pos):
                def hook(acts, hook, hh=h, pp=pos):  # noqa: ARG001
                    acts[:, pp, ...] = hh.to(device=acts.device, dtype=acts.dtype)
                    return acts

                return hook

            hooks = [(name, make_hook(h, pos)) for name, pos, h in chunk]
            with model.hooks(fwd_hooks=hooks):
                outs.append(model(batch)[:, -1, :].detach())
    return torch.cat(outs, dim=0)


def logits_frozen_gate(model, tokens, r_mid_new, g_rec, w, batch_size: int):
    """Patch resid_mid, then force MLP post = g_rec * (r_mid_new @ Win + bin)."""
    import torch

    u_new = mlp_pre(r_mid_new.float().cpu(), w)
    post = u_new * g_rec.float().cpu()
    return forward_replace(
        model,
        tokens,
        [
            ("blocks.0.hook_resid_mid", POS, r_mid_new),
            ("blocks.0.mlp.hook_post", POS, post),
        ],
        batch_size,
    )


def run_causal_contributions(args, model, w, p, eval_tok, eval_lab, eval_ok, eval_cache, quads):
    import torch

    rows = []
    r_mid = site_vec(eval_cache, "resid_mid")
    m_out = site_vec(eval_cache, "mlp_out")
    r_post = site_vec(eval_cache, "resid_post")
    a_out = site_vec(eval_cache, "attn_out")
    clean_z = eval_cache["logits"]
    # exact clean decomposition on held-out examples
    z_mid = logits_from_resid(r_mid, w)
    z_mlp = m_out.float() @ w["WU"]
    z_sum = z_mid + z_mlp + (w["b_U"] if w["b_U"] is not None else 0)
    recon = float((z_sum - clean_z).abs().max() / (clean_z.abs().max() + 1e-12))
    print(f"  logit split resid_mid+mlp_out max_rel={recon:.3e}")
    mid_m = true_margin(z_mid, eval_lab)
    mlp_m = true_margin(z_mlp + (w["b_U"] if w["b_U"] is not None else 0), eval_lab)
    full_m = true_margin(clean_z, eval_lab)
    rows.append(
        {
            "intervention": "clean_decomposition",
            "n": int(eval_lab.shape[0]),
            "logit_split_max_rel": recon,
            "mean_margin_from_resid_mid_alone": float(mid_m.mean()),
            "mean_margin_from_mlp_out_alone": float(mlp_m.mean()),
            "mean_margin_full": float(full_m.mean()),
            "mean_p_true_from_resid_mid_alone": float(class_prob(softmax(z_mid), eval_lab).mean()),
            "mean_p_true_from_mlp_out_alone": float(class_prob(softmax(z_mlp), eval_lab).mean()),
            "mean_p_true_full": float(class_prob(softmax(clean_z), eval_lab).mean()),
        }
    )

    # use quadruple a-swaps as matched donors: (a,b) <- (a',b), target a'+b
    rec = torch.tensor([q["ab"] for q in quads], dtype=torch.long)
    don = torch.tensor([q["apb"] for q in quads], dtype=torch.long)
    tokens = eval_tok[rec]
    true = eval_lab[rec]
    target = eval_lab[don]
    ok = eval_ok[rec].detach().cpu()
    with torch.no_grad():
        logits_b = model(tokens)[:, -1, :]

    specs = [
        ("noop", [("blocks.0.hook_resid_mid", POS, r_mid[rec])], r_mid[rec], r_mid[rec], true),
        ("zero_mlp_out", [("blocks.0.hook_mlp_out", POS, torch.zeros_like(m_out[rec]))], m_out[rec], torch.zeros_like(m_out[rec]), true),
        ("zero_attn_out", [("blocks.0.hook_attn_out", POS, torch.zeros_like(a_out[rec]))], a_out[rec], torch.zeros_like(a_out[rec]), true),
        ("replace_attn_out_recompute_mlp", [("blocks.0.hook_attn_out", POS, a_out[don])], a_out[rec], a_out[don], target),
        (
            "replace_attn_out_freeze_mlp_out",
            [("blocks.0.hook_attn_out", POS, a_out[don]), ("blocks.0.hook_mlp_out", POS, m_out[rec])],
            a_out[rec],
            a_out[don],
            target,
        ),
        ("replace_resid_mid", [("blocks.0.hook_resid_mid", POS, r_mid[don])], r_mid[rec], r_mid[don], target),
        ("replace_mlp_out", [("blocks.0.hook_mlp_out", POS, m_out[don])], m_out[rec], m_out[don], target),
        ("replace_resid_post", [("blocks.0.hook_resid_post", POS, r_post[don])], r_post[rec], r_post[don], target),
    ]
    for name, repls, hb, ha, tgt in specs:
        logits_a = forward_replace(model, tokens, repls, args.batch_size)
        row = intervention_metrics(logits_b, logits_a, true, tgt, hb, ha, ok)
        row["intervention"] = name
        row["relation"] = "a_swap_same_b"
        rows.append(row)
        print(
            f"  {name}: target_acc {row['acc_target_before']:.3f}->{row['acc_target_after']:.3f} "
            f"ΔP={row['mean_delta_p_target']:+.3f}"
        )
    return rows


def run_factorial(args, model, p, eval_tok, eval_lab, eval_ok, eval_cache, quads):
    import torch

    rows = []
    r_pre = eval_cache["blocks.0.hook_resid_pre"]
    z = eval_cache["blocks.0.attn.hook_z"]
    sites = [
        ("resid_pre_pos0", "blocks.0.hook_resid_pre", 0, lambda c, i: r_pre[i, 0, :]),
        ("resid_pre_pos1", "blocks.0.hook_resid_pre", 1, lambda c, i: r_pre[i, 1, :]),
        ("z_all", "blocks.0.attn.hook_z", POS, lambda c, i: z[i, POS, :, :]),
        ("attn_out", "blocks.0.hook_attn_out", POS, lambda c, i: site_vec(eval_cache, "attn_out")[i]),
        ("resid_mid", "blocks.0.hook_resid_mid", POS, lambda c, i: site_vec(eval_cache, "resid_mid")[i]),
        ("mlp_post", "blocks.0.mlp.hook_post", POS, lambda c, i: site_vec(eval_cache, "mlp_post")[i]),
        ("resid_post", "blocks.0.hook_resid_post", POS, lambda c, i: site_vec(eval_cache, "resid_post")[i]),
    ]
    rec = torch.tensor([q["ab"] for q in quads], dtype=torch.long)
    tokens = eval_tok[rec]
    true = eval_lab[rec]
    ok = eval_ok[rec]
    with torch.no_grad():
        logits_b = model(tokens)[:, -1, :]

    donors = {
        "neither": rec,
        "a": torch.tensor([q["apb"] for q in quads], dtype=torch.long),
        "b": torch.tensor([q["abp"] for q in quads], dtype=torch.long),
        "both": torch.tensor([q["apbp"] for q in quads], dtype=torch.long),
    }
    targets = {
        "neither": true,
        "a": (eval_tok[donors["a"], 0] + eval_tok[rec, 1]) % p,
        "b": (eval_tok[rec, 0] + eval_tok[donors["b"], 1]) % p,
        "both": (eval_tok[donors["both"], 0] + eval_tok[donors["both"], 1]) % p,
    }

    # operand-position pair: pos0 from a-donor and/or pos1 from b-donor
    for which, tgt in targets.items():
        if which == "neither":
            repls = []
            hb = r_pre[rec, 0, :]
            ha = hb
        elif which == "a":
            repls = [("blocks.0.hook_resid_pre", 0, r_pre[donors["a"], 0, :])]
            hb, ha = r_pre[rec, 0, :], r_pre[donors["a"], 0, :]
        elif which == "b":
            repls = [("blocks.0.hook_resid_pre", 1, r_pre[donors["b"], 1, :])]
            hb, ha = r_pre[rec, 1, :], r_pre[donors["b"], 1, :]
        else:
            repls = [
                ("blocks.0.hook_resid_pre", 0, r_pre[donors["a"], 0, :]),
                ("blocks.0.hook_resid_pre", 1, r_pre[donors["b"], 1, :]),
            ]
            hb, ha = r_pre[rec, 0, :], r_pre[donors["a"], 0, :]
        logits_a = forward_replace(model, tokens, repls, args.batch_size) if repls else logits_b
        row = intervention_metrics(logits_b, logits_a, true, tgt, hb, ha, ok)
        row.update({"site": "resid_pre_operands", "which": which, "target_sum": "matched_counterfactual"})
        rows.append(row)
        print(f"  resid_pre_operands {which}: acc {row['acc_target_before']:.3f}->{row['acc_target_after']:.3f}")

    for site, hook, pos, grab in sites:
        for which in ("neither", "a", "b", "both"):
            don = donors[which]
            hb = grab(eval_cache, rec)
            ha = grab(eval_cache, don)
            if which == "neither":
                logits_a = logits_b
            else:
                logits_a = forward_replace(model, tokens, [(hook, pos, ha)], args.batch_size)
            row = intervention_metrics(logits_b, logits_a, true, targets[which], hb, ha, ok)
            row.update({"site": site, "which": which, "target_sum": "matched_counterfactual"})
            rows.append(row)
        sub = [r for r in rows if r["site"] == site and r["which"] != "neither"]
        if sub:
            print(
                f"  {site} a/b/both acc_after="
                f"{[round(r['acc_target_after'], 3) for r in sub]}"
            )
    return rows


def four_corner(vecs):
    """vecs: dict ab, apb, abp, apbp each [n, d]. I = ab - abp - apb + apbp."""
    return vecs["ab"] - vecs["abp"] - vecs["apb"] + vecs["apbp"]


def run_interactions(args, w, eval_cache, quads):
    import torch

    idx = {name: torch.tensor([q[name] for q in quads], dtype=torch.long) for name in ("ab", "apb", "abp", "apbp")}
    rows = []
    sites = list(INFO_SITES) + ["logits"]
    for site in sites:
        if site == "logits":
            get = lambda i: eval_cache["logits"][i]
        else:
            get = lambda i, s=site: site_vec(eval_cache, s)[i]
        vecs = {name: get(idx[name]).float() for name in idx}
        I = four_corner(vecs)
        scale = torch.stack([vecs[k].norm(dim=-1) for k in vecs], dim=0).mean(dim=0).clamp_min(1e-12)
        add = ((vecs["ab"] - vecs["abp"]).pow(2).sum(-1) + (vecs["ab"] - vecs["apb"]).pow(2).sum(-1)).clamp_min(1e-12)
        center = sum(vecs[k] for k in vecs) / 4
        var4 = sum((vecs[k] - center).pow(2).sum(-1) for k in vecs).clamp_min(1e-12)
        inorm = I.norm(dim=-1)
        rel = inorm / scale
        frac = I.pow(2).sum(-1) / var4
        vs_add = I.pow(2).sum(-1) / add
        proj = float("nan")
        if site in {"resid_mid", "mlp_out", "resid_post"} and I.shape[1] == w["d_model"]:
            zI = I @ w["WU"]
            proj = float(zI.norm(dim=-1).mean())
        elif site == "logits":
            proj = float(inorm.mean())
        mean, lo, hi = bootstrap_mean(rel, args.n_boot, args.seed + 19)
        row = {
            "site": site,
            "n_quads": len(quads),
            "dim": int(I.shape[1]),
            "mean_abs_interaction_norm": float(inorm.mean()),
            "mean_rel_interaction": float(rel.mean()),
            "rel_ci95_lo": lo,
            "rel_ci95_hi": hi,
            "mean_frac_fourpoint_var": float(frac.mean()),
            "mean_interaction_over_additive": float(vs_add.mean()),
            "mean_output_proj_norm": proj,
        }
        rows.append(row)
        print(f"  {site}: ||I||={row['mean_abs_interaction_norm']:.4f} rel={row['mean_rel_interaction']:.4f}")

    # frozen-gate MLP interaction from the four resid_mids, gates of (a,b)
    r = {name: site_vec(eval_cache, "resid_mid")[idx[name]].float() for name in idx}
    g_ab = (mlp_pre(r["ab"], w) > 0).float()
    m_fr = {name: mlp_from_pre(mlp_pre(r[name], w), w, g_ab) for name in idx}
    I_fr = four_corner(m_fr)
    I_act = four_corner({name: site_vec(eval_cache, "mlp_out")[idx[name]].float() for name in idx})
    scale_m = site_vec(eval_cache, "mlp_out")[idx["ab"]].float().norm(dim=-1).clamp_min(1e-12)
    rows.append(
        {
            "site": "mlp_out_frozen_gate_from_resid_mid",
            "n_quads": len(quads),
            "dim": int(I_fr.shape[1]),
            "mean_abs_interaction_norm": float(I_fr.norm(dim=-1).mean()),
            "mean_rel_interaction": float((I_fr.norm(dim=-1) / scale_m).mean()),
            "mean_actual_mlp_out_interaction_norm": float(I_act.norm(dim=-1).mean()),
            "mean_gate_extra_interaction_norm": float((I_act - I_fr).norm(dim=-1).mean()),
        }
    )
    print(
        f"  frozen-gate I vs actual mlp_out I: "
        f"{rows[-1]['mean_abs_interaction_norm']:.4f} vs {rows[-1]['mean_actual_mlp_out_interaction_norm']:.4f}"
    )
    return rows


def run_relu_geometry(args, w, train_cache, eval_cache, eval_tok, eval_lab, eval_ok, quads):
    import torch

    u_tr = site_vec(train_cache, "mlp_pre")
    u_te = site_vec(eval_cache, "mlp_pre")
    g_tr = (u_tr > 0).float()
    g_te = (u_te > 0).float()
    rows_g = [
        {
            "split": "train",
            "mean_frac_active": float(g_tr.mean()),
            "n": int(g_tr.shape[0]),
            "n_units": int(g_tr.shape[1]),
        },
        {
            "split": "test",
            "mean_frac_active": float(g_te.mean()),
            "n": int(g_te.shape[0]),
            "n_units": int(g_te.shape[1]),
        },
    ]
    print(f"  mean active ReLU train={rows_g[0]['mean_frac_active']:.3f} test={rows_g[1]['mean_frac_active']:.3f}")

    idx = {name: torch.tensor([q[name] for q in quads], dtype=torch.long) for name in ("ab", "apb", "abp", "apbp")}
    gab = g_te[idx["ab"]]
    ham = {
        "a": (gab - g_te[idx["apb"]]).abs().sum(dim=-1),
        "b": (gab - g_te[idx["abp"]]).abs().sum(dim=-1),
        "both": (gab - g_te[idx["apbp"]]).abs().sum(dim=-1),
    }
    # do gates distinguish sums? Hamming vs same-sum vs different-sum random pairs
    lab = eval_lab.detach().cpu().long()
    g = torch.Generator().manual_seed(args.seed + 21)
    n = min(400, g_te.shape[0])
    i = torch.randint(0, g_te.shape[0], (n,), generator=g)
    j = torch.randint(0, g_te.shape[0], (n,), generator=g)
    same = lab[i] == lab[j]
    ham_rand = (g_te[i] - g_te[j]).abs().sum(dim=-1)
    rows_g.append(
        {
            "split": "quad_gate_hamming",
            "mean_hamming_a_only": float(ham["a"].mean()),
            "mean_hamming_b_only": float(ham["b"].mean()),
            "mean_hamming_both": float(ham["both"].mean()),
            "mean_hamming_random_same_sum": float(ham_rand[same].mean()) if int(same.sum()) else float("nan"),
            "mean_hamming_random_diff_sum": float(ham_rand[~same].mean()) if int((~same).sum()) else float("nan"),
            "n_quads": len(quads),
        }
    )
    print(
        f"  gate Hamming a={rows_g[-1]['mean_hamming_a_only']:.1f} "
        f"b={rows_g[-1]['mean_hamming_b_only']:.1f} both={rows_g[-1]['mean_hamming_both']:.1f}"
    )

    # verify frozen affine identity on clean examples
    r_mid = site_vec(eval_cache, "resid_mid")
    m_out = site_vec(eval_cache, "mlp_out")
    u = mlp_pre(r_mid, w)
    m_hat = mlp_from_pre(u, w)
    aff_rel = float((m_hat - m_out).abs().max() / (m_out.abs().max() + 1e-12))
    print(f"  frozen-affine reconstructs clean mlp_out max_rel={aff_rel:.3e}")
    if aff_rel > IDENTITY_TOL:
        raise RuntimeError(f"Frozen-gate affine identity failed ({aff_rel:.3e}).")

    # local vs operand-scale response
    rec = idx["ab"]
    don = idx["apb"]
    dr = r_mid[don] - r_mid[rec]
    g_rec = (u[rec] > 0).float()
    dm_frozen = mlp_from_pre(mlp_pre(r_mid[rec] + dr, w), w, g_rec) - mlp_from_pre(u[rec], w, g_rec)
    dm_actual = m_out[don] - m_out[rec]
    gate_change = (g_rec - (mlp_pre(r_mid[don], w) > 0).float()).abs().sum(dim=-1)
    err = (dm_actual - dm_frozen).norm(dim=-1)
    rel_err = err / (dm_actual.norm(dim=-1) + 1e-12)
    # small local noise
    noise = torch.randn_like(dr)
    noise = 0.05 * noise * (r_mid[rec].norm(dim=-1, keepdim=True) / (noise.norm(dim=-1, keepdim=True) + 1e-12))
    dm_f_loc = mlp_from_pre(mlp_pre(r_mid[rec] + noise, w), w, g_rec) - mlp_from_pre(u[rec], w, g_rec)
    dm_a_loc = mlp_from_pre(mlp_pre(r_mid[rec] + noise, w), w, None) - m_out[rec]
    err_loc = (dm_a_loc - dm_f_loc).norm(dim=-1)
    loc_rows = []
    for tag, e, re, gc, dn in (
        ("operand_a_swap", err, rel_err, gate_change, dm_actual.norm(dim=-1)),
        ("local_5pct_noise", err_loc, err_loc / (dm_a_loc.norm(dim=-1) + 1e-12), (g_rec - (mlp_pre(r_mid[rec] + noise, w) > 0).float()).abs().sum(-1), dm_a_loc.norm(dim=-1)),
    ):
        loc_rows.append(
            {
                "perturbation": tag,
                "n": int(e.shape[0]),
                "mean_response_error": float(e.mean()),
                "mean_rel_response_error": float(re.mean()),
                "mean_gate_hamming": float(gc.mean()),
                "mean_actual_dm_norm": float(dn.mean()),
                "corr_hamming_error": float(torch.corrcoef(torch.stack([gc, e]))[0, 1]) if e.numel() > 2 else float("nan"),
            }
        )
        print(
            f"  {tag}: rel_err={loc_rows[-1]['mean_rel_response_error']:.3f} "
            f"hamming={loc_rows[-1]['mean_gate_hamming']:.1f} corr={loc_rows[-1]['corr_hamming_error']:.3f}"
        )
    loc_rows.append({"perturbation": "clean_affine_identity", "max_rel_error": aff_rel, "n": int(r_mid.shape[0])})

    # gate vs errors / margins
    margins = true_margin(eval_cache["logits"], eval_lab)
    wrong = ~eval_ok.detach().cpu().bool()
    rows_g.append(
        {
            "split": "gate_vs_error",
            "mean_frac_active_correct": float(g_te[~wrong].mean()) if int((~wrong).sum()) else float("nan"),
            "mean_frac_active_incorrect": float(g_te[wrong].mean()) if int(wrong.sum()) else float("nan"),
            "corr_frac_active_margin": float(
                torch.corrcoef(torch.stack([g_te.mean(dim=-1), margins.cpu()]))[0, 1]
            ),
        }
    )
    return rows_g, loc_rows, gate_change.cpu(), err.cpu()


def run_nonlinear_causal(args, model, w, eval_tok, eval_lab, eval_ok, eval_cache, quads):
    import torch

    rec = torch.tensor([q["ab"] for q in quads], dtype=torch.long)
    don = torch.tensor([q["apb"] for q in quads], dtype=torch.long)
    tokens = eval_tok[rec]
    true = eval_lab[rec]
    target = eval_lab[don]
    ok = eval_ok[rec]
    r_mid = site_vec(eval_cache, "resid_mid")
    m_out = site_vec(eval_cache, "mlp_out")
    u = site_vec(eval_cache, "mlp_pre")
    g_rec = (u[rec] > 0).float()
    g_don = (u[don] > 0).float()
    ham = (g_rec - g_don).abs().sum(dim=-1)
    with torch.no_grad():
        logits_b = model(tokens)[:, -1, :]

    z_mid = forward_replace(model, tokens, [("blocks.0.hook_resid_mid", POS, r_mid[don])], args.batch_size)
    z_fr = logits_frozen_gate(model, tokens, r_mid[don], g_rec, w, args.batch_size)
    z_mlp = forward_replace(model, tokens, [("blocks.0.hook_mlp_out", POS, m_out[don])], args.batch_size)
    noise = torch.randn_like(r_mid[don])
    scale = (r_mid[don] - r_mid[rec]).norm(dim=-1, keepdim=True)
    noise = noise * (scale / (noise.norm(dim=-1, keepdim=True) + 1e-12))
    z_rand = forward_replace(model, tokens, [("blocks.0.hook_resid_mid", POS, r_mid[rec] + noise)], args.batch_size)

    rows = []
    for name, za, hb, ha in (
        ("resid_mid_donor_recompute", z_mid, r_mid[rec], r_mid[don]),
        ("resid_mid_donor_frozen_recipient_gates", z_fr, r_mid[rec], r_mid[don]),
        ("mlp_out_donor", z_mlp, m_out[rec], m_out[don]),
        ("resid_mid_norm_matched_random", z_rand, r_mid[rec], r_mid[rec] + noise),
    ):
        row = intervention_metrics(logits_b, za, true, target, hb, ha, ok)
        row["intervention"] = name
        row["mean_gate_hamming"] = float(ham.mean())
        rows.append(row)
        print(
            f"  {name}: target_acc {row['acc_target_before']:.3f}->{row['acc_target_after']:.3f} "
            f"ΔP={row['mean_delta_p_target']:+.3f}"
        )

    dz_act = z_mid.detach().cpu() - logits_b.detach().cpu()
    dz_fr = z_fr.detach().cpu() - logits_b.detach().cpu()
    extra = (dz_act - dz_fr).norm(dim=-1)
    rows.append(
        {
            "intervention": "actual_minus_frozen_logit_response",
            "n": int(rec.numel()),
            "mean_extra_logit_norm": float(extra.mean()),
            "mean_gate_hamming": float(ham.mean()),
            "corr_hamming_extra_logit": float(torch.corrcoef(torch.stack([ham.cpu(), extra]))[0, 1]),
            "acc_target_actual": rows[0]["acc_target_after"],
            "acc_target_frozen": rows[1]["acc_target_after"],
            "delta_acc_actual_minus_frozen": rows[0]["acc_target_after"] - rows[1]["acc_target_after"],
            "delta_p_actual_minus_frozen": rows[0]["mean_delta_p_target"] - rows[1]["mean_delta_p_target"],
        }
    )
    print(
        f"  extra nonlinear Δacc={rows[-1]['delta_acc_actual_minus_frozen']:+.3f} "
        f"ΔP={rows[-1]['delta_p_actual_minus_frozen']:+.3f} "
        f"corr(hamming, extra z)={rows[-1]['corr_hamming_extra_logit']:.3f}"
    )
    return rows


def pick(rows, **kwargs):
    out = rows
    for k, v in kwargs.items():
        out = [r for r in out if r.get(k) == v]
    return out


def decide(info, causal, factorial, interact, gates, local, nonlinear) -> dict:
    def acc(site, target="sum"):
        rows = pick(info, site=site, target=target)
        return rows[0]["linear_acc_test"] if rows else float("nan")

    dec_pre, dec_mid, dec_post = acc("resid_pre"), acc("resid_mid"), acc("resid_post")
    dec_mlp = acc("mlp_out")
    dec_attn = acc("attn_out")

    def ival(site, key="mean_rel_interaction"):
        rows = pick(interact, site=site)
        return rows[0][key] if rows else float("nan")

    i_mid, i_post, i_logit = ival("resid_mid"), ival("resid_post"), ival("logits")
    i_fr = pick(interact, site="mlp_out_frozen_gate_from_resid_mid")
    extra_i = i_fr[0]["mean_gate_extra_interaction_norm"] if i_fr else float("nan")
    act_i = i_fr[0]["mean_actual_mlp_out_interaction_norm"] if i_fr else float("nan")

    c_attn = pick(causal, intervention="replace_attn_out_recompute_mlp")
    c_attn_fr = pick(causal, intervention="replace_attn_out_freeze_mlp_out")
    c_mid = pick(causal, intervention="replace_resid_mid")
    c_mlp = pick(causal, intervention="replace_mlp_out")
    c_zero_mlp = pick(causal, intervention="zero_mlp_out")
    attn_acc = c_attn[0]["acc_target_after"] if c_attn else float("nan")
    attn_fr_acc = c_attn_fr[0]["acc_target_after"] if c_attn_fr else float("nan")
    mid_acc = c_mid[0]["acc_target_after"] if c_mid else float("nan")
    mlp_acc = c_mlp[0]["acc_target_after"] if c_mlp else float("nan")
    zero_mlp_true = c_zero_mlp[0]["acc_true_after"] if c_zero_mlp else float("nan")

    fac_mid = pick(factorial, site="resid_mid", which="a")
    fac_op = pick(factorial, site="resid_pre_operands", which="a")
    fac_mid_acc = fac_mid[0]["acc_target_after"] if fac_mid else float("nan")
    fac_op_acc = fac_op[0]["acc_target_after"] if fac_op else float("nan")

    nl = pick(nonlinear, intervention="actual_minus_frozen_logit_response")
    dacc = nl[0]["delta_acc_actual_minus_frozen"] if nl else float("nan")
    dp = nl[0]["delta_p_actual_minus_frozen"] if nl else float("nan")
    act_nl = pick(nonlinear, intervention="resid_mid_donor_recompute")
    fr_nl = pick(nonlinear, intervention="resid_mid_donor_frozen_recipient_gates")
    rand_nl = pick(nonlinear, intervention="resid_mid_norm_matched_random")
    act_nl_acc = act_nl[0]["acc_target_after"] if act_nl else float("nan")
    fr_nl_acc = fr_nl[0]["acc_target_after"] if fr_nl else float("nan")
    rand_acc = rand_nl[0]["acc_target_after"] if rand_nl else float("nan")

    loc = pick(local, perturbation="operand_a_swap")
    loc_rel = loc[0]["mean_rel_response_error"] if loc else float("nan")
    loc_corr = loc[0]["corr_hamming_error"] if loc else float("nan")

    earliest = "none"
    for site in INFO_SITES:
        a = acc(site)
        sh = pick(info, site=site, target="sum")[0]["linear_acc_shuffled"]
        if a == a and a >= 0.35 and a >= sh + 0.15:
            earliest = site
            break

    decode_jumps_at_mlp = (dec_mlp == dec_mlp and dec_mid == dec_mid and dec_mlp - dec_mid >= 0.15)
    mid_already_strong = dec_mid == dec_mid and dec_mid >= 0.70
    gating_helps = dacc == dacc and dacc >= 0.15 and act_nl_acc >= rand_acc + 0.20
    gating_irrelevant = dacc == dacc and abs(dacc) < 0.08 and abs(dp) < 0.08
    attn_writes = attn_acc == attn_acc and attn_acc >= 0.70
    mlp_writes = mlp_acc == mlp_acc and mlp_acc >= 0.70
    mid_writes = mid_acc == mid_acc and mid_acc >= 0.70
    extra_i_big = extra_i == extra_i and act_i == act_i and extra_i >= 0.35 * (act_i + 1e-12)

    letter, title = "D", "Causal mechanism unresolved"
    if mid_already_strong and mid_writes and gating_irrelevant and (not decode_jumps_at_mlp):
        letter, title = "B", "Attention-side computation localized"
    elif decode_jumps_at_mlp and gating_helps and (not mid_already_strong):
        letter, title = "A", "MLP nonlinear computation localized"
    elif (attn_writes or mid_writes) and mlp_writes and (gating_helps or extra_i_big) and (dec_mid >= 0.35 or attn_writes):
        letter, title = "C", "Distributed arithmetic computation"
    elif mid_writes and mlp_writes and not gating_helps and not mid_already_strong:
        letter, title = "C", "Distributed arithmetic computation"

    # obstruction needs a task-structured object, not just "MLP is nonlinear" or "z copies the answer"
    ready = "NO"
    missing = (
        "No compact task-structured representation with held-out group composition "
        "(07g–07i G_Δ / G_a G_b failed). 07j may localize a computation but does not "
        "supply a compatibility relation beyond the Transformer's exact forward map."
    )
    if letter in {"A", "B", "C"} and False:
        ready = "YES"
        missing = ""

    h_rows = [
        {
            "hypothesis": "H1_attention_arithmetic",
            "supported": letter == "B",
            "evidence": (
                f"sum decode resid_pre={dec_pre:.3f} attn={dec_attn:.3f} mid={dec_mid:.3f}; "
                f"attn replace acc={attn_acc:.3f} freeze-mlp acc={attn_fr_acc:.3f}; "
                f"I_rel mid={i_mid:.3f}"
            ),
        },
        {
            "hypothesis": "H2_mlp_nonlinear_arithmetic",
            "supported": letter == "A",
            "evidence": (
                f"decode mid={dec_mid:.3f} mlp_out={dec_mlp:.3f} post={dec_post:.3f}; "
                f"frozen-vs-actual Δacc={dacc:.3f} ΔP={dp:.3f}; gate extra I={extra_i}; "
                f"local rel_err={loc_rel:.3f} corr(hamming,err)={loc_corr:.3f}"
            ),
        },
        {
            "hypothesis": "H3_distributed",
            "supported": letter == "C",
            "evidence": (
                f"attn/mid/mlp write acc {attn_acc:.3f}/{mid_acc:.3f}/{mlp_acc:.3f}; "
                f"zero_mlp true acc={zero_mlp_true:.3f}; operand resid_pre a-swap acc={fac_op_acc:.3f}; "
                f"resid_mid a-swap acc={fac_mid_acc:.3f}"
            ),
        },
    ]
    return {
        "letter": letter,
        "title": title,
        "earliest_decode_site": earliest,
        "decode": {"resid_pre": dec_pre, "attn_out": dec_attn, "resid_mid": dec_mid, "mlp_out": dec_mlp, "resid_post": dec_post},
        "obstruction_ready": ready,
        "missing_object": missing,
        "h_rows": h_rows,
        "gating_delta_acc": dacc,
        "gating_delta_p": dp,
        "act_nl_acc": act_nl_acc,
        "fr_nl_acc": fr_nl_acc,
        "i_mid": i_mid,
        "i_post": i_post,
        "i_logit": i_logit,
        "zero_mlp_true": zero_mlp_true,
        "attn_acc": attn_acc,
        "mid_acc": mid_acc,
        "mlp_acc": mlp_acc,
        "fac_op_acc": fac_op_acc,
    }


def mechanism_rows(decision, info, causal, factorial, interact, nonlinear) -> list[dict]:
    rows = [
        {
            "step": "operand_representation",
            "information": "a at pos 0, b at pos 1 (07i operand-copy; 07j factorial resid_pre)",
            "causal_validation": "resid_pre pos0/pos1 matched a/b swaps",
            "map_type": "embedding / residual copy, linear in token id via W_E",
            "demonstrated": True,
            "task_arithmetic": False,
            "note": "Not the sum. Official 07i.",
        },
        {
            "step": "attention_transport",
            "information": "equals attends to a and b, not to itself (07i patterns)",
            "causal_validation": "z_all / attn_out replacement; single heads fail (07i, 07j)",
            "map_type": "input-dependent linear combination of values (softmax attention)",
            "demonstrated": True,
            "task_arithmetic": decision["letter"] == "B",
            "note": "Joint residual at '=' after attention.",
        },
        {
            "step": "resid_mid",
            "information": f"linear sum decode={decision['decode']['resid_mid']:.3f}",
            "causal_validation": "full resid_mid donor patch (07h/07j)",
            "map_type": "affine skip: r_pre + attn_out",
            "demonstrated": True,
            "task_arithmetic": decision["letter"] in {"B", "C"},
        },
        {
            "step": "mlp_pre_and_relu_gates",
            "information": "512-D preactivation and binary gate pattern",
            "causal_validation": "frozen-gate vs recomputed-gate donor patch",
            "map_type": "affine then input-dependent ReLU (piecewise affine)",
            "demonstrated": True,
            "task_arithmetic": decision["letter"] in {"A", "C"},
        },
        {
            "step": "mlp_out_and_resid_post",
            "information": f"linear sum decode={decision['decode']['mlp_out']:.3f}/{decision['decode']['resid_post']:.3f}",
            "causal_validation": "mlp_out / resid_post donor patch; T = W_out W_U (07g)",
            "map_type": "affine given gates; then linear unembed",
            "demonstrated": True,
            "task_arithmetic": False,
            "note": "Output channel. 07g B: not a modular operator.",
        },
    ]
    return rows


def write_report(path, decision, ident, coverage):
    md = f"""# 07j scientific report

**Official 07d label: NO_GO (unchanged).**
**07g–07i decision: B (unchanged).**

**07j decision: {decision['letter']} — {decision['title']}**

**OBSTRUCTION_READY = {decision['obstruction_ready']}**

{decision['missing_object']}

This report separates observation, inference, and hypothesis.

## Observation

- Forward identities at position 2 hold (see `07j_forward_identity.csv`). Worst relative reconstruction error is below {IDENTITY_TOL}.
- Earliest site where the modular sum is linearly decodable above chance: **{decision['earliest_decode_site']}**.
- Linear sum decode: resid_pre={decision['decode']['resid_pre']:.3f}, attn_out={decision['decode']['attn_out']:.3f}, resid_mid={decision['decode']['resid_mid']:.3f}, mlp_out={decision['decode']['mlp_out']:.3f}, resid_post={decision['decode']['resid_post']:.3f}.
- Attention a-swap write acc={decision['attn_acc']:.3f}; resid_mid write acc={decision['mid_acc']:.3f}; mlp_out write acc={decision['mlp_acc']:.3f}.
- Zeroing MLP output leaves true-answer acc={decision['zero_mlp_true']:.3f}.
- Four-corner relative interaction: resid_mid={decision['i_mid']:.3f}, resid_post={decision['i_post']:.3f}, logits={decision['i_logit']:.3f}.
- Frozen-gate vs actual resid_mid donor patch: acc {decision['fr_nl_acc']:.3f} vs {decision['act_nl_acc']:.3f} (Δacc={decision['gating_delta_acc']:+.3f}, ΔP={decision['gating_delta_p']:+.3f}).
- Matched operand resid_pre a-swap counterfactual acc={decision['fac_op_acc']:.3f}.
- Coverage: {coverage}.

## Inference

- Decoding the sum is information availability, not a proof that the site computes addition.
- Zero ablation of the MLP can destroy the answer without proving the MLP is the adder.
- A nonzero four-corner interaction after the MLP can be inherited from resid_mid through a fixed affine map.
- 07g–07i already showed T-top / G_Δ is not a modular group. That conclusion is not reopened.

## Hypothesis (not claimed as fact)

The learned computation is operand embedding → attention copy into '=' → piecewise-affine MLP → linear unembed. Whether the arithmetic lives in attention, in ReLU gates, or in both is the 07j letter above.

## H1 / H2 / H3

See `07j_mechanism_summary.csv`. Supported hypothesis matches the 07j letter.

## What this is not

Not a new representation claim. Not a modular algorithm claim from decode accuracy. Not an obstruction success. Not a change to 07d/07f/07g/07h/07i.
"""
    write_text(path, md)


def make_figures(figures_dir: Path, interact, local_scatter, causal) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = [r for r in interact if r["site"] in set(INFO_SITES) | {"logits"}]
    if sites:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(sites)), [float(r["mean_rel_interaction"]) for r in sites])
        ax.set_xticks(range(len(sites)), [r["site"] for r in sites], rotation=30, ha="right")
        ax.set_ylabel("||I|| / mean ||r||")
        ax.set_title("07j four-corner interaction (scale-normalized)")
        fig.tight_layout()
        fig.savefig(figures_dir / "07j_interaction_by_site.png", dpi=150)
        plt.close(fig)

    ham, err = local_scatter
    if ham is not None and err is not None and ham.numel():
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(ham.numpy(), err.numpy(), s=12, alpha=0.6)
        ax.set_xlabel("ReLU gate Hamming (a-swap)")
        ax.set_ylabel("||Δm_actual − Δm_frozen||")
        ax.set_title("07j gate change vs frozen-gate error")
        fig.tight_layout()
        fig.savefig(figures_dir / "07j_gate_transition_vs_response_error.png", dpi=150)
        plt.close(fig)

    keep = [
        "noop",
        "zero_mlp_out",
        "zero_attn_out",
        "replace_attn_out_recompute_mlp",
        "replace_attn_out_freeze_mlp_out",
        "replace_resid_mid",
        "replace_mlp_out",
    ]
    sub = [r for r in causal if r.get("intervention") in keep and "acc_target_after" in r]
    if sub:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(sub)), [float(r["acc_target_after"]) for r in sub])
        ax.set_xticks(range(len(sub)), [r["intervention"] for r in sub], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("acc toward intervention target")
        ax.set_title("07j attention vs MLP causal interventions")
        fig.tight_layout()
        fig.savefig(figures_dir / "07j_attention_vs_mlp_causal_effect.png", dpi=150)
        plt.close(fig)


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
        raise RuntimeError(f"Refusing step {step}; 07j is frozen at 9080.")
    if cfg.n_layers != 1:
        raise RuntimeError("07j assumes the audited one-layer model.")
    model.eval()
    device = next(model.parameters()).device
    print(f"07j checkpoint step {step} device={device}")
    print("07d remains NO_GO. 07g–07i remain B. This run does not rebuild O.")

    tokens, labels = make_dataset(cfg)
    (train_tokens, train_labels), (test_tokens, test_labels) = split_dataset(cfg, tokens, labels)
    train_tokens, train_labels = train_tokens.to(device), train_labels.to(device)
    test_tokens, test_labels = test_tokens.to(device), test_labels.to(device)
    with torch.no_grad():
        train_ok = model(train_tokens)[:, -1, :].argmax(-1) == train_labels
        test_ok = model(test_tokens)[:, -1, :].argmax(-1) == test_labels
    print(f"train_acc={float(train_ok.float().mean()):.4f} test_acc={float(test_ok.float().mean()):.4f}")

    val_idx, eval_idx = split_pool_indices(test_tokens.shape[0], args.val_frac, args.val_seed)
    val_tok, eval_tok = test_tokens[val_idx], test_tokens[eval_idx]
    val_lab, eval_lab = test_labels[val_idx], test_labels[eval_idx]
    val_ok, eval_ok = test_ok[val_idx], test_ok[eval_idx]
    p = cfg.p

    print("\n=== Part A — cache and verify forward identities ===")
    w = weight_maps(model)
    print(f"  Win {tuple(w['Win'].shape)} Wout {tuple(w['Wout'].shape)} WU {tuple(w['WU'].shape)}")
    print(f"  raw W_in {w['W_in_raw']} W_out {w['W_out_raw']} W_U {w['W_U_raw']}")
    train_cache = cache_hooks(model, train_tokens, args.cache_batch)
    val_cache = cache_hooks(model, val_tok, args.cache_batch)
    eval_cache = cache_hooks(model, eval_tok, args.cache_batch)
    ident = verify_forward(eval_cache, w, eval_tok)
    write_table(results_dir / "07j_forward_identity.csv", ident)

    cov = [
        coverage_row(train_tokens, train_labels, p, "train"),
        coverage_row(val_tok, val_lab, p, "val"),
        coverage_row(eval_tok, eval_lab, p, "test"),
    ]

    print("\n=== Part B — information by site ===")
    info = run_information(
        args, p, train_cache, val_cache, eval_cache, train_tokens, val_tok, eval_tok, train_labels, val_lab, eval_lab
    )
    write_table(results_dir / "07j_information_by_site.csv", info)

    print("\n=== sample quadruples ===")
    val_quads = sample_quadruples(val_tok, args.n_quad_val, args.seed + 31, p)
    test_quads = sample_quadruples(eval_tok, args.n_quad_test, args.seed + 47, p)
    print(f"  val quads={len(val_quads)} test quads={len(test_quads)}")
    if len(test_quads) < 16:
        raise RuntimeError(f"Only {len(test_quads)} test quadruples; cannot run Parts D–G.")
    cov.append({"split": "test_quads", "n": len(test_quads), "p": p})
    write_table(results_dir / "07j_coverage.csv", cov)

    print("\n=== Part C — attention vs MLP causal contributions ===")
    causal = run_causal_contributions(args, model, w, p, eval_tok, eval_lab, eval_ok, eval_cache, test_quads)
    write_table(results_dir / "07j_attention_mlp_causal_contributions.csv", causal)

    print("\n=== Part D — factorial operand interventions ===")
    factorial = run_factorial(args, model, p, eval_tok, eval_lab, eval_ok, eval_cache, test_quads)
    write_table(results_dir / "07j_factorial_operand_interventions.csv", factorial)

    print("\n=== Part E — four-corner interactions ===")
    interact = run_interactions(args, w, eval_cache, test_quads)
    write_table(results_dir / "07j_four_corner_interactions.csv", interact)

    print("\n=== Part F — ReLU gate geometry ===")
    gates, local, ham, err = run_relu_geometry(args, w, train_cache, eval_cache, eval_tok, eval_lab, eval_ok, test_quads)
    write_table(results_dir / "07j_relu_gate_geometry.csv", gates)
    write_table(results_dir / "07j_local_linear_response.csv", local)

    print("\n=== Part G — nonlinear causal tests ===")
    nonlinear = run_nonlinear_causal(args, model, w, eval_tok, eval_lab, eval_ok, eval_cache, test_quads)
    write_table(results_dir / "07j_nonlinear_causal_tests.csv", nonlinear)

    print("\n=== Parts I–J — mechanism and decision ===")
    decision = decide(info, causal, factorial, interact, gates, local, nonlinear)
    mech = mechanism_rows(decision, info, causal, factorial, interact, nonlinear)
    write_table(results_dir / "07j_mechanism_summary.csv", mech + decision["h_rows"])
    make_figures(figures_dir, interact, (ham, err), causal)
    write_report(results_dir / "07j_scientific_report.md", decision, ident, cov)
    write_text(
        results_dir / "07j_decision.json",
        json.dumps(
            {
                "official_07d_label": "NO_GO",
                "prior_07g_07i": "B",
                "checkpoint_step": 9080,
                "decision": decision["letter"],
                "decision_title": decision["title"],
                "obstruction_ready": decision["obstruction_ready"],
                "missing_object": decision["missing_object"],
                "earliest_decode_site": decision["earliest_decode_site"],
                "decode": decision["decode"],
                "seed": args.seed,
                "val_seed": args.val_seed,
            },
            indent=2,
        )
        + "\n",
    )
    print(
        f"\n07j decision {decision['letter']} — {decision['title']}\n"
        f"earliest decode site: {decision['earliest_decode_site']}\n"
        f"OBSTRUCTION_READY = {decision['obstruction_ready']}\n"
        f"{decision['missing_object']}\n"
    )
    print(f"saved {results_dir}")
    print(f"saved {figures_dir}")


if __name__ == "__main__":
    main()
