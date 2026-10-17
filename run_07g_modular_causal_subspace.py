"""Experiment 0B: recover the causally effective subspace at L0_mlp_post.

Frozen checkpoint step 9080. Official 07d remains NO_GO. Does not retrain, does
not retune 07d, and does not rebuild obstruction.

Part A verifies the exact linear map T = W_out W_U from mlp_post to logits.
Part B compares rank-limited projections of donor deltas.
Part C tests whether a recovered subspace supports a shift operator G_Δ.
Part D records what this hook can and cannot prove.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    logits_with_delta,
    pair_metrics,
    project_onto,
    project_out,
    sample_pairs,
    write_csv,
    write_text,
)

RANKS = (4, 8, 16, 32, 64, 128)
SHIFT_DELTAS = (1, 2, 3, 5, 10)
COMPOSE_PAIRS = ((1, 2, 3), (1, 1, 2), (5, 5, 10), (2, 3, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover the causal mlp_post subspace at step 9080.")
    parser.add_argument("--checkpoint", default="artifacts/toy_modular/checkpoints/step_0009080.pt")
    parser.add_argument("--results-dir", default="artifacts/results/toy_modular")
    parser.add_argument("--figures-dir", default="artifacts/figures/toy_modular")
    parser.add_argument("--n-freqs", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--g-ridges", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=1, help="Split original test into val/eval; not the train split seed.")
    parser.add_argument("--val-frac", type=float, default=0.3)
    parser.add_argument("--n-recipients", type=int, default=96)
    parser.add_argument("--n-donors-each", type=int, default=2)
    parser.add_argument("--n-train-diff-recipients", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sv-tol", type=float, default=1e-6)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def write_table(path, rows: list[dict]) -> None:
    if not rows:
        write_csv(path, rows)
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def output_map_T(model):
    """T such that Δz = Δh @ T for edits at mlp.hook_post, no LayerNorm."""
    W_out = model.blocks[0].mlp.W_out.detach().float().cpu()
    W_U = model.W_U.detach().float().cpu()
    if W_U.shape[0] == W_out.shape[1]:
        return W_out @ W_U
    if W_U.shape[1] == W_out.shape[1]:
        return W_out @ W_U.T
    raise RuntimeError(f"Cannot align W_out {tuple(W_out.shape)} with W_U {tuple(W_U.shape)}")


def svd_map(T, tol: float):
    import torch

    U, s, Vh = torch.linalg.svd(T.float(), full_matrices=False)
    s_list = [float(v) for v in s]
    smax = s_list[0] if s_list else 0.0
    rank = sum(1 for v in s_list if v > tol * max(smax, 1e-12))
    energy = [v * v for v in s_list]
    total = sum(energy) or 1.0
    cum = []
    running = 0.0
    for e in energy:
        running += e
        cum.append(running / total)
    pr = (total * total) / (sum(e * e for e in energy) + 1e-12)
    return {
        "U": U[:, :rank],
        "s": s,
        "Vh": Vh[:rank],
        "singular_values": s_list,
        "numerical_rank": rank,
        "energy_frac_by_k": cum,
        "participation_ratio": float(pr),
        "energy_total": float(total),
    }


def orthonormalize(basis, k: int | None = None):
    import torch

    q, _ = torch.linalg.qr(basis.float())
    if k is None:
        k = q.shape[1]
    return q[:, : min(k, q.shape[1])]


def top_k(basis, k: int):
    return orthonormalize(basis, k)


def subspace_overlap(B1, B2) -> float:
    """Mean of cos² principal angles: ||B1^T B2||_F^2 / min(k1, k2)."""
    import torch

    if B1.numel() == 0 or B2.numel() == 0:
        return float("nan")
    M = B1.float().T @ B2.float()
    s = torch.linalg.svdvals(M)
    return float((s * s).sum() / min(B1.shape[1], B2.shape[1]))


def split_pool_indices(n: int, val_frac: float, seed: int):
    import torch

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = int(val_frac * n)
    return perm[:n_val], perm[n_val:]


def take_pool(tokens, labels, correct, h, idx):
    return tokens[idx], labels[idx], correct[idx], h[idx.cpu()]


def gather_pairs(pool, delta: int, n_recipients: int, n_donors: int, seed: int, p: int):
    tokens, labels, correct, h = pool
    rec, don = sample_pairs(tokens, labels, correct, delta, n_recipients, n_donors, seed, p)
    if rec.numel() == 0:
        return None
    return {
        "delta": delta,
        "rec": rec,
        "don": don,
        "tokens": tokens[rec],
        "labels": labels[rec],
        "donor_sums": labels[don],
        "h_rec": h[rec.cpu()],
        "h_don": h[don.cpu()],
        "recip_correct": correct[rec].detach().cpu(),
    }


def collect_train_diffs(train_pool, p: int, n_recipients: int, n_donors: int, seed: int):
    import torch

    chunks = []
    for delta in (0,) + SHIFT_DELTAS:
        bundle = gather_pairs(train_pool, delta, n_recipients, n_donors, seed + 3 * delta, p)
        if bundle is None:
            continue
        chunks.append(bundle["h_don"].float() - bundle["h_rec"].float())
    if not chunks:
        raise RuntimeError("No training donor differences. Check train accuracy at this checkpoint.")
    return torch.cat(chunks, dim=0)


def _on(mask, tensor):
    return mask.to(device=tensor.device)


def stratified(logits_b, logits_a, labels, donor_sums, h_r, h_d, recip_correct):
    recip_correct = recip_correct.detach().cpu()
    row = pair_metrics(logits_b, logits_a, labels, donor_sums, h_r, h_d, recip_correct)
    for name, mask in (
        ("recip_correct", recip_correct.bool()),
        ("recip_incorrect", ~recip_correct.bool()),
    ):
        if int(mask.sum()) < 8:
            continue
        sub = pair_metrics(
            logits_b[_on(mask, logits_b)],
            logits_a[_on(mask, logits_a)],
            labels[_on(mask, labels)],
            donor_sums[_on(mask, donor_sums)],
            h_r[_on(mask, h_r)],
            h_d[_on(mask, h_d)],
            recip_correct[mask],
        )
        for key in (
            "n_pairs",
            "mean_delta_p_donor",
            "frac_pred_donor_after",
            "mean_delta_margin_donor",
            "mean_kl",
        ):
            row[f"{name}_{key}"] = sub[key]
    return row


def eval_delta_patch(model, bundle, delta_h, batch_size: int):
    import torch

    with torch.no_grad():
        logits_b = model(bundle["tokens"])[:, -1, :]
    logits_a = logits_with_delta(model, bundle["tokens"], bundle["h_rec"], delta_h, batch_size)
    row = stratified(
        logits_b,
        logits_a,
        bundle["labels"],
        bundle["donor_sums"],
        bundle["h_rec"],
        bundle["h_rec"].float() + delta_h.float(),
        bundle["recip_correct"],
    )
    row["mean_applied_delta_norm"] = float(delta_h.float().norm(dim=-1).mean())
    return row, logits_b, logits_a


def fourier_basis(train_h, train_tokens, p: int, freqs, ridge: float, k: int):
    import torch

    y = latent_targets(train_tokens, p, freqs).cpu()
    W = fit_ridge(train_h, y, ridge)
    U, s, _ = torch.linalg.svd(W[:-1].float(), full_matrices=False)
    return U[:, : min(k, int(s.numel()))], W, s


def build_method_bases(train_h, train_tokens, train_diffs, T_info, p, freqs, ridge, seed, k: int):
    import torch

    d = train_h.shape[1]
    U_T = T_info["U"]
    _, _, Vh_diff = torch.linalg.svd(train_diffs.float(), full_matrices=False)
    bases = {
        "T_top": U_T[:, : min(k, U_T.shape[1])],
        "pca_acts": pca_subspace(train_h, min(k, d)),
        "svd_donor_diff": Vh_diff[: min(k, Vh_diff.shape[0])].T,
        "random": random_subspace(d, min(k, d), seed),
    }
    fourier, W_f, _ = fourier_basis(train_h, train_tokens, p, freqs, ridge, k)
    bases["fourier_probe"] = fourier
    return bases, W_f


def part_a(model, T, T_info, bundles, batch_size: int):
    import torch

    rows = []
    U = T_info["U"]
    for name, bundle in bundles.items():
        if bundle is None:
            continue
        delta = bundle["h_don"].float() - bundle["h_rec"].float()
        row_full, logits_b, logits_a = eval_delta_patch(model, bundle, delta, batch_size)
        emp = (logits_a - logits_b).float().cpu()
        pred = delta @ T.float()
        abs_err = (emp - pred).abs()
        rel = float(abs_err.max() / (emp.abs().max() + 1e-12))
        delta_T = project_onto(delta, U)
        delta_ker = project_out(delta, U)
        row_T, _, logits_T = eval_delta_patch(model, bundle, delta_T, batch_size)
        row_ker, _, logits_ker = eval_delta_patch(model, bundle, delta_ker, batch_size)
        ker_logit = (logits_ker - logits_b).float().abs().max().item()
        rows.append(
            {
                "bundle": name,
                "delta": bundle["delta"],
                "n_pairs": row_full["n_pairs"],
                "max_abs_error_z": float(abs_err.max()),
                "mean_abs_error_z": float(abs_err.mean()),
                "max_rel_error_z": rel,
                "full_mean_delta_p_donor": row_full["mean_delta_p_donor"],
                "T_proj_mean_delta_p_donor": row_T["mean_delta_p_donor"],
                "ker_mean_delta_p_donor": row_ker["mean_delta_p_donor"],
                "ker_max_abs_logit_change": ker_logit,
                "T_frac_of_full_dP": (
                    row_T["mean_delta_p_donor"] / row_full["mean_delta_p_donor"]
                    if abs(row_full["mean_delta_p_donor"]) > 1e-6
                    else float("nan")
                ),
                "full_frac_pred_donor_after": row_full["frac_pred_donor_after"],
                "T_frac_pred_donor_after": row_T["frac_pred_donor_after"],
                "ker_frac_pred_donor_after": row_ker["frac_pred_donor_after"],
            }
        )
        _ = logits_T
    return rows


def part_b_eval(model, bases, bundle, full_row, batch_size: int, rank: int, split: str, method: str):
    delta = bundle["h_don"].float() - bundle["h_rec"].float()
    B = bases[method]
    natural = project_onto(delta, B)
    matched_full = match_norm_in_subspace(delta, B)
    rows = []
    for variant, applied in (("proj", natural), ("norm_matched_full", matched_full)):
        metrics, _, _ = eval_delta_patch(model, bundle, applied, batch_size)
        recov = (
            metrics["mean_delta_p_donor"] / full_row["mean_delta_p_donor"]
            if abs(full_row["mean_delta_p_donor"]) > 1e-6
            else float("nan")
        )
        row = {
            "split": split,
            "bundle": f"delta_{bundle['delta']}",
            "delta": bundle["delta"],
            "method": method,
            "rank": rank,
            "variant": variant,
            "basis_dim": int(B.shape[1]),
            "frac_full_dP_recovered": recov,
        }
        row.update(metrics)
        rows.append(row)
    return rows


def fit_G(C_rec, C_don, ridge: float):
    import torch

    k = C_rec.shape[1]
    gram = C_rec.T @ C_rec + ridge * torch.eye(k, dtype=C_rec.dtype)
    return torch.linalg.solve(gram, C_rec.T @ C_don)


def apply_G(h_rec, B, G):
    C = h_rec.float() @ B
    C_new = C @ G
    return h_rec.float() + (C_new - C) @ B.T


def part_c(
    model,
    train_pool,
    val_bundles,
    test_bundles,
    bases_by_method,
    selected,
    p: int,
    n_recipients: int,
    n_donors: int,
    seed: int,
    g_ridges: list[float],
    batch_size: int,
):
    import torch

    train_tokens, train_labels, train_correct, train_h = train_pool
    rows = []
    compose_rows = []
    Gs: dict[tuple[str, int, float], object] = {}

    methods = [m for m in selected if m in bases_by_method]
    # Choose ridge on val using T_top if present, else the first method.
    probe_method = "T_top" if "T_top" in methods else methods[0]
    B_probe = bases_by_method[probe_method]
    best_ridge = g_ridges[0]
    best_score = -1e9
    for ridge in g_ridges:
        scores = []
        for delta in SHIFT_DELTAS:
            tr = gather_pairs(train_pool, delta, n_recipients, n_donors, seed + 40 * delta, p)
            va = val_bundles.get(delta)
            if tr is None or va is None:
                continue
            G = fit_G(tr["h_rec"].float() @ B_probe, tr["h_don"].float() @ B_probe, ridge)
            h_new = apply_G(va["h_rec"], B_probe, G)
            metrics, _, _ = eval_delta_patch(model, va, h_new - va["h_rec"].float(), batch_size)
            scores.append(metrics["mean_delta_p_donor"])
        if scores:
            mean_s = sum(scores) / len(scores)
            if mean_s > best_score:
                best_score, best_ridge = mean_s, ridge

    for method in methods:
        B = bases_by_method[method]
        G_by_delta = {}
        for delta in SHIFT_DELTAS:
            tr = gather_pairs(train_pool, delta, n_recipients, n_donors, seed + 40 * delta, p)
            if tr is None:
                continue
            G = fit_G(tr["h_rec"].float() @ B, tr["h_don"].float() @ B, best_ridge)
            G_by_delta[delta] = G
            Gs[(method, delta, best_ridge)] = G
            for split, bundle in (("val", val_bundles.get(delta)), ("test", test_bundles.get(delta))):
                if bundle is None:
                    continue
                h_pred = apply_G(bundle["h_rec"], B, G)
                metrics, _, _ = eval_delta_patch(
                    model, bundle, h_pred - bundle["h_rec"].float(), batch_size
                )
                C_r = bundle["h_rec"].float() @ B
                C_d = bundle["h_don"].float() @ B
                pred_C = C_r @ G
                resid = float((pred_C - C_d).pow(2).mean().sqrt())
                scale = float(C_d.pow(2).mean().sqrt() + 1e-12)
                row = {
                    "method": method,
                    "rank": int(B.shape[1]),
                    "delta": delta,
                    "split": split,
                    "ridge": best_ridge,
                    "repr_rmse": resid,
                    "repr_rel_rmse": resid / scale,
                    "n_train_pairs": int(tr["rec"].numel()),
                }
                row.update(metrics)
                rows.append(row)

        for a, b, c in COMPOSE_PAIRS:
            if a not in G_by_delta or b not in G_by_delta or c not in G_by_delta:
                continue
            Ga, Gb, Gc = G_by_delta[a], G_by_delta[b], G_by_delta[c]
            composed = Ga @ Gb
            denom = float(Gc.pow(2).mean().sqrt() + 1e-12)
            f_err = float((composed - Gc).pow(2).mean().sqrt() / denom)
            bundle = test_bundles.get(c)
            if bundle is None:
                continue
            h_comp = apply_G(apply_G(bundle["h_rec"], B, Ga), B, Gb)
            h_direct = apply_G(bundle["h_rec"], B, Gc)
            met_comp, _, _ = eval_delta_patch(
                model, bundle, h_comp - bundle["h_rec"].float(), batch_size
            )
            met_dir, _, _ = eval_delta_patch(
                model, bundle, h_direct - bundle["h_rec"].float(), batch_size
            )
            compose_rows.append(
                {
                    "method": method,
                    "rank": int(B.shape[1]),
                    "delta_a": a,
                    "delta_b": b,
                    "delta_c": c,
                    "fro_rel_error": f_err,
                    "compose_mean_delta_p_donor": met_comp["mean_delta_p_donor"],
                    "direct_mean_delta_p_donor": met_dir["mean_delta_p_donor"],
                    "compose_frac_pred_donor_after": met_comp["frac_pred_donor_after"],
                    "direct_frac_pred_donor_after": met_dir["frac_pred_donor_after"],
                }
            )
    _ = train_tokens, train_labels, train_correct, train_h, Gs
    return rows, compose_rows, best_ridge


def part_d_text() -> str:
    return """# 07g — Where the computation can live

Official 07d remains NO_GO. 07f showed that replacing the full 512-D vector at
`blocks.0.mlp.hook_post`, token `'='`, is a sufficient interchange for the
donor sum. 07g locates the *linear output channel* of that vector. Neither
result proves that modular addition is *computed* at this hook.

## What `L0_mlp_post` is

```
resid_mid['=']
    -> W_in, ReLU          # already finished before this hook
    -> hook_post  h ∈ R^{512}
    -> W_out               # linear
    -> add resid_mid       # skip
    -> resid_post
    -> W_U                 # linear, no LayerNorm in this run
    -> logits
```

Edits at `hook_post` therefore act through the exact map `T = W_out W_U`.
Part A is that identity, not a discovered algorithm. Successful interchange
here means: the final answer's linear readout of the MLP *output features*
is causally sufficient. It does not mean the trig identity `cos(a+b)=…` is
implemented in those 512 coordinates.

## What would count as computing the sum

A genuine modular-composition mechanism has to mix `a` and `b`. That mixing
cannot happen at `hook_post`; ReLU has already run. Candidate earlier sites:

| Site | Hook | Why it could compute the sum |
| --- | --- | --- |
| Operand embeddings | `hook_embed` pos 0, 1 | Fourier features of `a` and of `b` separately. Not the sum. |
| Attention at `'='` | `blocks.0.attn.hook_z` per head, pos 2 | Can copy/combine operand information into `'='`. Heads run **in parallel**. |
| Attention output | `hook_attn_out` / `resid_mid` pos 2 | `resid_mid = resid_pre + Σ_h z_h W_O_h`. First joint residual at `'='`. |
| MLP input | `mlp.hook_pre` pos 2 | Linear map of `resid_mid`. Last linear site before the nonlinearity. |
| MLP nonlinearity | `hook_pre → hook_post` | ReLU; the only nonlinearity on this path. If the model uses a trig identity in the MLP, it is here. |
| MLP output in residual space | `hook_mlp_out` / add into `resid_post` | Linear image of `hook_post` under `W_out`. Same causal channel as T, in 128-D. |

The next justified *mechanism* experiment is interchange or path patching at
`resid_mid` and `mlp.hook_pre`, not a new obstruction score on the 10-D Fourier
germ at `hook_post`.

## Actual DAG (one layer, no LN)

Do not use the 07c chain `embed0→embed1→resid_pre→H0→H1→H2→H3→…`.

```
embed[a], embed[b], embed['=']
        \\      |       /
         v      v      v
      resid_pre[0,1,2]          # layer 0 residual = embeddings (+ pos embed)
               |
      attention, 4 heads PARALLEL
        Q,K,V from all positions
        z_h['='] ∈ R^{d_head}
        attn_out = concat(z) @ W_O
               |
      resid_mid['='] = resid_pre['='] + attn_out['=']
               |
      mlp_pre = resid_mid @ W_in + b_in     linear
      mlp_post = ReLU(mlp_pre)              nonlinear   ← 07f/07g hook
      mlp_out = mlp_post @ W_out + b_out    linear
               |
      resid_post['='] = resid_mid['='] + mlp_out
               |
      logits = resid_post['='] @ W_U        linear
```

Linear transports exist along `W_in`, `W_out`, `W_O`, `W_U`. Attention mixing
and ReLU cannot be a single input-independent linear transport.

Obstruction, if rebuilt, may only be defined on this DAG. The 10-D Fourier
ridge germ at `hook_post` is a low-energy readout inside the output channel
and is not the object to put on that graph.
"""


def make_figures(figures_dir: Path, T_info, b_rows, c_rows, compose_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = T_info["singular_values"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(1, len(s) + 1), s, marker="o", ms=3)
    ax.set_xlabel("component")
    ax.set_ylabel("singular value of T")
    ax.set_title("07g Part A — SVD of W_out W_U")
    fig.tight_layout()
    fig.savefig(figures_dir / "07g_T_singular_values.png", dpi=150)
    plt.close(fig)

    test_proj = [
        r
        for r in b_rows
        if r.get("split") == "test" and r.get("variant") == "proj" and r.get("delta", 0) != 0
    ]
    if test_proj:
        fig, ax = plt.subplots(figsize=(7, 4))
        methods = sorted({r["method"] for r in test_proj})
        for method in methods:
            by_k: dict[int, list[float]] = {}
            for r in test_proj:
                if r["method"] != method:
                    continue
                by_k.setdefault(int(r["rank"]), []).append(float(r["frac_full_dP_recovered"]))
            ks = sorted(by_k)
            ax.plot(ks, [sum(by_k[k]) / len(by_k[k]) for k in ks], marker="o", label=method)
        ax.set_xlabel("rank")
        ax.set_ylabel("fraction of full-patch ΔP(donor) recovered")
        ax.set_title("07g Part B — different-sum test pairs")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(figures_dir / "07g_rank_recovery.png", dpi=150)
        plt.close(fig)

    test_g = [r for r in c_rows if r.get("split") == "test"]
    if test_g:
        fig, ax = plt.subplots(figsize=(7, 4))
        methods = sorted({r["method"] for r in test_g})
        for method in methods:
            sub = [r for r in test_g if r["method"] == method]
            xs = [int(r["delta"]) for r in sub]
            ys = [float(r["mean_delta_p_donor"]) for r in sub]
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order], marker="o", label=method)
        ax.set_xlabel("Δ")
        ax.set_ylabel("ΔP(s+Δ) from learned G_Δ (no donor h)")
        ax.set_title("07g Part C — held-out shift generalization")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "07g_shift_generalization.png", dpi=150)
        plt.close(fig)

    if compose_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [f"{r['method']}\n{r['delta_a']}+{r['delta_b']}" for r in compose_rows]
        ax.bar(range(len(compose_rows)), [float(r["fro_rel_error"]) for r in compose_rows])
        ax.set_xticks(range(len(compose_rows)), labels, fontsize=7, rotation=30)
        ax.set_ylabel("relative Frobenius error")
        ax.set_title("07g Part C — G_Δ1 G_Δ2 vs G_Δ1+Δ2")
        fig.tight_layout()
        fig.savefig(figures_dir / "07g_composition.png", dpi=150)
        plt.close(fig)


def decide(T_info, a_rows, b_rows, c_rows, compose_rows, selected_ranks) -> tuple[str, str, str]:
    test_diff = [
        r
        for r in b_rows
        if r.get("split") == "test" and r.get("variant") == "proj" and int(r.get("delta", 0)) != 0
    ]
    if not a_rows or not test_diff:
        return "D", "Inconclusive", "Missing Part A or Part B test pairs."

    max_rel = max(float(r["max_rel_error_z"]) for r in a_rows)
    ker_p = max(abs(float(r["ker_mean_delta_p_donor"])) for r in a_rows)
    if max_rel > 1e-2:
        return (
            "D",
            "Inconclusive",
            f"The exact map T does not match empirical logit changes (max relative error {max_rel:.3g}).",
        )

    def mean_recov(method: str, k: int) -> float:
        vals = [
            float(r["frac_full_dP_recovered"])
            for r in test_diff
            if r["method"] == method and int(r["rank"]) == k
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    t_full_k = min(128, int(T_info["numerical_rank"]))
    # nearest evaluated rank
    eval_ks = sorted({int(r["rank"]) for r in test_diff})
    t_k = min(eval_ks, key=lambda k: abs(k - t_full_k)) if eval_ks else 128
    rec_T = mean_recov("T_top", t_k)
    rec_T16 = mean_recov("T_top", 16) if 16 in eval_ks else float("nan")
    rec_T32 = mean_recov("T_top", 32) if 32 in eval_ks else float("nan")
    rec_rand16 = mean_recov("random", 16) if 16 in eval_ks else float("nan")
    rec_fourier = mean_recov("fourier_probe", min(8, max(eval_ks) if eval_ks else 8))
    rec_diff16 = mean_recov("svd_donor_diff", 16) if 16 in eval_ks else float("nan")
    rec_diff32 = mean_recov("svd_donor_diff", 32) if 32 in eval_ks else float("nan")

    compact_T = (rec_T16 >= 0.8) or (rec_T32 >= 0.8)
    compact_learned = (rec_diff16 >= 0.8 and rec_diff16 > rec_rand16 + 0.15) or (
        rec_diff32 >= 0.8 and rec_diff32 > mean_recov("random", 32) + 0.15
    )
    high_rank_needed = not compact_T and rec_T < 0.8

    test_g = [r for r in c_rows if r.get("split") == "test"]
    def g_score(method: str) -> float:
        vals = [float(r["mean_delta_p_donor"]) for r in test_g if r["method"] == method]
        return sum(vals) / len(vals) if vals else float("nan")

    g_T = g_score("T_top")
    g_diff = g_score("svd_donor_diff")
    g_fourier = g_score("fourier_probe")
    g_rand = g_score("random")
    modular_learned = (g_diff == g_diff) and g_diff >= 0.15 and (not (g_rand == g_rand) or g_diff > g_rand + 0.1)
    modular_T = (g_T == g_T) and g_T >= 0.15
    compose_ok = False
    if compose_rows:
        learned_comp = [r for r in compose_rows if r["method"] in {"svd_donor_diff", "pca_acts"}]
        pool = learned_comp or compose_rows
        compose_ok = sum(float(r["fro_rel_error"]) for r in pool) / len(pool) < 0.35

    _ = selected_ranks, rec_fourier, ker_p, modular_T, compose_ok

    if compact_learned and modular_learned:
        return (
            "A",
            "Compact modular causal representation",
            f"Donor-difference SVD recovers targeted interchange at rank 16/32 "
            f"(recov 16={rec_diff16:.3f}, 32={rec_diff32:.3f}) and learned G_Δ "
            f"moves held-out pairs toward s+Δ (mean ΔP={g_diff:.3f}), above random "
            f"({g_rand:.3f}). Fourier probe remains a weak readout (recov={rec_fourier:.3f}).",
        )
    if (compact_T or rec_T >= 0.8) and not modular_learned:
        return (
            "B",
            "Causal answer-output channel only",
            f"The exact map T (rank {T_info['numerical_rank']}) accounts for donor "
            f"patches (T recovery {rec_T:.3f} at k={t_k}; k=16 {rec_T16:.3f}). "
            f"Compact learned bases and G_Δ do not validate a modular operator "
            f"(donor-SVD ΔP={g_diff:.3f}, Fourier ΔP={g_fourier:.3f}). "
            f"T-top is the output channel, not a discovered algorithm.",
        )
    if high_rank_needed and not compact_learned:
        return (
            "C",
            "Distributed or high-rank representation",
            f"Even top singular directions of T need high rank to recover the "
            f"interchange (k=16 {rec_T16:.3f}, k=32 {rec_T32:.3f}, k={t_k} {rec_T:.3f}). "
            f"Compact Fourier/donor-SVD bases fail.",
        )
    return (
        "D",
        "Inconclusive",
        "Rank curves and shift maps do not cleanly separate a compact modular "
        f"operator from the output channel (T16={rec_T16:.3f}, diff16={rec_diff16:.3f}, "
        f"G_Δ T={g_T:.3f}, G_Δ diff={g_diff:.3f}).",
    )


def recommend(letter: str) -> str:
    if letter == "A":
        return (
            "Next: put obstruction on the recovered compact donor-difference "
            "(or jointly T-aligned) subspace, using the real DAG in 07g Part D, "
            "still at step 9080. Do not reuse the 10-D Fourier germ."
        )
    if letter == "B":
        return (
            "Next: test modular composition at earlier sites on the real DAG "
            "(`resid_mid`, `mlp.hook_pre`), using T as the known output channel "
            "from `hook_post`. Do not rebuild O on the Fourier germ or treat "
            "col(T) as a cyclic algorithm."
        )
    if letter == "C":
        return (
            "Next: accept a high-rank output channel at `hook_post` and move "
            "mechanism tests to `resid_mid` / attention, or to a joint multi-site "
            "interchange. A low-rank obstruction at this hook is not justified."
        )
    return (
        "Next: enlarge val/test pair counts or fix the T identity residual "
        "before any new representation claim. 07d stays NO_GO."
    )


def main() -> None:
    import torch

    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    figures_dir = ensure_dir(args.figures_dir)
    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Pass step_0009080.pt from the 07d run.")

    model, cfg, step = load_checkpoint(path, device=args.device)
    if step != 9080:
        raise RuntimeError(f"Refusing step {step}; 07g is frozen at 9080.")
    if cfg.n_layers != 1:
        raise RuntimeError("07g assumes the audited one-layer model.")
    model.eval()
    device = next(model.parameters()).device
    print(f"07g checkpoint step {step} device={device}")
    print("07d operational label remains NO_GO.")

    tokens, labels = make_dataset(cfg)
    (train_tokens, train_labels), (test_tokens, test_labels) = split_dataset(cfg, tokens, labels)
    train_tokens, train_labels = train_tokens.to(device), train_labels.to(device)
    test_tokens, test_labels = test_tokens.to(device), test_labels.to(device)
    freqs = key_frequencies(model, cfg.p, top_k=args.n_freqs)

    with torch.no_grad():
        train_logits = model(train_tokens)[:, -1, :]
        test_logits = model(test_tokens)[:, -1, :]
    train_correct = train_logits.argmax(dim=-1) == train_labels
    test_correct = test_logits.argmax(dim=-1) == test_labels
    print(
        f"train_acc={float(train_correct.float().mean()):.4f} "
        f"test_acc={float(test_correct.float().mean()):.4f}"
    )

    train_h = cache_mlp_post(model, train_tokens)
    test_h = cache_mlp_post(model, test_tokens)
    val_idx, eval_idx = split_pool_indices(test_tokens.shape[0], args.val_frac, args.val_seed)
    train_pool = (train_tokens, train_labels, train_correct, train_h)
    val_pool = take_pool(test_tokens, test_labels, test_correct, test_h, val_idx)
    eval_pool = take_pool(test_tokens, test_labels, test_correct, test_h, eval_idx)
    print(f"val n={val_idx.numel()} eval n={eval_idx.numel()} (original test split, seed={args.val_seed})")

    T = output_map_T(model)
    T_info = svd_map(T, args.sv_tol)
    print(
        f"T shape={tuple(T.shape)} rank={T_info['numerical_rank']} "
        f"participation_ratio={T_info['participation_ratio']:.2f} "
        f"top5 s={[round(v, 4) for v in T_info['singular_values'][:5]]}"
    )
    write_csv(
        results_dir / "07g_T_spectrum.csv",
        [
            {
                "k": i + 1,
                "singular_value": v,
                "energy_frac_cum": T_info["energy_frac_by_k"][i],
            }
            for i, v in enumerate(T_info["singular_values"])
        ],
    )

    def bundle_dict(pool, tag_seed: int):
        out = {}
        for delta in (0,) + SHIFT_DELTAS:
            out[delta] = gather_pairs(
                pool, delta, args.n_recipients, args.n_donors_each, args.seed + tag_seed + 9 * delta, cfg.p
            )
            if out[delta] is not None:
                print(f"  pairs seed{tag_seed} Δ={delta} n={out[delta]['rec'].numel()}")
        return out

    print("Building pair pools")
    val_bundles = bundle_dict(val_pool, 100)
    test_bundles = bundle_dict(eval_pool, 200)
    a_bundles = {f"test_delta_{d}": test_bundles[d] for d in test_bundles}

    print("Part A — exact T identity")
    a_rows = part_a(model, T, T_info, a_bundles, args.batch_size)
    write_table(results_dir / "07g_partA_T_identity.csv", a_rows)
    for r in a_rows:
        print(
            f"  {r['bundle']}: max|Δz-ΔhT|={r['max_abs_error_z']:.3e} "
            f"rel={r['max_rel_error_z']:.3e} T_dP={r['T_proj_mean_delta_p_donor']:+.3f} "
            f"ker_dP={r['ker_mean_delta_p_donor']:+.3f}"
        )

    print("Part B — rank-limited interchange")
    train_diffs = collect_train_diffs(
        train_pool, cfg.p, args.n_train_diff_recipients, args.n_donors_each, args.seed
    )
    print(f"  train donor-diff matrix {tuple(train_diffs.shape)}")
    overlap_rows = []
    b_rows = []
    full_cache: dict[tuple[str, int], dict] = {}
    for split, bundles in (("val", val_bundles), ("test", test_bundles)):
        for delta, bundle in bundles.items():
            if bundle is None:
                continue
            full_row, _, _ = eval_delta_patch(
                model, bundle, bundle["h_don"].float() - bundle["h_rec"].float(), args.batch_size
            )
            full_row.update({"split": split, "method": "full_patch", "rank": bundle["h_rec"].shape[1], "delta": delta, "variant": "full"})
            b_rows.append(full_row)
            full_cache[(split, delta)] = full_row

    selected_ranks: dict[str, int] = {}
    bases_selected: dict[str, object] = {}
    for k in RANKS:
        bases, _W_f = build_method_bases(
            train_h, train_tokens, train_diffs, T_info, cfg.p, freqs, args.ridge, args.seed, k
        )
        for method, B in bases.items():
            overlap_rows.append(
                {
                    "rank": k,
                    "method": method,
                    "overlap_with_T": subspace_overlap(B, top_k(T_info["U"], min(k, T_info["U"].shape[1]))),
                    "basis_dim": int(B.shape[1]),
                }
            )
        for split, bundles in (("val", val_bundles), ("test", test_bundles)):
            for delta, bundle in bundles.items():
                if bundle is None:
                    continue
                full_row = full_cache[(split, delta)]
                for method in bases:
                    b_rows.extend(
                        part_b_eval(model, bases, bundle, full_row, args.batch_size, k, split, method)
                    )

    # Select rank per method on val different-sum proj variant.
    methods = sorted({r["method"] for r in b_rows if r["method"] != "full_patch"})
    for method in methods:
        scores = []
        for k in RANKS:
            vals = [
                float(r["mean_delta_p_donor"])
                for r in b_rows
                if r.get("split") == "val"
                and r.get("method") == method
                and r.get("variant") == "proj"
                and int(r.get("rank", -1)) == k
                and int(r.get("delta", 0)) != 0
            ]
            if vals:
                scores.append((sum(vals) / len(vals), k))
        if scores:
            selected_ranks[method] = max(scores)[1]
            print(f"  val-selected rank {method}={selected_ranks[method]}")

    write_table(results_dir / "07g_partB_rank_limited.csv", b_rows)
    write_table(results_dir / "07g_subspace_overlap.csv", overlap_rows)

    bases_for_c = {}
    for method, k in selected_ranks.items():
        bases, _ = build_method_bases(
            train_h, train_tokens, train_diffs, T_info, cfg.p, freqs, args.ridge, args.seed, k
        )
        bases_for_c[method] = bases[method]

    print("Part C — shift operators G_Δ")
    c_rows, compose_rows, best_ridge = part_c(
        model,
        train_pool,
        val_bundles,
        test_bundles,
        bases_for_c,
        list(selected_ranks),
        cfg.p,
        args.n_recipients,
        args.n_donors_each,
        args.seed,
        list(args.g_ridges),
        args.batch_size,
    )
    print(f"  val-selected G ridge={best_ridge}")
    write_table(results_dir / "07g_partC_shift_operators.csv", c_rows)
    write_table(results_dir / "07g_partC_composition.csv", compose_rows)
    for r in c_rows:
        if r["split"] == "test" and r["method"] in {"T_top", "svd_donor_diff", "fourier_probe"}:
            print(
                f"  {r['method']} Δ={r['delta']} ΔP={r['mean_delta_p_donor']:+.3f} "
                f"pred {r['frac_pred_donor_before']:.3f}->{r['frac_pred_donor_after']:.3f} "
                f"repr_rel={r['repr_rel_rmse']:.3f}"
            )

    write_text(results_dir / "07g_partD_computation_location.md", part_d_text())
    make_figures(figures_dir, T_info, b_rows, c_rows, compose_rows)

    letter, title, explanation = decide(T_info, a_rows, b_rows, c_rows, compose_rows, selected_ranks)
    rec = recommend(letter)
    summary = {
        "official_07d_label": "NO_GO",
        "checkpoint_step": 9080,
        "T_shape": list(T.shape),
        "T_numerical_rank": T_info["numerical_rank"],
        "T_participation_ratio": T_info["participation_ratio"],
        "val_selected_ranks": selected_ranks,
        "g_ridge": best_ridge,
        "decision": letter,
        "decision_title": title,
        "explanation": explanation,
        "recommendation": rec,
    }
    write_text(results_dir / "07g_decision.json", json.dumps(summary, indent=2) + "\n")
    md = "\n".join(
        [
            "# 07g causal subspace recovery",
            "",
            "**Official 07d label: NO_GO (unchanged).**",
            "",
            f"**07g decision: {letter} — {title}**",
            "",
            explanation,
            "",
            f"- T = W_out W_U has shape {tuple(T.shape)}, numerical rank {T_info['numerical_rank']}, "
            f"participation ratio {T_info['participation_ratio']:.2f}",
            f"- val-selected ranks: {selected_ranks}",
            f"- G_Δ ridge selected on val: {best_ridge}",
            "",
            "These results identify (or fail to identify) a causal output channel at "
            "`L0_mlp_post`. They are not an obstruction test.",
            "",
            "## Recommendation",
            "",
            rec,
            "",
        ]
    )
    write_text(results_dir / "07g_decision_summary.md", md)
    print("\n" + md)
    print(f"saved {results_dir}")
    print(f"saved {figures_dir}")


if __name__ == "__main__":
    main()
