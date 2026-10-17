"""Experiment 0B: donor interchange at earlier sites than mlp.hook_post.

Frozen checkpoint 9080. Does not retrain. Official 07d remains NO_GO.

07g showed that hook_post is the linear unembed channel T = W_out W_U, not the
adder. This script asks whether the sum is already written at the two sites
where a and b can still mix into the MLP:

  resid_mid['=']  = residual after attention, before the MLP
  mlp.hook_pre    = linear MLP input, immediately before ReLU

Full donor patch is the positive control. Rank / G_Δ run only if that control
moves the recipient toward the donor sum. There is no exact global T here:
ReLU sits downstream, so any linear map is a local approximation.
"""

from __future__ import annotations

import argparse
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
    fit_G,
    gather_pairs,
    output_map_T,
    split_pool_indices,
    stratified,
    subspace_overlap,
    svd_map,
    take_pool,
    write_table,
)

SITES = (
    {"site_id": "L0_resid_mid", "hook_name": "blocks.0.hook_resid_mid", "position": 2},
    {"site_id": "L0_mlp_pre", "hook_name": "blocks.0.mlp.hook_pre", "position": 2},
)
FULL_PATCH_DP = 0.25
G_MODULAR_DP = 0.40
COMPOSE_FRO = 0.35
COMPACT_RECOV = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="07h earlier-site interchange at step 9080.")
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def cache_site(model, tokens, hook_name: str, position: int, batch_size: int = 2048):
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start : start + batch_size]
            _, cache = model.run_with_cache(batch, names_filter=lambda n, h=hook_name: n == h)
            chunks.append(cache[hook_name][:, position, :].detach().cpu())
    return torch.cat(chunks, dim=0)


def logits_with_site(model, tokens, hook_name: str, position: int, replacement, batch_size: int):
    import torch

    outs = []
    n = tokens.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = tokens[start:end]
            h_new = replacement[start:end]

            def hook(acts, hook, h=h_new, pos=position):  # noqa: ARG001
                acts[:, pos, :] = h.to(device=acts.device, dtype=acts.dtype)
                return acts

            with model.hooks(fwd_hooks=[(hook_name, hook)]):
                outs.append(model(batch)[:, -1, :].detach())
    return torch.cat(outs, dim=0)


def eval_site_patch(model, bundle, delta_h, hook_name: str, position: int, batch_size: int):
    import torch

    with torch.no_grad():
        logits_b = model(bundle["tokens"])[:, -1, :]
    h_new = bundle["h_rec"].float() + delta_h.float()
    logits_a = logits_with_site(model, bundle["tokens"], hook_name, position, h_new, batch_size)
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


def weight_maps(model):
    W_in = model.blocks[0].mlp.W_in.detach().float().cpu()
    W_out = model.blocks[0].mlp.W_out.detach().float().cpu()
    T_post = output_map_T(model)
    if W_in.shape[0] == W_out.shape[0]:
        # some TL builds store W_in as [d_mlp, d_model]
        W_in_use = W_in.T
    elif W_in.shape[1] == W_out.shape[0]:
        W_in_use = W_in
    else:
        raise RuntimeError(f"Cannot align W_in {tuple(W_in.shape)} with W_out {tuple(W_out.shape)}")
    return W_in_use, T_post


def linearized_T(site_id: str, W_in, T_post):
    """Global linear map pretending every ReLU unit is on. Not exact."""
    if site_id == "L0_mlp_pre":
        return T_post
    if site_id == "L0_resid_mid":
        return W_in @ T_post
    raise KeyError(site_id)


def local_pred_delta_z(site_id: str, delta, h_rec_pre, W_in, T_post):
    """First-order ReLU-gated prediction of Δz. Large donor deltas can break it."""
    gate = (h_rec_pre.float() > 0).float()
    if site_id == "L0_mlp_pre":
        return (delta.float() * gate) @ T_post
    if site_id == "L0_resid_mid":
        return ((delta.float() @ W_in) * gate) @ T_post
    raise KeyError(site_id)


def fourier_basis(train_h, train_tokens, p: int, freqs, ridge: float, k: int):
    import torch

    y = latent_targets(train_tokens, p, freqs).cpu()
    W = fit_ridge(train_h, y, ridge)
    U, s, _ = torch.linalg.svd(W[:-1].float(), full_matrices=False)
    return U[:, : min(k, int(s.numel()))]


def build_bases(train_h, train_tokens, train_diffs, U_lin, p, freqs, ridge, seed, k: int):
    import torch

    d = train_h.shape[1]
    _, _, Vh = torch.linalg.svd(train_diffs.float(), full_matrices=False)
    return {
        "T_lin_top": U_lin[:, : min(k, U_lin.shape[1])],
        "pca_acts": pca_subspace(train_h, min(k, d)),
        "svd_donor_diff": Vh[: min(k, Vh.shape[0])].T,
        "random": random_subspace(d, min(k, d), seed),
        "fourier_probe": fourier_basis(train_h, train_tokens, p, freqs, ridge, k),
    }


def site_full_patch_ok(rows: list[dict]) -> bool:
    diffs = [r for r in rows if int(r.get("delta", 0)) != 0]
    if not diffs:
        return False
    dP = sum(float(r["mean_delta_p_donor"]) for r in diffs) / len(diffs)
    pred = sum(float(r["frac_pred_donor_after"]) - float(r["frac_pred_donor_before"]) for r in diffs) / len(diffs)
    return dP >= FULL_PATCH_DP and pred >= 0.10


def run_full_patches(model, site, bundles, batch_size: int, split: str):
    rows = []
    for delta, bundle in bundles.items():
        if bundle is None:
            continue
        delta_h = bundle["h_don"].float() - bundle["h_rec"].float()
        metrics, _, _ = eval_site_patch(
            model, bundle, delta_h, site["hook_name"], site["position"], batch_size
        )
        metrics.update(
            {
                "site_id": site["site_id"],
                "hook_name": site["hook_name"],
                "split": split,
                "delta": bundle["delta"],
                "method": "full_patch",
                "variant": "full",
            }
        )
        rows.append(metrics)
        print(
            f"  {site['site_id']} {split} Δ={delta} n={metrics['n_pairs']} "
            f"ΔP={metrics['mean_delta_p_donor']:+.3f} "
            f"pred {metrics['frac_pred_donor_before']:.3f}->{metrics['frac_pred_donor_after']:.3f}"
        )
    return rows


def run_linearization_check(site, test_bundles, pre_by_idx, W_in, T_post, model, batch_size):
    """Compare empirical Δz to ReLU-gated first-order map on donor deltas."""
    import torch

    rows = []
    for delta, bundle in test_bundles.items():
        if bundle is None:
            continue
        dh = bundle["h_don"].float() - bundle["h_rec"].float()
        _, z_b, z_a = eval_site_patch(
            model, bundle, dh, site["hook_name"], site["position"], batch_size
        )
        emp = (z_a - z_b).float().cpu()
        if site["site_id"] == "L0_mlp_pre":
            h_pre = bundle["h_rec"]
        else:
            h_pre = pre_by_idx[bundle["rec"].cpu()]
        pred = local_pred_delta_z(site["site_id"], dh, h_pre, W_in, T_post)
        abs_err = (emp - pred).abs()
        rows.append(
            {
                "site_id": site["site_id"],
                "delta": delta,
                "n_pairs": int(dh.shape[0]),
                "max_abs_error_z": float(abs_err.max()),
                "mean_abs_error_z": float(abs_err.mean()),
                "max_rel_error_z": float(abs_err.max() / (emp.abs().max() + 1e-12)),
                "note": "ReLU-gated first-order map; not an exact identity",
            }
        )
    return rows


def run_rank_and_G(args, model, cfg, site, freqs, train_pool, val_bundles, test_bundles, full_cache, U_lin):
    train_tokens, _, _, train_h = train_pool
    train_diffs = collect_train_diffs(
        train_pool, cfg.p, args.n_train_diff_recipients, args.n_donors_each, args.seed
    )
    print(f"  {site['site_id']} train diffs {tuple(train_diffs.shape)}")
    b_rows = []
    overlap_rows = []
    for k in RANKS:
        if k > train_h.shape[1]:
            continue
        bases = build_bases(
            train_h, train_tokens, train_diffs, U_lin, cfg.p, freqs, args.ridge, args.seed, k
        )
        for method, B in bases.items():
            overlap_rows.append(
                {
                    "site_id": site["site_id"],
                    "rank": k,
                    "method": method,
                    "overlap_with_T_lin": subspace_overlap(B, U_lin[:, : min(k, U_lin.shape[1])]),
                    "basis_dim": int(B.shape[1]),
                }
            )
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
                        metrics, _, _ = eval_site_patch(
                            model, bundle, applied, site["hook_name"], site["position"], args.batch_size
                        )
                        recov = (
                            metrics["mean_delta_p_donor"] / full_row["mean_delta_p_donor"]
                            if abs(full_row["mean_delta_p_donor"]) > 1e-6
                            else float("nan")
                        )
                        row = {
                            "site_id": site["site_id"],
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
    methods = sorted({r["method"] for r in b_rows})
    for method in methods:
        scores = []
        for k in RANKS:
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
            print(f"  {site['site_id']} val rank {method}={selected[method]}")

    bases_c = {}
    for method, k in selected.items():
        bases_c[method] = build_bases(
            train_h, train_tokens, train_diffs, U_lin, cfg.p, freqs, args.ridge, args.seed, k
        )[method]

    c_rows, compose_rows = run_G(
        model,
        site,
        train_pool,
        val_bundles,
        test_bundles,
        bases_c,
        cfg.p,
        args,
    )
    return b_rows, overlap_rows, c_rows, compose_rows, selected


def run_G(model, site, train_pool, val_bundles, test_bundles, bases, p, args):
    if not bases:
        return [], []
    probe = "T_lin_top" if "T_lin_top" in bases else next(iter(bases))
    B0 = bases[probe]
    best_ridge, best = args.g_ridges[0], -1e9
    for ridge in args.g_ridges:
        scores = []
        for delta in SHIFT_DELTAS:
            tr = gather_pairs(train_pool, delta, args.n_recipients, args.n_donors_each, args.seed + 40 * delta, p)
            va = val_bundles.get(delta)
            if tr is None or va is None:
                continue
            G = fit_G(tr["h_rec"].float() @ B0, tr["h_don"].float() @ B0, ridge)
            metrics, _, _ = eval_site_patch(
                model,
                va,
                apply_G(va["h_rec"], B0, G) - va["h_rec"].float(),
                site["hook_name"],
                site["position"],
                args.batch_size,
            )
            scores.append(metrics["mean_delta_p_donor"])
        if scores and sum(scores) / len(scores) > best:
            best, best_ridge = sum(scores) / len(scores), ridge
    print(f"  {site['site_id']} G ridge={best_ridge}")

    c_rows, compose_rows = [], []
    for method, B in bases.items():
        G_by = {}
        for delta in SHIFT_DELTAS:
            tr = gather_pairs(train_pool, delta, args.n_recipients, args.n_donors_each, args.seed + 40 * delta, p)
            if tr is None:
                continue
            G = fit_G(tr["h_rec"].float() @ B, tr["h_don"].float() @ B, best_ridge)
            G_by[delta] = G
            for split, bundle in (("val", val_bundles.get(delta)), ("test", test_bundles.get(delta))):
                if bundle is None:
                    continue
                metrics, _, _ = eval_site_patch(
                    model,
                    bundle,
                    apply_G(bundle["h_rec"], B, G) - bundle["h_rec"].float(),
                    site["hook_name"],
                    site["position"],
                    args.batch_size,
                )
                C_r = bundle["h_rec"].float() @ B
                C_d = bundle["h_don"].float() @ B
                rel = float(((C_r @ G) - C_d).pow(2).mean().sqrt() / (C_d.pow(2).mean().sqrt() + 1e-12))
                row = {
                    "site_id": site["site_id"],
                    "method": method,
                    "rank": int(B.shape[1]),
                    "delta": delta,
                    "split": split,
                    "ridge": best_ridge,
                    "repr_rel_rmse": rel,
                }
                row.update(metrics)
                c_rows.append(row)
                if split == "test" and method in {"T_lin_top", "svd_donor_diff", "fourier_probe"}:
                    print(
                        f"  {site['site_id']} {method} Δ={delta} G_ΔP={metrics['mean_delta_p_donor']:+.3f} "
                        f"pred {metrics['frac_pred_donor_before']:.3f}->{metrics['frac_pred_donor_after']:.3f}"
                    )
        for a, b, c in COMPOSE_PAIRS:
            if a not in G_by or b not in G_by or c not in G_by:
                continue
            bundle = test_bundles.get(c)
            if bundle is None:
                continue
            Ga, Gb, Gc = G_by[a], G_by[b], G_by[c]
            fro = float(((Ga @ Gb) - Gc).pow(2).mean().sqrt() / (Gc.pow(2).mean().sqrt() + 1e-12))
            met_c, _, _ = eval_site_patch(
                model,
                bundle,
                apply_G(apply_G(bundle["h_rec"], B, Ga), B, Gb) - bundle["h_rec"].float(),
                site["hook_name"],
                site["position"],
                args.batch_size,
            )
            met_d, _, _ = eval_site_patch(
                model,
                bundle,
                apply_G(bundle["h_rec"], B, Gc) - bundle["h_rec"].float(),
                site["hook_name"],
                site["position"],
                args.batch_size,
            )
            compose_rows.append(
                {
                    "site_id": site["site_id"],
                    "method": method,
                    "rank": int(B.shape[1]),
                    "delta_a": a,
                    "delta_b": b,
                    "delta_c": c,
                    "fro_rel_error": fro,
                    "compose_mean_delta_p_donor": met_c["mean_delta_p_donor"],
                    "direct_mean_delta_p_donor": met_d["mean_delta_p_donor"],
                }
            )
    return c_rows, compose_rows


def decide_site(site_id: str, full_ok: bool, b_rows, c_rows, compose_rows) -> tuple[str, str]:
    if not full_ok:
        return (
            "SITE_FAIL",
            f"{site_id}: full donor patch does not move the recipient to the donor sum. "
            "The adder is not sitting at this hook as a replaceable activation.",
        )
    test = [
        r
        for r in b_rows
        if r.get("split") == "test" and r.get("variant") == "proj" and int(r.get("delta", 0)) != 0
    ]

    def recov(method, k):
        vals = [float(r["frac_full_dP_recovered"]) for r in test if r["method"] == method and int(r["rank"]) == k]
        return sum(vals) / len(vals) if vals else float("nan")

    t16, d32, f8, r16 = recov("T_lin_top", 16), recov("svd_donor_diff", 32), recov("fourier_probe", 8), recov("random", 16)
    g_test = [r for r in c_rows if r.get("split") == "test"]
    def gmean(method):
        vals = [float(r["mean_delta_p_donor"]) for r in g_test if r["method"] == method]
        return sum(vals) / len(vals) if vals else float("nan")

    g_t, g_d, g_r = gmean("T_lin_top"), gmean("svd_donor_diff"), gmean("random")
    compose = [r for r in compose_rows if r["method"] in {"T_lin_top", "svd_donor_diff"}]
    fro = sum(float(r["fro_rel_error"]) for r in compose) / len(compose) if compose else 1e9
    compact = (t16 >= COMPACT_RECOV) or (d32 >= COMPACT_RECOV and d32 > r16 + 0.15)
    modular = (g_d == g_d and g_d >= G_MODULAR_DP and g_d > (g_r if g_r == g_r else 0) + 0.15 and fro < COMPOSE_FRO)
    if compact and modular:
        return (
            "A",
            f"{site_id}: compact subspace recovers interchange and G_Δ composes "
            f"(T16={t16:.3f}, diff32={d32:.3f}, G_Δ={g_d:.3f}, fro={fro:.3f}).",
        )
    return (
        "B",
        f"{site_id}: full patch works, so the answer can be written here, but this is not "
        f"a modular shift operator (T16={t16:.3f}, Fourier8={f8:.3f}, G_Δ T={g_t:.3f} "
        f"diff={g_d:.3f}, compose fro={fro:.3f}). Same bar as 07g: B, not A.",
    )


def make_figures(figures_dir: Path, full_rows, b_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_full = [r for r in full_rows if r.get("split") == "test" and int(r.get("delta", 0)) != 0]
    if test_full:
        fig, ax = plt.subplots(figsize=(7, 4))
        sites = sorted({r["site_id"] for r in test_full})
        for site in sites:
            sub = [r for r in test_full if r["site_id"] == site]
            xs = [int(r["delta"]) for r in sub]
            ys = [float(r["mean_delta_p_donor"]) for r in sub]
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order], marker="o", label=site)
        ax.axhline(FULL_PATCH_DP, color="black", ls="--", lw=0.8, label="full-patch gate")
        ax.set_xlabel("Δ")
        ax.set_ylabel("full-patch ΔP(donor sum)")
        ax.set_title("07h — is the answer already written here?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "07h_full_patch.png", dpi=150)
        plt.close(fig)

    test_b = [
        r
        for r in b_rows
        if r.get("split") == "test" and r.get("variant") == "proj" and int(r.get("delta", 0)) != 0
    ]
    for site in sorted({r["site_id"] for r in test_b}):
        fig, ax = plt.subplots(figsize=(7, 4))
        sub = [r for r in test_b if r["site_id"] == site]
        for method in sorted({r["method"] for r in sub}):
            by_k: dict[int, list[float]] = {}
            for r in sub:
                if r["method"] != method:
                    continue
                by_k.setdefault(int(r["rank"]), []).append(float(r["frac_full_dP_recovered"]))
            ks = sorted(by_k)
            ax.plot(ks, [sum(by_k[k]) / len(by_k[k]) for k in ks], marker="o", label=method)
        ax.set_xlabel("rank")
        ax.set_ylabel("fraction of full-patch ΔP recovered")
        ax.set_title(f"07h rank curve — {site}")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(figures_dir / f"07h_rank_{site}.png", dpi=150)
        plt.close(fig)


def recommend(letter: str) -> str:
    if letter == "A":
        return (
            "A compact modular object appeared before ReLU. Next: obstruction on that "
            "subspace and the real DAG, still at 9080. Do not use the 10-D Fourier germ."
        )
    if letter == "B":
        return (
            "The answer can be overwritten at an earlier residual/MLP-input site, but "
            "there is still no +Δ group. Next: attention / operand-copy path at '=' "
            "(hook_z, resid_pre), not obstruction on T or donor-SVD."
        )
    if letter == "C":
        return (
            "Full patches failed at resid_mid and hook_pre. The sum is not a replaceable "
            "activation at those sites. Next: per-head z at '=', or accept that the "
            "algorithm is not a local residual object. Do not rebuild O on hook_post."
        )
    return "Controls were inconclusive. Enlarge pairs or fix a hook identity before any new O claim."


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
        raise RuntimeError(f"Refusing step {step}; 07h is frozen at 9080.")
    model.eval()
    device = next(model.parameters()).device
    print(f"07h checkpoint step {step} device={device}")
    print("07d remains NO_GO. 07g at hook_post was the unembed channel, not the adder.")

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
    W_in, T_post = weight_maps(model)
    print(f"W_in {tuple(W_in.shape)} T_post {tuple(T_post.shape)}")

    all_full, all_lin, all_b, all_ov, all_c, all_comp = [], [], [], [], [], []
    site_verdicts = []

    for site in SITES:
        print(f"\n=== {site['site_id']} {site['hook_name']} ===")
        train_h = cache_site(model, train_tokens, site["hook_name"], site["position"])
        test_h = cache_site(model, test_tokens, site["hook_name"], site["position"])
        train_pool = (train_tokens, train_labels, train_ok, train_h)
        val_pool = take_pool(test_tokens, test_labels, test_ok, test_h, val_idx)
        eval_pool = take_pool(test_tokens, test_labels, test_ok, test_h, eval_idx)

        def bundles(pool, tag):
            out = {}
            for delta in (0,) + SHIFT_DELTAS:
                out[delta] = gather_pairs(
                    pool, delta, args.n_recipients, args.n_donors_each, args.seed + tag + 9 * delta, cfg.p
                )
            return out

        val_b, test_b = bundles(val_pool, 100), bundles(eval_pool, 200)
        full_val = run_full_patches(model, site, val_b, args.batch_size, "val")
        full_test = run_full_patches(model, site, test_b, args.batch_size, "test")
        all_full.extend(full_val + full_test)
        ok = site_full_patch_ok(full_test)
        print(f"  full-patch gate: {'PASS' if ok else 'FAIL'}")

        T_lin = linearized_T(site["site_id"], W_in, T_post)
        T_info = svd_map(T_lin, 1e-6)
        print(
            f"  linearized T (ReLU ignored) shape={tuple(T_lin.shape)} "
            f"rank={T_info['numerical_rank']} PR={T_info['participation_ratio']:.2f}"
        )
        eval_pre = (
            test_h
            if site["site_id"] == "L0_mlp_pre"
            else cache_site(model, eval_pool[0], "blocks.0.mlp.hook_pre", 2)
        )
        lin_rows = run_linearization_check(site, test_b, eval_pre, W_in, T_post, model, args.batch_size)
        all_lin.extend(lin_rows)
        for r in lin_rows:
            print(f"  local-linear Δ={r['delta']} rel_err={r['max_rel_error_z']:.3f} (not exact)")

        if not ok:
            site_verdicts.append(
                decide_site(site["site_id"], False, [], [], [])
            )
            continue

        full_cache = {(r["split"], r["delta"]): r for r in full_val + full_test}
        b_rows, ov, c_rows, comp, selected = run_rank_and_G(
            args, model, cfg, site, freqs, train_pool, val_b, test_b, full_cache, T_info["U"]
        )
        all_b.extend(b_rows)
        all_ov.extend(ov)
        all_c.extend(c_rows)
        all_comp.extend(comp)
        site_verdicts.append(decide_site(site["site_id"], True, b_rows, c_rows, comp))
        print(f"  decision {site_verdicts[-1][0]}: {site_verdicts[-1][1]}")

    write_table(results_dir / "07h_full_patch.csv", all_full)
    write_table(results_dir / "07h_local_linearization.csv", all_lin)
    write_table(results_dir / "07h_rank_limited.csv", all_b)
    write_table(results_dir / "07h_subspace_overlap.csv", all_ov)
    write_table(results_dir / "07h_shift_operators.csv", all_c)
    write_table(results_dir / "07h_composition.csv", all_comp)
    make_figures(figures_dir, all_full, all_b)

    letters = [v[0] for v in site_verdicts]
    if "A" in letters:
        letter, title = "A", "Compact modular representation before ReLU"
    elif "B" in letters:
        letter, title = "B", "Earlier site can write the answer, not a +Δ operator"
    elif letters and all(v == "SITE_FAIL" for v in letters):
        letter, title = "C", "Neither resid_mid nor hook_pre is a replaceable adder"
    else:
        letter, title = "D", "Inconclusive"
    explanation = " ".join(v[1] for v in site_verdicts)
    rec = recommend(letter)
    md = "\n".join(
        [
            "# 07h earlier-site interchange",
            "",
            "**Official 07d label: NO_GO (unchanged).**",
            "",
            f"**07h decision: {letter} — {title}**",
            "",
            explanation,
            "",
            "Full donor patch is the positive control. Rank / G_Δ ran only where that "
            "control passed. Linear maps here go through ReLU and are not identities.",
            "",
            "## Recommendation",
            "",
            rec,
            "",
        ]
    )
    write_text(results_dir / "07h_decision_summary.md", md)
    write_text(
        results_dir / "07h_decision.json",
        json.dumps(
            {
                "official_07d_label": "NO_GO",
                "checkpoint_step": 9080,
                "decision": letter,
                "decision_title": title,
                "site_verdicts": [{"letter": a, "text": b} for a, b in site_verdicts],
                "recommendation": rec,
            },
            indent=2,
        )
        + "\n",
    )
    print("\n" + md)
    print(f"saved {results_dir}")


if __name__ == "__main__":
    main()
