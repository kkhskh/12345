"""Experiment 0B step 4: causal intervention on the detected latent representation.

A probe that decodes the latent state proves nothing causal on its own. This script runs the
two causal tests the updated plan requires:

  necessity  - project the cyclic subspace out of a site and measure how much generalization
               breaks, against matched-dimension random, PCA, and spectral-direction controls;
  repair     - on examples the model gets wrong, edit the site so its germ matches the true
               latent state, and measure whether obstruction falls, margin rises, and the
               answer is repaired, against matched-norm edits in control subspaces.

Collateral damage is measured by applying the same edit rule to already-correct examples.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.toy_modular import (
    cache_site_activations,
    candidate_margin,
    chain_edges,
    germ_subspace,
    extract_site,
    final_logits,
    fit_latent_germs,
    fit_transports,
    germ_target_delta,
    germ_values,
    hardest_incorrect,
    key_frequencies,
    kl_divergence,
    latent_obstruction,
    latent_targets,
    load_checkpoint,
    make_dataset,
    make_toy_sites,
    match_norm_in_subspace,
    phase_rotate_delta,
    pca_subspace,
    random_subspace,
    site_ablate_hook,
    site_edit_hook,
    spectral_subspace,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causally intervene on the cyclic latent representation.")
    parser.add_argument("--checkpoint", default=None, help="defaults to the last checkpoint in the directory")
    parser.add_argument("--checkpoint-dir", default="artifacts/toy_modular/checkpoints")
    parser.add_argument("--output-ablation", default="artifacts/toy_modular/intervention_ablation.csv")
    parser.add_argument("--output-repair", default="artifacts/toy_modular/intervention_repair.csv")
    parser.add_argument("--summary", default="artifacts/toy_modular/intervention_decision_summary.md")
    parser.add_argument("--n-freqs", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--max-repair-examples", type=int, default=512)
    parser.add_argument(
        "--edit-radii",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Fractions of the latent-target edit to apply, giving the intervention-radius ablation.",
    )
    parser.add_argument(
        "--max-relative-edit-norm",
        type=float,
        default=1.0,
        help="Cap on ||delta|| / ||h_u||. Without a cap, rank-deficient germs produce unbounded edits.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    paths = sorted(Path(args.checkpoint_dir).glob("step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {args.checkpoint_dir}. Run run_07a first.")
    return paths[-1]


def build_subspaces(model, germ_W, train_acts, n_dims, seed, layer=0):
    """The cyclic subspace under test plus the three required control subspaces.

    Every data-dependent subspace is fit on training activations only, so the comparison is
    not contaminated by the held-out examples the interventions are evaluated on.
    """
    d_site = train_acts.shape[1]
    subspaces = {
        "cyclic": germ_subspace(germ_W),
        "random": random_subspace(d_site, n_dims, seed),
        "pca": pca_subspace(train_acts, n_dims),
    }
    spectral = spectral_subspace(model, layer, n_dims)
    if spectral.shape[0] == d_site:
        subspaces["spectral"] = spectral
    return subspaces


def main() -> None:
    import torch
    from tqdm.auto import tqdm

    args = parse_args()
    path = resolve_checkpoint(args)
    model, cfg, step = load_checkpoint(path, device=args.device)
    device = next(model.parameters()).device

    tokens, labels = make_dataset(cfg)
    (train_tokens, train_labels), (test_tokens, test_labels) = split_dataset(cfg, tokens, labels)
    train_tokens, train_labels = train_tokens.to(device), train_labels.to(device)
    test_tokens, test_labels = test_tokens.to(device), test_labels.to(device)

    freqs = key_frequencies(model, cfg.p, top_k=args.n_freqs)
    train_latent = latent_targets(train_tokens, cfg.p, freqs).cpu()
    test_latent = latent_targets(test_tokens, cfg.p, freqs).cpu()
    latent_dim = train_latent.shape[1]

    sites = make_toy_sites(cfg)
    edges = chain_edges(sites)
    train_acts = cache_site_activations(model, train_tokens, sites)
    test_acts = cache_site_activations(model, test_tokens, sites)
    germs = fit_latent_germs(train_acts, train_latent, sites, args.ridge)
    germ_train = germ_values(train_acts, germs, sites)
    transports = fit_transports(germ_train, edges, args.ridge)

    with torch.no_grad():
        base_logits = final_logits(model, test_tokens)
    base_pred = base_logits.argmax(dim=-1)
    base_correct = (base_pred == test_labels)
    base_incorrect_id = hardest_incorrect(base_logits, test_labels)
    base_margin = candidate_margin(base_logits, test_labels, base_incorrect_id)
    germ_test = germ_values(test_acts, germs, sites)
    base_obstruction = latent_obstruction(germ_test, transports, edges)["normalized"]

    print(f"checkpoint step {step}: test_acc={float(base_correct.float().mean()):.4f} freqs={freqs}")

    # ------------------------------------------------------------------
    # Necessity: ablate each subspace and measure the damage
    # ------------------------------------------------------------------
    ablation_rows: list[dict] = []
    for site in tqdm(sites, desc="ablation sites"):
        subspaces = build_subspaces(
            model, germs[site.site_id], train_acts[site.site_id], latent_dim, args.seed, layer=site.layer
        )
        for name, basis in subspaces.items():
            with torch.no_grad(), model.hooks(fwd_hooks=[(site.hook_name, site_ablate_hook(site, basis))]):
                logits = final_logits(model, test_tokens)
            pred = logits.argmax(dim=-1)
            margin = candidate_margin(logits, test_labels, base_incorrect_id)
            ablation_rows.append(
                {
                    "step": step,
                    **site.to_row(),
                    "subspace": name,
                    "subspace_dim": int(basis.shape[1]),
                    "test_acc_before": float(base_correct.float().mean()),
                    "test_acc_after": float((pred == test_labels).float().mean()),
                    "test_acc_drop": float((base_correct.float() - (pred == test_labels).float()).mean()),
                    "mean_margin_before": float(base_margin.mean()),
                    "mean_margin_after": float(margin.mean()),
                    "mean_kl": float(kl_divergence(base_logits, logits).mean()),
                }
            )
    write_csv(args.output_ablation, ablation_rows)

    # ------------------------------------------------------------------
    # Repair: steer the germ to the true latent state on failing examples
    # ------------------------------------------------------------------
    wrong_idx = (~base_correct).nonzero(as_tuple=True)[0][: args.max_repair_examples]
    right_idx = base_correct.nonzero(as_tuple=True)[0][: args.max_repair_examples]
    repair_rows: list[dict] = []

    for subset_name, idx in [("incorrect", wrong_idx), ("correct", right_idx)]:
        if idx.numel() == 0:
            continue
        subset_tokens = test_tokens[idx]
        subset_labels = test_labels[idx]
        subset_incorrect_id = base_incorrect_id[idx]
        subset_logits_before = base_logits[idx]
        subset_margin_before = base_margin[idx]
        subset_obstruction_before = base_obstruction[idx.cpu()]
        subset_latent = test_latent[idx.cpu()]

        for site in tqdm(sites, desc=f"repair sites ({subset_name})"):
            acts = test_acts[site.site_id][idx.cpu()]
            delta_full = germ_target_delta(acts, germs[site.site_id], subset_latent)
            # Cap the edit relative to the activation norm. An uncapped minimum-norm edit is
            # unbounded where the germ is rank-deficient, which leaves the local regime that
            # the response approximation is valid in.
            act_norm = acts.float().norm(dim=-1, keepdim=True)
            delta_norm = delta_full.float().norm(dim=-1, keepdim=True)
            cap = args.max_relative_edit_norm * act_norm
            scale = (cap / (delta_norm + 1e-12)).clamp(max=1.0)
            delta_capped = delta_full * scale
            frac_capped = float((scale < 1.0).float().mean())

            subspaces = build_subspaces(
                model, germs[site.site_id], train_acts[site.site_id], latent_dim, args.seed, layer=site.layer
            )
            phase_full = phase_rotate_delta(
                acts, germs[site.site_id], angle_rad=1.5707963267948966, n_freqs=len(freqs)
            )
            phase_norm = phase_full.float().norm(dim=-1, keepdim=True)
            phase_scale = (cap / (phase_norm + 1e-12)).clamp(max=1.0)
            phase_capped = phase_full * phase_scale

            for radius in args.edit_radii:
                base_edit = radius * delta_capped
                edits = {
                    "latent_target": base_edit,
                    "phase_rotate": radius * phase_capped,
                }
                for name, basis in subspaces.items():
                    if name == "cyclic":
                        continue
                    edits[f"matched_norm_{name}"] = match_norm_in_subspace(base_edit, basis)

                for edit_name, edit in edits.items():
                    with torch.no_grad(), model.hooks(
                        fwd_hooks=[(site.hook_name, site_edit_hook(site, edit))]
                    ):
                        all_logits, cache = model.run_with_cache(
                            subset_tokens,
                            names_filter=lambda n: n in {s.hook_name for s in sites},
                        )
                    logits = all_logits[:, -1, :]
                    after_acts = {s.site_id: extract_site(cache, s).detach().cpu() for s in sites}
                    germ_after = germ_values(after_acts, germs, sites)
                    obstruction_after = latent_obstruction(germ_after, transports, edges)["normalized"]
                    pred = logits.argmax(dim=-1)
                    margin_after = candidate_margin(logits, subset_labels, subset_incorrect_id)
                    repaired = (pred == subset_labels).float()

                    repair_rows.append(
                        {
                            "step": step,
                            "subset": subset_name,
                            **site.to_row(),
                            "edit": edit_name,
                            "edit_radius": radius,
                            "n_examples": int(idx.numel()),
                            "mean_edit_norm": float(edit.float().norm(dim=-1).mean()),
                            "mean_relative_edit_norm": float(
                                (edit.float().norm(dim=-1) / (act_norm.squeeze(-1) + 1e-12)).mean()
                            ),
                            "frac_edit_capped": frac_capped,
                            "mean_margin_before": float(subset_margin_before.mean()),
                            "mean_margin_after": float(margin_after.mean()),
                            "mean_delta_margin": float((margin_after - subset_margin_before).mean()),
                            "mean_obstruction_before": float(subset_obstruction_before.mean()),
                            "mean_obstruction_after": float(obstruction_after.mean()),
                            "mean_obstruction_reduction": float(
                                (subset_obstruction_before - obstruction_after).mean()
                            ),
                            "accuracy_after": float(repaired.mean()),
                            "repair_rate": float(repaired.mean()) if subset_name == "incorrect" else float("nan"),
                            "breakage_rate": float(1.0 - repaired.mean())
                            if subset_name == "correct"
                            else float("nan"),
                            "mean_kl": float(kl_divergence(subset_logits_before, logits).mean()),
                        }
                    )
            write_csv(args.output_repair, repair_rows)

    write_csv(args.output_repair, repair_rows)
    write_summary(args.summary, step, ablation_rows, repair_rows, float(base_correct.float().mean()))
    print(f"saved ablation results to {args.output_ablation}")
    print(f"saved repair results to {args.output_repair}")
    print(f"saved summary to {args.summary}")


CONTROL_SUBSPACES = ["random", "pca", "spectral"]
CONTROL_EDITS = ["matched_norm_random", "matched_norm_pca", "matched_norm_spectral", "phase_rotate"]


KL_FLOOR = 0.1  # below this the edit barely perturbs the model, so efficiency ratios are noise


def damage_per_kl(row: dict) -> float:
    """Accuracy drop per unit collateral KL, as a secondary efficiency diagnostic.

    Guarded by a KL floor: without it a no-op edit with near-zero KL produces an arbitrarily
    large ratio, which would reward doing nothing.
    """
    kl = float(row["mean_kl"])
    if kl < KL_FLOOR:
        return float("nan")
    return float(row["test_acc_drop"]) / kl


def within_site_ablation_wins(ablation_rows: list[dict]) -> tuple[int, int, list[dict]]:
    """Per site, does ablating the cyclic subspace damage more than every matched-dim control?

    The gate is raw accuracy drop at a fixed site. Within-site comparison already controls for
    the fact that some sites are intrinsically more destructive to perturb than others.
    """
    by_site: dict[str, list[dict]] = {}
    for row in ablation_rows:
        by_site.setdefault(row["site_id"], []).append(row)
    wins, total, detail = 0, 0, []
    for site_id, group in by_site.items():
        cyclic = next((row for row in group if row["subspace"] == "cyclic"), None)
        controls = [row for row in group if row["subspace"] in CONTROL_SUBSPACES]
        if not cyclic or not controls:
            continue
        total += 1
        best_control = max(controls, key=lambda row: float(row["test_acc_drop"]))
        won = float(cyclic["test_acc_drop"]) > float(best_control["test_acc_drop"])
        wins += int(won)
        detail.append(
            {
                "site_id": site_id,
                "cyclic_acc_drop": float(cyclic["test_acc_drop"]),
                "cyclic_kl": float(cyclic["mean_kl"]),
                "cyclic_damage_per_kl": damage_per_kl(cyclic),
                "best_control": best_control["subspace"],
                "best_control_acc_drop": float(best_control["test_acc_drop"]),
                "best_control_damage_per_kl": damage_per_kl(best_control),
                "cyclic_wins": won,
            }
        )
    return wins, total, sorted(detail, key=lambda row: row["cyclic_acc_drop"], reverse=True)


def within_cell_repair_wins(repair_rows: list[dict]) -> tuple[int, int, list[dict]]:
    """Per (site, radius), does the latent-target edit repair more than every matched-norm control?"""
    cells: dict[tuple[str, str], list[dict]] = {}
    for row in repair_rows:
        if row["subset"] != "incorrect":
            continue
        cells.setdefault((row["site_id"], str(row["edit_radius"])), []).append(row)
    wins, total, detail = 0, 0, []
    for (site_id, radius), group in cells.items():
        latent = next((row for row in group if row["edit"] == "latent_target"), None)
        controls = [row for row in group if row["edit"] in CONTROL_EDITS]
        if not latent or not controls:
            continue
        total += 1
        best_control = max(controls, key=lambda row: float(row["repair_rate"]))
        won = float(latent["repair_rate"]) > float(best_control["repair_rate"])
        wins += int(won)
        detail.append(
            {
                "site_id": site_id,
                "edit_radius": radius,
                "latent_repair_rate": float(latent["repair_rate"]),
                "latent_kl": float(latent["mean_kl"]),
                "latent_obstruction_reduction": float(latent["mean_obstruction_reduction"]),
                "best_control": best_control["edit"],
                "best_control_repair_rate": float(best_control["repair_rate"]),
                "latent_wins": won,
            }
        )
    return wins, total, sorted(detail, key=lambda row: row["latent_repair_rate"], reverse=True)


def write_summary(
    path: str | Path,
    step: int,
    ablation_rows: list[dict],
    repair_rows: list[dict],
    test_acc: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_incorrect = next(
        (int(row["n_examples"]) for row in repair_rows if row["subset"] == "incorrect"), 0
    )
    abl_wins, abl_total, abl_detail = within_site_ablation_wins(ablation_rows)
    rep_wins, rep_total, rep_detail = within_cell_repair_wins(repair_rows)

    lines = [
        "# Experiment 0B Causal Intervention Summary",
        "",
        f"- checkpoint step: {step}",
        f"- baseline test accuracy: {test_acc:.4f}",
        f"- incorrect examples available for repair: {n_incorrect}",
        "",
        "All comparisons are within-site: a subspace is compared only against matched-dimension",
        "controls at the same site, and an edit only against matched-norm edits at the same site",
        "and radius. Cross-site comparison would confound subspace choice with site choice.",
        "",
        "## Necessity: Cyclic vs Matched-Dimension Controls",
        "",
        f"Cyclic subspace causes the larger accuracy drop at {abl_wins}/{abl_total} sites.",
        "",
    ]
    for row in abl_detail:
        lines.append(
            f"- `{row['site_id']}`: cyclic drop {row['cyclic_acc_drop']:.4f} (KL {row['cyclic_kl']:.2f}, "
            f"damage/KL {row['cyclic_damage_per_kl']:.4f}) vs best control `{row['best_control']}` "
            f"drop {row['best_control_acc_drop']:.4f} (damage/KL {row['best_control_damage_per_kl']:.4f}) "
            f"-> {'win' if row['cyclic_wins'] else 'loss'}"
        )
    lines.extend(
        [
            "",
            "## Repair: Latent-Target Edit vs Matched-Norm Controls",
            "",
            f"Latent-target edit wins at {rep_wins}/{rep_total} (site, radius) cells.",
            "",
        ]
    )
    for row in rep_detail[:8]:
        lines.append(
            f"- `{row['site_id']}` r={row['edit_radius']}: repair {row['latent_repair_rate']:.4f} "
            f"(KL {row['latent_kl']:.3f}, obstruction reduction {row['latent_obstruction_reduction']:+.5f}) "
            f"vs best control `{row['best_control']}` {row['best_control_repair_rate']:.4f} "
            f"-> {'win' if row['latent_wins'] else 'loss'}"
        )

    necessity_ok = abl_total > 0 and abl_wins / abl_total > 0.5
    repair_ok = rep_total > 0 and rep_wins / rep_total > 0.5
    underpowered = n_incorrect < 50
    if necessity_ok and repair_ok:
        decision = "PASS_STRONG"
    elif necessity_ok or repair_ok:
        decision = "PASS_WEAK"
    else:
        decision = "NO_GO"
    if underpowered:
        decision += " (UNDERPOWERED)"

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- cyclic subspace wins the necessity test at a majority of sites: {necessity_ok}",
            f"- latent-target edit wins the repair test at a majority of cells: {repair_ok}",
            "",
            f"Decision: **{decision}**",
            "",
            "A probe that decodes the latent state is not sufficient. This decision requires the",
            "detected representation to be causally load-bearing relative to matched controls.",
        ]
    )
    if underpowered:
        lines.extend(
            [
                "",
                f"Warning: only {n_incorrect} incorrect examples were available, so repair rates are",
                "not statistically meaningful. Re-run against a partially grokked checkpoint, where the",
                "model still makes many errors, using `--checkpoint`.",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
