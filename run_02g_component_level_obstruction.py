from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.cache import run_with_resid_cache
from src.data_ioi import read_jsonl
from src.germs import unembed_contrast_vector
from src.model import load_gpt2_small


TEST_PATH = Path("data/processed/ioi_test.jsonl")
SCORES_PATH = Path("artifacts/results/component_level_scores.csv")
ORACLE_PATH = Path("artifacts/results/component_level_patching_oracle.csv")
SUMMARY_PATH = Path("artifacts/results/component_level_metric_summary.csv")
SCATTER_PATH = Path("artifacts/figures/component_score_vs_observed_patch_delta_margin.png")
REPAIR_PATH = Path("artifacts/figures/component_repair_at_k_vs_baselines.png")

POSITION_NAMES = ["first_name", "second_name", "giver_name", "final_token"]
DEFAULT_MAX_EXAMPLES = 100
SEED = 12345


@dataclass(frozen=True)
class Site:
    kind: str
    layer: int
    position_name: str
    head: int | None = None

    @property
    def id(self) -> str:
        head = "none" if self.head is None else str(self.head)
        return f"{self.kind}:L{self.layer}:H{head}:{self.position_name}"

    @property
    def hook_name(self) -> str:
        if self.kind == "head":
            return f"blocks.{self.layer}.attn.hook_z"
        if self.kind == "mlp":
            return f"blocks.{self.layer}.hook_mlp_out"
        if self.kind == "resid_pre":
            return f"blocks.{self.layer}.hook_resid_pre"
        raise ValueError(f"Unknown site kind: {self.kind}")


def token_len(model, prompt: str) -> int:
    return len(model.tokenizer.encode(prompt, add_special_tokens=False)) + 1


def find_positions(model, tokens: torch.Tensor, ex: dict, prompt: str, condition: str):
    final_pos = token_len(model, prompt) - 1
    ids = tokens[: final_pos + 1].tolist()

    name_a_id = ex["clean_competitor_id"]
    name_b_id = ex["clean_correct_id"]
    name_a_positions = [idx for idx, tok in enumerate(ids) if tok == name_a_id]
    name_b_positions = [idx for idx, tok in enumerate(ids) if tok == name_b_id]

    if not name_a_positions or not name_b_positions:
        raise ValueError(
            f"Could not find both names in prompt for {ex['id']} / {condition}: "
            f"{prompt!r}"
        )

    giver_positions = (
        name_b_positions if condition == "swapped_coherent" else name_a_positions
    )
    giver_pos = giver_positions[1] if len(giver_positions) > 1 else giver_positions[0]

    return {
        "first_name": name_a_positions[0],
        "second_name": name_b_positions[0],
        "giver_name": giver_pos,
        "final_token": final_pos,
    }


def rowwise_margin(logits, examples, positions):
    vals = []
    for i, ex in enumerate(examples):
        final_pos = positions[i]["final_token"]
        answer_a_id = ex["repair_answer_a_id"]
        answer_b_id = ex["repair_answer_b_id"]
        vals.append(logits[i, final_pos, answer_a_id] - logits[i, final_pos, answer_b_id])
    return torch.stack(vals)


def build_sites(model, include_resid_controls: bool = True):
    sites = []
    for layer in range(model.cfg.n_layers):
        for position_name in POSITION_NAMES:
            for head in range(model.cfg.n_heads):
                sites.append(Site("head", layer, position_name, head=head))
            sites.append(Site("mlp", layer, position_name))
            if include_resid_controls:
                sites.append(Site("resid_pre", layer, position_name))
    return sites


def names_filter(model):
    names = []
    for layer in range(model.cfg.n_layers):
        names.append(f"blocks.{layer}.attn.hook_z")
        names.append(f"blocks.{layer}.hook_mlp_out")
        names.append(f"blocks.{layer}.hook_resid_pre")
    return names


def site_activation(cache, site: Site, row_idx: int, token_pos: int):
    act = cache[site.hook_name]
    if site.kind == "head":
        return act[row_idx, token_pos, site.head, :]
    return act[row_idx, token_pos, :]


def component_to_resid(model, site: Site, value: torch.Tensor):
    if site.kind == "head":
        return value @ model.W_O[site.layer, site.head]
    return value


def component_contribution_scalar(model, site: Site, value: torch.Tensor, du: torch.Tensor):
    resid_value = component_to_resid(model, site, value.to(du.device))
    return resid_value @ du


def compute_component_scores_for_example(model, ex: dict, site_list: list[Site]):
    clean_prompt = ex["clean_prompt"]
    swapped_prompt = ex["swapped_coherent_prompt"]
    cache_names = names_filter(model)

    clean_tokens, _, clean_cache = run_with_resid_cache(model, [clean_prompt], cache_names)
    swapped_tokens = model.to_tokens([swapped_prompt], prepend_bos=True)
    clean_positions = find_positions(
        model,
        clean_tokens[0],
        ex,
        clean_prompt,
        "clean_coherent",
    )
    swapped_positions = find_positions(
        model,
        swapped_tokens[0],
        ex,
        swapped_prompt,
        "swapped_coherent",
    )
    du = unembed_contrast_vector(
        model,
        ex["repair_answer_a_id"],
        ex["repair_answer_b_id"],
    )

    saved = {}

    def make_save_hook(name):
        def hook_fn(act, hook):
            act.retain_grad()
            saved[name] = act
            return act

        return hook_fn

    hooks = [(name, make_save_hook(name)) for name in cache_names]
    with torch.enable_grad():
        logits = model.run_with_hooks(swapped_tokens, fwd_hooks=hooks)
        margin = (
            logits[0, swapped_positions["final_token"], ex["repair_answer_a_id"]]
            - logits[0, swapped_positions["final_token"], ex["repair_answer_b_id"]]
        )
        model.zero_grad(set_to_none=True)
        margin.backward()

    rows = []
    for site in site_list:
        clean_pos = clean_positions[site.position_name]
        swapped_pos = swapped_positions[site.position_name]
        clean_value = site_activation(clean_cache, site, 0, clean_pos).detach()
        swapped_value = saved[site.hook_name][0, swapped_pos]
        grad = saved[site.hook_name].grad[0, swapped_pos]
        if site.kind == "head":
            swapped_value = swapped_value[site.head, :]
            grad = grad[site.head, :]

        delta = clean_value.to(grad.device) - swapped_value.detach()
        pred_delta_m = (grad.detach() * delta).sum()

        clean_contrib = component_contribution_scalar(model, site, clean_value, du)
        swapped_contrib = component_contribution_scalar(
            model,
            site,
            swapped_value.detach(),
            du,
        )
        # Local candidate-specific incompatibility proxy: patching removes the
        # clean-vs-swapped scalar contribution mismatch at this component.
        pred_delta_o = -((clean_contrib - swapped_contrib) ** 2)
        cost = delta.pow(2).sum()
        score = -pred_delta_o + pred_delta_m - 0.0 * cost

        row = {
            **asdict(site),
            "site_id": site.id,
            "id": ex["id"],
            "pred_delta_m": float(pred_delta_m.cpu()),
            "pred_delta_O": float(pred_delta_o.detach().cpu()),
            "cost": float(cost.cpu()),
            "score": float(score.detach().cpu()),
            "activation_delta_norm": float(cost.sqrt().cpu()),
            "gradient_saliency": float(grad.detach().pow(2).sum().sqrt().cpu()),
            "clean_contribution": float(clean_contrib.cpu()),
            "swapped_contribution": float(swapped_contrib.detach().cpu()),
        }
        rows.append(row)

    model.reset_hooks()
    return rows


@torch.no_grad()
def patch_site_batch(model, batch, site: Site, cache_names):
    clean_prompts = [ex["clean_prompt"] for ex in batch]
    swapped_prompts = [ex["swapped_coherent_prompt"] for ex in batch]

    clean_tokens, _, clean_cache = run_with_resid_cache(model, clean_prompts, cache_names)
    swapped_tokens, swapped_logits, _ = run_with_resid_cache(
        model,
        swapped_prompts,
        cache_names,
    )

    clean_positions = [
        find_positions(model, clean_tokens[i], ex, ex["clean_prompt"], "clean_coherent")
        for i, ex in enumerate(batch)
    ]
    swapped_positions = [
        find_positions(
            model,
            swapped_tokens[i],
            ex,
            ex["swapped_coherent_prompt"],
            "swapped_coherent",
        )
        for i, ex in enumerate(batch)
    ]
    baseline_margin = rowwise_margin(swapped_logits, batch, swapped_positions)

    clean_values = []
    for i in range(len(batch)):
        clean_pos = clean_positions[i][site.position_name]
        clean_values.append(site_activation(clean_cache, site, i, clean_pos).detach())

    def patch_hook(act, hook):
        patched = act.clone()
        for i, clean_value in enumerate(clean_values):
            pos = swapped_positions[i][site.position_name]
            if site.kind == "head":
                patched[i, pos, site.head, :] = clean_value.to(
                    patched.device,
                    dtype=patched.dtype,
                )
            else:
                patched[i, pos, :] = clean_value.to(patched.device, dtype=patched.dtype)
        return patched

    patched_logits = model.run_with_hooks(swapped_tokens, fwd_hooks=[(site.hook_name, patch_hook)])
    patched_margin = rowwise_margin(patched_logits, batch, swapped_positions)

    rows = []
    for i, ex in enumerate(batch):
        delta_margin = patched_margin[i] - baseline_margin[i]
        rows.append(
            {
                **asdict(site),
                "site_id": site.id,
                "id": ex["id"],
                "margin_swapped": float(baseline_margin[i].cpu()),
                "margin_patched": float(patched_margin[i].cpu()),
                "observed_delta_margin": float(delta_margin.cpu()),
                "repair_success_margin_increase": bool(delta_margin.cpu() > 0),
                "repair_success_flip": bool(patched_margin[i].cpu() > 0),
            }
        )
    return rows


def aggregate_scores(score_df: pd.DataFrame, oracle_df: pd.DataFrame):
    score_site = (
        score_df.groupby(["site_id", "kind", "layer", "head", "position_name"], dropna=False)
        .agg(
            pred_delta_m=("pred_delta_m", "mean"),
            pred_delta_O=("pred_delta_O", "mean"),
            cost=("cost", "mean"),
            score=("score", "mean"),
            activation_delta_norm=("activation_delta_norm", "mean"),
            gradient_saliency=("gradient_saliency", "mean"),
        )
        .reset_index()
    )
    oracle_site = (
        oracle_df.groupby(["site_id", "kind", "layer", "head", "position_name"], dropna=False)
        .agg(
            observed_delta_margin=("observed_delta_margin", "mean"),
            repair_success_flip=("repair_success_flip", "mean"),
            repair_success_margin_increase=("repair_success_margin_increase", "mean"),
        )
        .reset_index()
    )
    return score_site.merge(
        oracle_site,
        on=["site_id", "kind", "layer", "head", "position_name"],
    )


def safe_spearman(x, y):
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def effective_site_auc(df: pd.DataFrame, score_col: str):
    threshold = df["observed_delta_margin"].quantile(0.90)
    y = (df["observed_delta_margin"] >= threshold).astype(int)
    if y.nunique() < 2:
        return float("nan")
    return roc_auc_score(y, df[score_col])


def repair_at_k(df: pd.DataFrame, method_col: str, ks=(1, 5, 10, 20, 50)):
    rows = []
    ranked = df.sort_values(method_col, ascending=False)
    for k in ks:
        sub = ranked.head(k)
        rows.append(
            {
                "method": method_col,
                "k": k,
                "mean_observed_delta_margin": sub["observed_delta_margin"].mean(),
                "repair_success_flip": sub["repair_success_flip"].mean(),
                "repair_success_margin_increase": sub[
                    "repair_success_margin_increase"
                ].mean(),
            }
        )
    return rows


def make_figures(site_df: pd.DataFrame, repair_df: pd.DataFrame):
    SCATTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.8, 4.2))
    plt.scatter(site_df["score"], site_df["observed_delta_margin"], s=12, alpha=0.55)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("component obstruction-response score")
    plt.ylabel("observed patch Δmargin")
    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=200)

    plt.figure(figsize=(6, 4.2))
    for method, sub in repair_df.groupby("method"):
        sub = sub.sort_values("k")
        plt.plot(sub["k"], sub["repair_success_flip"], marker="o", label=method)
    plt.xlabel("top-k single component sites")
    plt.ylabel("mean single-site repair flip rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPAIR_PATH, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument("--include-resid-controls", action="store_true")
    args = parser.parse_args()

    rng = random.Random(SEED)
    examples = read_jsonl(TEST_PATH)[: args.max_examples]

    print("Loading GPT-2 small...", flush=True)
    model = load_gpt2_small()
    site_list = build_sites(model, include_resid_controls=args.include_resid_controls)
    if args.max_sites is not None:
        site_list = site_list[: args.max_sites]

    print(
        f"Running component oracle on {len(examples)} examples and {len(site_list)} sites",
        flush=True,
    )
    cache_names = names_filter(model)

    score_rows = []
    for ex in tqdm(examples, desc="component scores"):
        score_rows.extend(compute_component_scores_for_example(model, ex, site_list))

    oracle_rows = []
    for site in tqdm(site_list, desc="patch oracle sites"):
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            oracle_rows.extend(patch_site_batch(model, batch, site, cache_names))

    score_df = pd.DataFrame(score_rows)
    oracle_df = pd.DataFrame(oracle_rows)
    site_df = aggregate_scores(score_df, oracle_df)
    site_df["random"] = [rng.random() for _ in range(len(site_df))]

    rho_score, p_score = safe_spearman(site_df["score"], site_df["observed_delta_margin"])
    rho_attr, p_attr = safe_spearman(
        site_df["pred_delta_m"],
        site_df["observed_delta_margin"],
    )

    metric_rows = [
        {
            "metric": "spearman_score_observed_delta_margin",
            "value": rho_score,
            "p": p_score,
        },
        {
            "metric": "spearman_pred_delta_m_observed_delta_margin",
            "value": rho_attr,
            "p": p_attr,
        },
        {"metric": "auc_score_detects_top_causal_components", "value": effective_site_auc(site_df, "score")},
        {
            "metric": "auc_attribution_detects_top_causal_components",
            "value": effective_site_auc(site_df, "pred_delta_m"),
        },
        {
            "metric": "auc_activation_norm_detects_top_causal_components",
            "value": effective_site_auc(site_df, "activation_delta_norm"),
        },
        {
            "metric": "auc_gradient_saliency_detects_top_causal_components",
            "value": effective_site_auc(site_df, "gradient_saliency"),
        },
        {"metric": "auc_random_detects_top_causal_components", "value": effective_site_auc(site_df, "random")},
    ]

    repair_rows = []
    for method_col in [
        "score",
        "pred_delta_m",
        "activation_delta_norm",
        "gradient_saliency",
        "random",
    ]:
        repair_rows.extend(repair_at_k(site_df, method_col))
    repair_df = pd.DataFrame(repair_rows)

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(SCORES_PATH, index=False)
    oracle_df.to_csv(ORACLE_PATH, index=False)
    pd.concat([pd.DataFrame(metric_rows), repair_df], ignore_index=True).to_csv(
        SUMMARY_PATH,
        index=False,
    )
    make_figures(site_df, repair_df)

    print("Component-level metric summary")
    print(pd.DataFrame(metric_rows))
    print(repair_df)
    print(f"Saved scores to {SCORES_PATH}")
    print(f"Saved patching oracle to {ORACLE_PATH}")
    print(f"Saved metric summary to {SUMMARY_PATH}")
    print(f"Saved figures to {SCATTER_PATH} and {REPAIR_PATH}")

    obstruction_beats_attr = (
        effective_site_auc(site_df, "score") > effective_site_auc(site_df, "pred_delta_m")
        or rho_score > rho_attr
    )
    if rho_score > 0.30 and obstruction_beats_attr:
        print("GO: component obstruction-response passes initial ranking criteria.")
    else:
        print("NO-GO: component score does not beat attribution cleanly yet.")


if __name__ == "__main__":
    main()
