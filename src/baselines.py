from __future__ import annotations

import random

import torch

from .response import clean_corrupt_resid_chains, margin_grads_wrt_resid_sites


def rank_random(model, seed: int = 0):
    rng = random.Random(seed)
    layers = list(range(model.cfg.n_layers + 1))
    rng.shuffle(layers)
    return layers


@torch.no_grad()
def rank_activation_delta_norm(model, example):
    h_clean, h_corrupt, _, _ = clean_corrupt_resid_chains(
        model,
        example["clean_prompt"],
        example["corrupt_prompt"],
    )
    delta = h_clean - h_corrupt
    scores = delta.pow(2).sum(dim=-1).sqrt()
    return torch.argsort(scores, descending=True).tolist(), scores.cpu().tolist()


def rank_gradient_saliency(model, example):
    """
    Saliency baseline independent of clean patch direction: ||grad_h m||_2.
    """
    grads, _ = margin_grads_wrt_resid_sites(
        model,
        example["corrupt_prompt"],
        example["repair_answer_a_id"],
        example["repair_answer_b_id"],
    )
    scores = grads.pow(2).sum(dim=-1).sqrt()
    return torch.argsort(scores, descending=True).tolist(), scores.detach().cpu().tolist()


def rank_attribution_patching(model, example):
    """
    Required baseline: grad_h m_c^T (h_clean - h_corrupt).
    """
    h_clean, h_corrupt, _, _ = clean_corrupt_resid_chains(
        model,
        example["clean_prompt"],
        example["corrupt_prompt"],
    )
    delta = h_clean - h_corrupt

    grads, _ = margin_grads_wrt_resid_sites(
        model,
        example["corrupt_prompt"],
        example["repair_answer_a_id"],
        example["repair_answer_b_id"],
    )

    scores = (grads * delta).sum(dim=-1)
    return torch.argsort(scores, descending=True).tolist(), scores.detach().cpu().tolist()


def rank_obstruction_response(score_rows):
    rows = sorted(score_rows, key=lambda row: row["score"], reverse=True)
    return [row["layer"] for row in rows], [row["score"] for row in rows]


def activation_patching_oracle_ranking(single_layer_sweep_rows):
    """
    Oracle upper bound. Not a deployable method; use only as ceiling.
    """
    rows = sorted(
        single_layer_sweep_rows,
        key=lambda row: row["delta_margin"],
        reverse=True,
    )
    return [row["layers"][0] for row in rows], [row["delta_margin"] for row in rows]
