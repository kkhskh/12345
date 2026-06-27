from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .cache import resid_site_names

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def unembed_contrast_vector(
    model: "HookedTransformer",
    answer_a_id: int,
    answer_b_id: int,
):
    """
    W_U shape in TransformerLens GPT-style models is [d_model, d_vocab].
    """
    return model.W_U[:, answer_a_id] - model.W_U[:, answer_b_id]


def scalar_germs(
    model: "HookedTransformer",
    cache,
    answer_a_id: int,
    answer_b_id: int,
    token_pos: int = -1,
) -> torch.Tensor:
    """
    Return s: [batch, L+1].
    """
    du = unembed_contrast_vector(model, answer_a_id, answer_b_id)
    vals = []

    for name in resid_site_names(model):
        h = cache[name][:, token_pos, :]
        vals.append(h @ du)

    return torch.stack(vals, dim=1)
