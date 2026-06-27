from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def resid_site_names(model: "HookedTransformer") -> list[str]:
    """
    Return L+1 residual-chain hook names:
      0,...,L-1: blocks.l.hook_resid_pre
      L:         blocks.{L-1}.hook_resid_post
    """
    n_layers = model.cfg.n_layers
    names = [f"blocks.{layer}.hook_resid_pre" for layer in range(n_layers)]
    names.append(f"blocks.{n_layers - 1}.hook_resid_post")
    return names


@torch.no_grad()
def run_with_resid_cache(
    model: "HookedTransformer",
    prompts: list[str],
    names_filter: list[str] | None = None,
):
    tokens = model.to_tokens(prompts, prepend_bos=True)
    if names_filter is None:
        names_filter = resid_site_names(model)

    logits, cache = model.run_with_cache(
        tokens,
        names_filter=names_filter,
        remove_batch_dim=False,
    )
    return tokens, logits, cache


def extract_resid_chain(cache, model: "HookedTransformer", token_pos: int = -1):
    """
    Return tensor h: [batch, L+1, d_model].
    """
    hs = []
    for name in resid_site_names(model):
        hs.append(cache[name][:, token_pos, :])
    return torch.stack(hs, dim=1)
