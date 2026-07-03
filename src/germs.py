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


def _bos_token_id(model: "HookedTransformer") -> int:
    bos_id = getattr(model.tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(model.tokenizer, "eos_token_id", None)
    if bos_id is None:
        raise ValueError("Tokenizer has no BOS or EOS token id")
    return int(bos_id)


def _tokens_for_prompt_prefix(
    model: "HookedTransformer",
    prompt: str,
    prefix_token_ids: list[int],
) -> tuple[torch.Tensor, int]:
    prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    token_ids = [_bos_token_id(model)] + prompt_ids + list(prefix_token_ids)
    tokens = torch.tensor([token_ids], device=model.W_U.device)
    prediction_pos = len(prompt_ids) + len(prefix_token_ids)
    return tokens, prediction_pos


@torch.no_grad()
def answer_sequence_germs(
    model: "HookedTransformer",
    prompt: str,
    answer_token_ids: list[int],
    reduction: str = "mean",
) -> torch.Tensor:
    """Return residual-chain evidence for a full answer continuation.

    For answer token k, the residual site is evaluated at the position whose
    logits predict that token: prompt plus the answer prefix before k. This
    gives later answer tokens their own candidate-prefix context instead of
    reusing only the first-token prompt state.
    """

    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported sequence germ reduction: {reduction}")
    if not answer_token_ids:
        raise ValueError("answer_token_ids must be non-empty")

    per_token_vals = []
    names = resid_site_names(model)
    for offset, token_id in enumerate(answer_token_ids):
        tokens, prediction_pos = _tokens_for_prompt_prefix(
            model,
            prompt,
            answer_token_ids[:offset],
        )
        _, cache = model.run_with_cache(
            tokens,
            names_filter=names,
            remove_batch_dim=False,
        )
        du = model.W_U[:, int(token_id)]
        vals = []
        for name in names:
            h = cache[name][:, prediction_pos, :]
            vals.append((h @ du)[0])
        per_token_vals.append(torch.stack(vals, dim=0))

    stacked = torch.stack(per_token_vals, dim=0)
    if reduction == "sum":
        return stacked.sum(dim=0)
    return stacked.mean(dim=0)


@torch.no_grad()
def sequence_contrast_germs(
    model: "HookedTransformer",
    prompt: str,
    answer_a_token_ids: list[int],
    answer_b_token_ids: list[int],
    reduction: str = "mean",
) -> torch.Tensor:
    """Return s: [L+1] for a multi-token candidate contrast."""

    evidence_a = answer_sequence_germs(
        model,
        prompt,
        [int(token_id) for token_id in answer_a_token_ids],
        reduction=reduction,
    )
    evidence_b = answer_sequence_germs(
        model,
        prompt,
        [int(token_id) for token_id in answer_b_token_ids],
        reduction=reduction,
    )
    return evidence_a - evidence_b


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
