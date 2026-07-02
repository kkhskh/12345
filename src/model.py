from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def load_pretrained_model(
    model_name: str = "gpt2-small",
    device: str | None = None,
) -> "HookedTransformer":
    import torch
    from transformer_lens import HookedTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model


def load_gpt2_small(device: str | None = None) -> "HookedTransformer":
    return load_pretrained_model("gpt2-small", device=device)


def answer_token_id(model: "HookedTransformer", answer: str) -> int:
    toks = model.tokenizer.encode(answer, add_special_tokens=False)
    if len(toks) != 1:
        raise ValueError(f"Answer {answer!r} is not one token: {toks}")
    return toks[0]


def tokenize_prompts(model: "HookedTransformer", prompts: list[str]):
    return model.to_tokens(prompts, prepend_bos=True)


def logit_margin_from_logits(
    logits,
    answer_a_id: int,
    answer_b_id: int,
    token_pos: int = -1,
):
    """
    logits: [batch, pos, vocab]
    returns: [batch]
    """
    final_logits = logits[:, token_pos, :]
    return final_logits[:, answer_a_id] - final_logits[:, answer_b_id]

def run_logits(model, prompts: list[str]):
    import torch

    with torch.no_grad():
        tokens = tokenize_prompts(model, prompts)
        logits = model(tokens)
        return logits
