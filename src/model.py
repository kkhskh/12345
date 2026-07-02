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


def answer_token_ids(model: "HookedTransformer", answer: str) -> list[int]:
    """Return tokenizer ids for an answer continuation.

    Factual-conflict experiments use answers as continuations, including the
    leading space. Multi-token answers are valid for sequence scoring.
    """

    toks = model.tokenizer.encode(answer, add_special_tokens=False)
    if not toks:
        raise ValueError(f"Answer {answer!r} tokenized to an empty sequence")
    return list(toks)


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


def continuation_logprob(
    model: "HookedTransformer",
    prompt: str,
    answer_token_ids_: list[int],
) -> float:
    """Return log p(answer | prompt) for a tokenized answer continuation."""

    import torch

    prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = prompt_ids + list(answer_token_ids_)
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("answer_token_ids_ must contain at least one token")

    device = getattr(getattr(model, "cfg", None), "device", None)
    bos_id = getattr(model.tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(model.tokenizer, "eos_token_id", None)
    if bos_id is None:
        raise ValueError("Tokenizer has no BOS or EOS token id for manual scoring")
    tokens = torch.tensor([[bos_id] + full_ids], device=device)

    with torch.no_grad():
        logits = model(tokens)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        total = 0.0
        for offset, answer_id in enumerate(answer_token_ids_):
            logit_pos = len(prompt_ids) + offset
            total += float(log_probs[logit_pos, answer_id].detach().cpu())
    return total


def continuation_logprob_margin(
    model: "HookedTransformer",
    prompt: str,
    answer_a_token_ids: list[int],
    answer_b_token_ids: list[int],
) -> float:
    """Return log p(answer_a | prompt) - log p(answer_b | prompt)."""

    return continuation_logprob(model, prompt, answer_a_token_ids) - continuation_logprob(
        model,
        prompt,
        answer_b_token_ids,
    )
