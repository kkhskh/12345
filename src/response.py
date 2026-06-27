from __future__ import annotations

import torch

from .cache import extract_resid_chain, resid_site_names, run_with_resid_cache
from .germs import scalar_germs, unembed_contrast_vector
from .model import logit_margin_from_logits


def margin_grads_wrt_resid_sites(
    model,
    prompt: str,
    answer_a_id: int,
    answer_b_id: int,
    token_pos: int = -1,
):
    """
    Compute grad_{h_l} m_c for all residual-chain sites on one prompt.

    Returns:
      grads: [L+1, d_model]
      logits
    """
    model.reset_hooks()
    site_names = resid_site_names(model)
    saved = {}

    def make_save_hook(name):
        def hook_fn(act, hook):
            act.retain_grad()
            saved[name] = act
            return act

        return hook_fn

    hooks = [(name, make_save_hook(name)) for name in site_names]
    tokens = model.to_tokens([prompt], prepend_bos=True)

    with torch.enable_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        margin = logit_margin_from_logits(
            logits,
            answer_a_id,
            answer_b_id,
            token_pos=token_pos,
        )[0]
        model.zero_grad(set_to_none=True)
        margin.backward()

    grads = []
    for name in site_names:
        grad = saved[name].grad[0, token_pos, :].detach()
        grads.append(grad)

    model.reset_hooks()
    return torch.stack(grads, dim=0), logits.detach()


def obstruction_grad_wrt_s(s: torch.Tensor, r, w):
    """
    Analytic gradient dO/ds_j for raw obstruction.

    O = sum_l w_l (r_l s_l - s_{l+1})^2

    Args:
      s: [L+1]
      r, w: [L]

    Returns:
      grad_s: [L+1]
    """
    r = torch.as_tensor(r, device=s.device, dtype=s.dtype)
    w = torch.as_tensor(w, device=s.device, dtype=s.dtype)

    residual = r * s[:-1] - s[1:]

    grad_s = torch.zeros_like(s)
    grad_s[:-1] += 2.0 * w * residual * r
    grad_s[1:] += -2.0 * w * residual

    return grad_s


def obstruction_grads_wrt_h(
    model,
    s: torch.Tensor,
    r,
    w,
    answer_a_id: int,
    answer_b_id: int,
):
    """
    Since s_l = h_l dot du, dO/dh_l = dO/ds_l * du.

    Args:
      s: [L+1]

    Returns:
      grad_h: [L+1, d_model]
    """
    du = unembed_contrast_vector(model, answer_a_id, answer_b_id).to(s.device)
    grad_s = obstruction_grad_wrt_s(s, r, w)
    return grad_s[:, None] * du[None, :]


@torch.no_grad()
def clean_corrupt_resid_chains(
    model,
    clean_prompt: str,
    corrupt_prompt: str,
    token_pos: int = -1,
):
    _, _, clean_cache = run_with_resid_cache(model, [clean_prompt])
    _, _, corrupt_cache = run_with_resid_cache(model, [corrupt_prompt])

    h_clean = extract_resid_chain(clean_cache, model, token_pos=token_pos)[0]
    h_corrupt = extract_resid_chain(corrupt_cache, model, token_pos=token_pos)[0]

    return h_clean, h_corrupt, clean_cache, corrupt_cache


def score_response_layers(
    model,
    example: dict,
    r,
    w,
    gamma: float = 0.0,
    token_pos: int = -1,
):
    """
    Score all residual-chain sites for one corrupt -> clean-target repair example.
    """
    answer_a_id = example["repair_answer_a_id"]
    answer_b_id = example["repair_answer_b_id"]

    h_clean, h_corrupt, _, corrupt_cache = clean_corrupt_resid_chains(
        model,
        example["clean_prompt"],
        example["corrupt_prompt"],
        token_pos=token_pos,
    )

    delta = h_clean - h_corrupt

    grad_m, _ = margin_grads_wrt_resid_sites(
        model,
        example["corrupt_prompt"],
        answer_a_id,
        answer_b_id,
        token_pos=token_pos,
    )

    s_corrupt = scalar_germs(
        model,
        corrupt_cache,
        answer_a_id,
        answer_b_id,
        token_pos=token_pos,
    )[0].detach()

    grad_o = obstruction_grads_wrt_h(
        model,
        s_corrupt,
        r,
        w,
        answer_a_id,
        answer_b_id,
    )

    pred_delta_m = (grad_m * delta).sum(dim=-1)
    pred_delta_o = (grad_o * delta).sum(dim=-1)
    cost = delta.pow(2).sum(dim=-1)

    score = -pred_delta_o + pred_delta_m - gamma * cost

    rows = []
    for layer in range(score.shape[0]):
        rows.append(
            {
                "id": example["id"],
                "layer": layer,
                "pred_delta_m": float(pred_delta_m[layer].detach().cpu()),
                "pred_delta_O": float(pred_delta_o[layer].detach().cpu()),
                "cost": float(cost[layer].detach().cpu()),
                "score": float(score[layer].detach().cpu()),
            }
        )

    return rows
