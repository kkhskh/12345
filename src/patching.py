from __future__ import annotations

import torch
import torch.nn.functional as F

from .cache import resid_site_names, run_with_resid_cache
from .germs import scalar_germs
from .model import logit_margin_from_logits
from .obstruction import compute_obstruction


def get_clean_site_values(model, clean_prompt: str, token_pos: int = -1):
    _, _, clean_cache = run_with_resid_cache(model, [clean_prompt])
    values = {}
    for layer, name in enumerate(resid_site_names(model)):
        values[layer] = clean_cache[name][:, token_pos, :].detach()
    return values


def patch_resid_layers(
    model,
    corrupt_prompt: str,
    clean_site_values: dict[int, torch.Tensor],
    layers_to_patch: list[int],
    token_pos: int = -1,
    cache_after_patch: bool = True,
):
    site_names = resid_site_names(model)

    def make_patch_hook(layer: int):
        clean_value = clean_site_values[layer]

        def hook_fn(act, hook):
            patched = act.clone()
            patched[:, token_pos, :] = clean_value.to(act.device, dtype=act.dtype)
            return patched

        return hook_fn

    hooks = [(site_names[layer], make_patch_hook(layer)) for layer in layers_to_patch]
    tokens = model.to_tokens([corrupt_prompt], prepend_bos=True)

    if cache_after_patch:
        with model.hooks(fwd_hooks=hooks):
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=site_names,
                remove_batch_dim=False,
            )
        return logits, cache

    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return logits, None


@torch.no_grad()
def off_target_kl(corrupt_logits, patched_logits, token_pos: int = -1):
    """
    KL(p_corrupt || p_patched) over full next-token distribution.
    """
    log_p_patched = patched_logits[:, token_pos, :].log_softmax(dim=-1)
    p_corrupt = corrupt_logits[:, token_pos, :].softmax(dim=-1)

    return F.kl_div(
        log_p_patched,
        p_corrupt,
        reduction="batchmean",
    )


@torch.no_grad()
def evaluate_patch(
    model,
    example: dict,
    r,
    w,
    layers_to_patch: list[int],
    token_pos: int = -1,
):
    answer_a_id = example["repair_answer_a_id"]
    answer_b_id = example["repair_answer_b_id"]

    clean_values = get_clean_site_values(
        model,
        example["clean_prompt"],
        token_pos=token_pos,
    )

    _, corrupt_logits, corrupt_cache = run_with_resid_cache(
        model,
        [example["corrupt_prompt"]],
    )

    corrupt_margin = logit_margin_from_logits(
        corrupt_logits,
        answer_a_id,
        answer_b_id,
        token_pos=token_pos,
    )[0]

    s_corrupt = scalar_germs(
        model,
        corrupt_cache,
        answer_a_id,
        answer_b_id,
        token_pos=token_pos,
    )
    o_corrupt = compute_obstruction(s_corrupt, r, w)["normalized"][0]

    patched_logits, patched_cache = patch_resid_layers(
        model,
        example["corrupt_prompt"],
        clean_values,
        layers_to_patch,
        token_pos=token_pos,
        cache_after_patch=True,
    )

    patched_margin = logit_margin_from_logits(
        patched_logits,
        answer_a_id,
        answer_b_id,
        token_pos=token_pos,
    )[0]

    s_patched = scalar_germs(
        model,
        patched_cache,
        answer_a_id,
        answer_b_id,
        token_pos=token_pos,
    )
    o_patched = compute_obstruction(s_patched, r, w)["normalized"][0]

    kl = off_target_kl(corrupt_logits, patched_logits, token_pos=token_pos)

    return {
        "id": example["id"],
        "layers": tuple(layers_to_patch),
        "k": len(layers_to_patch),
        "margin_corrupt": float(corrupt_margin.cpu()),
        "margin_patched": float(patched_margin.cpu()),
        "delta_margin": float((patched_margin - corrupt_margin).cpu()),
        "O_corrupt": float(o_corrupt.cpu()),
        "O_patched": float(o_patched.cpu()),
        "delta_O": float((o_patched - o_corrupt).cpu()),
        "repair_success_margin_increase": bool((patched_margin > corrupt_margin).cpu()),
        "repair_success_flip": bool((patched_margin > 0).cpu()),
        "off_target_kl_corrupt_to_patched": float(kl.cpu()),
    }


def run_single_layer_sweep(model, example, r, w):
    l_plus_1 = model.cfg.n_layers + 1
    return [evaluate_patch(model, example, r, w, [layer]) for layer in range(l_plus_1)]


def run_topk_sweep(model, example, r, w, ranked_layers: list[int], ks=(1, 2, 3, 5)):
    rows = []
    for k in ks:
        rows.append(
            evaluate_patch(
                model,
                example,
                r,
                w,
                ranked_layers[:k],
            )
        )
    return rows
