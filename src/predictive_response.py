from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.low_rank_germs import final_token_pos, safe_model_name


@dataclass(frozen=True)
class ComponentSite:
    site_id: str
    site_type: str
    layer: int
    head: int | None
    hook_name: str

    def to_row(self) -> dict:
        return asdict(self)


def make_component_sites(
    model: Any,
    include_attn: bool = True,
    include_mlp: bool = True,
    include_resid: bool = False,
    max_sites: int | None = None,
) -> list[ComponentSite]:
    sites: list[ComponentSite] = []
    for layer in range(model.cfg.n_layers):
        if include_attn:
            for head in range(model.cfg.n_heads):
                sites.append(
                    ComponentSite(
                        site_id=f"head:L{layer}:H{head}:final_token",
                        site_type="head",
                        layer=layer,
                        head=head,
                        hook_name=f"blocks.{layer}.attn.hook_z",
                    )
                )
        if include_mlp:
            sites.append(
                ComponentSite(
                    site_id=f"mlp:L{layer}:Hnone:final_token",
                    site_type="mlp",
                    layer=layer,
                    head=None,
                    hook_name=f"blocks.{layer}.hook_mlp_out",
                )
            )
        if include_resid:
            sites.append(
                ComponentSite(
                    site_id=f"resid_pre:L{layer}:Hnone:final_token",
                    site_type="resid_pre",
                    layer=layer,
                    head=None,
                    hook_name=f"blocks.{layer}.hook_resid_pre",
                )
            )
    return sites[:max_sites] if max_sites is not None else sites


def _bos_token_id(model: Any) -> int:
    bos_id = getattr(model.tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(model.tokenizer, "eos_token_id", None)
    if bos_id is None:
        raise ValueError("Tokenizer has no BOS or EOS token id")
    return int(bos_id)


def _tokens_for_prompt_answer(model: Any, prompt: str, answer: str):
    import torch

    prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = model.tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError(f"Answer {answer!r} tokenized to an empty sequence")
    device = getattr(getattr(model, "cfg", None), "device", None)
    tokens = torch.tensor([[_bos_token_id(model)] + prompt_ids + answer_ids], device=device)
    return tokens, len(prompt_ids), [int(tok) for tok in answer_ids]


def _site_slice(act: Any, site: ComponentSite, row_idx: int, token_pos: int):
    if site.site_type == "head":
        return act[row_idx, token_pos, site.head, :]
    return act[row_idx, token_pos, :]


def cache_component_activations(model: Any, prompt: str, sites: list[ComponentSite], token_pos: int = -1):
    import torch

    if token_pos == -1:
        token_pos = final_token_pos(model, prompt)
    names = sorted({site.hook_name for site in sites})
    tokens = model.to_tokens([prompt], prepend_bos=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names, remove_batch_dim=False)
    return {
        site.site_id: _site_slice(cache[site.hook_name], site, 0, token_pos).detach().cpu()
        for site in sites
    }


def make_patch_direction(clean_cache: dict[str, Any], conflict_cache: dict[str, Any], site: ComponentSite):
    return clean_cache[site.site_id].float() - conflict_cache[site.site_id].float()


def _patch_hook(site: ComponentSite, clean_value: Any, token_pos: int):
    def hook_fn(act, hook):
        patched = act.clone()
        value = clean_value.to(patched.device, dtype=patched.dtype)
        if site.site_type == "head":
            patched[0, token_pos, site.head, :] = value
        else:
            patched[0, token_pos, :] = value
        return patched

    return hook_fn


def patch_component_site(model: Any, conflict_prompt: str, site: ComponentSite, clean_value: Any, token_pos: int = -1):
    from src.cache import resid_site_names

    if token_pos == -1:
        token_pos = final_token_pos(model, conflict_prompt)
    tokens = model.to_tokens([conflict_prompt], prepend_bos=True)
    hooks = [(site.hook_name, _patch_hook(site, clean_value, token_pos))]
    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(tokens, names_filter=resid_site_names(model), remove_batch_dim=False)
    return logits, cache


def _continuation_logprob_from_logits(logits: Any, prompt_len: int, answer_ids: list[int]):
    import torch

    log_probs = torch.log_softmax(logits[0], dim=-1)
    total = 0.0
    for offset, answer_id in enumerate(answer_ids):
        total = total + log_probs[prompt_len + offset, answer_id]
    return total


def multi_token_margin_from_logits_or_model(
    model: Any,
    prompt: str,
    true_answer: str,
    false_answer: str,
    logits: Any | None = None,
):
    if logits is None:
        true_tokens, prompt_len, true_ids = _tokens_for_prompt_answer(model, prompt, true_answer)
        false_tokens, _, false_ids = _tokens_for_prompt_answer(model, prompt, false_answer)
        logits_true = model(true_tokens)
        logits_false = model(false_tokens)
    else:
        _, prompt_len, true_ids = _tokens_for_prompt_answer(model, prompt, true_answer)
        _, _, false_ids = _tokens_for_prompt_answer(model, prompt, false_answer)
        logits_true = logits_false = logits
    return _continuation_logprob_from_logits(logits_true, prompt_len, true_ids) - _continuation_logprob_from_logits(
        logits_false, prompt_len, false_ids
    )


def load_low_rank_artifacts(
    artifact_dir: str | Path,
    model_name: str,
    variant: str,
    rank: int,
) -> dict[str, Any]:
    import torch

    artifact_dir = Path(artifact_dir)
    slug = safe_model_name(model_name)
    projection_path = artifact_dir / "projections" / f"{slug}_{variant}_r{rank}.pt"
    transport_path = artifact_dir / "transports" / f"{slug}_{variant}_r{rank}.pt"
    if not projection_path.exists() or not transport_path.exists():
        raise FileNotFoundError(
            "Missing Experiment 2B low-rank artifacts. Run run_04a first.\n"
            f"projection: {projection_path}\ntransport: {transport_path}"
        )
    projection = torch.load(projection_path, map_location="cpu")
    transport = torch.load(transport_path, map_location="cpu")
    return {
        "projection_path": str(projection_path),
        "transport_path": str(transport_path),
        "mu_by_layer": projection["mu_by_layer"],
        "P_by_layer": projection["P_by_layer"],
        "R": transport["R"],
        "weights": transport["weights"],
        "projection": projection,
        "transport": transport,
    }


def _project_resid(H: Any, artifacts: dict[str, Any]):
    import torch

    mu = artifacts["mu_by_layer"].to(H.device, dtype=H.dtype)
    P = artifacts["P_by_layer"].to(H.device, dtype=H.dtype)
    return torch.einsum("nld,ldr->nlr", H - mu.unsqueeze(0), P)


def _low_rank_obstruction_from_H(H: Any, artifacts: dict[str, Any], eps: float = 1e-8):
    import torch

    S = _project_resid(H, artifacts)
    R = artifacts["R"].to(S.device, dtype=S.dtype)
    w = artifacts["weights"].to(S.device, dtype=S.dtype)
    pred = torch.einsum("lrk,nlk->nlr", R, S[:, :-1, :])
    residual = pred - S[:, 1:, :]
    edge_terms = w[None, :] * residual.pow(2).sum(dim=2)
    raw = edge_terms.sum(dim=1)
    normalized = raw / (S.pow(2).sum(dim=(1, 2)) + eps)
    return {"raw": raw, "normalized": normalized, "edge_terms": edge_terms, "S": S}


def compute_low_rank_obstruction_from_resid_cache(model: Any, cache: Any, artifacts: dict[str, Any], token_pos: int = -1):
    import torch
    from src.cache import resid_site_names

    hs = []
    for name in resid_site_names(model):
        hs.append(cache[name][:, token_pos, :])
    H = torch.stack(hs, dim=1)
    return _low_rank_obstruction_from_H(H, artifacts)


def _saved_hook(saved: dict[str, Any], key: str):
    def hook_fn(act, hook):
        act.retain_grad()
        saved[key] = act
        return act

    return hook_fn


def _component_grad_from_saved(saved_act: Any, site: ComponentSite, token_pos: int):
    grad = saved_act.grad
    if site.site_type == "head":
        return grad[0, token_pos, site.head, :].detach().cpu()
    return grad[0, token_pos, :].detach().cpu()


def compute_obstruction_gradient_wrt_component(
    model: Any,
    conflict_prompt: str,
    site: ComponentSite,
    artifacts: dict[str, Any],
    true_answer: str,
    false_answer: str,
    token_pos: int = -1,
):
    import torch
    from src.cache import resid_site_names

    if token_pos == -1:
        token_pos = final_token_pos(model, conflict_prompt)
    tokens = model.to_tokens([conflict_prompt], prepend_bos=True)
    saved: dict[str, Any] = {}
    resid_names = resid_site_names(model)
    hook_names = sorted(set(resid_names + [site.hook_name]))

    def make_hook(name: str):
        def hook_fn(act, hook):
            if name == site.hook_name:
                act.retain_grad()
                saved["component"] = act
            if name in resid_names:
                saved[name] = act
            return act

        return hook_fn

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        model.run_with_hooks(tokens, fwd_hooks=[(name, make_hook(name)) for name in hook_names])
        H = torch.stack([saved[name][:, token_pos, :] for name in resid_names], dim=1)
        obstruction = _low_rank_obstruction_from_H(H, artifacts)["normalized"][0]
        obstruction.backward()
    grad = _component_grad_from_saved(saved["component"], site, token_pos)
    model.reset_hooks()
    return grad


def compute_margin_gradient_wrt_component(
    model: Any,
    conflict_prompt: str,
    site: ComponentSite,
    true_answer: str,
    false_answer: str,
    token_pos: int = -1,
    margin_gradient_mode: str = "first_token",
):
    import torch

    if margin_gradient_mode not in {"first_token", "sequence"}:
        raise ValueError(f"Unsupported margin_gradient_mode: {margin_gradient_mode}")
    if token_pos == -1:
        token_pos = final_token_pos(model, conflict_prompt)
    saved: dict[str, Any] = {}
    model.zero_grad(set_to_none=True)

    if margin_gradient_mode == "first_token":
        true_id = model.tokenizer.encode(true_answer, add_special_tokens=False)[0]
        false_id = model.tokenizer.encode(false_answer, add_special_tokens=False)[0]
        tokens = model.to_tokens([conflict_prompt], prepend_bos=True)
        with torch.enable_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(site.hook_name, _saved_hook(saved, "component"))])
            margin = logits[0, token_pos, true_id] - logits[0, token_pos, false_id]
            margin.backward()
    else:
        true_tokens, prompt_len, true_ids = _tokens_for_prompt_answer(model, conflict_prompt, true_answer)
        false_tokens, _, false_ids = _tokens_for_prompt_answer(model, conflict_prompt, false_answer)
        with torch.enable_grad():
            logits_true = model.run_with_hooks(true_tokens, fwd_hooks=[(site.hook_name, _saved_hook(saved, "component"))])
            logprob_true = _continuation_logprob_from_logits(logits_true, prompt_len, true_ids)
            model.zero_grad(set_to_none=True)
            logprob_true.backward(retain_graph=False)
            grad_true = _component_grad_from_saved(saved["component"], site, token_pos)
        saved.clear()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits_false = model.run_with_hooks(false_tokens, fwd_hooks=[(site.hook_name, _saved_hook(saved, "component"))])
            logprob_false = _continuation_logprob_from_logits(logits_false, prompt_len, false_ids)
            logprob_false.backward()
            grad_false = _component_grad_from_saved(saved["component"], site, token_pos)
        model.reset_hooks()
        return grad_true - grad_false

    grad = _component_grad_from_saved(saved["component"], site, token_pos)
    model.reset_hooks()
    return grad


def score_site_response(
    example: dict,
    site: ComponentSite,
    delta: Any,
    grad_O: Any,
    grad_m: Any,
    beta: float,
    gamma: float,
    model_name: str,
    variant: str,
    rank: int,
    margin_gradient_mode: str,
) -> dict:
    pred_delta_O = float((grad_O.float() * delta.float()).sum())
    pred_delta_m = float((grad_m.float() * delta.float()).sum())
    cost = float(delta.float().pow(2).sum())
    grad_saliency = float(grad_m.float().pow(2).sum().sqrt())
    response_score = -pred_delta_O + beta * pred_delta_m - gamma * cost
    return {
        "id": example["id"],
        "subject": example.get("subject", ""),
        "relation": example.get("relation", ""),
        "true_answer": example["true_answer"],
        "false_answer": example["false_answer"],
        "model": model_name,
        "variant": variant,
        "rank": rank,
        **site.to_row(),
        "pred_delta_O": pred_delta_O,
        "pred_obstruction_reduction": -pred_delta_O,
        "pred_delta_m": pred_delta_m,
        "delta_norm_sq": cost,
        "grad_saliency_score": grad_saliency,
        "activation_norm_score": math.sqrt(cost),
        "attribution_score": pred_delta_m,
        "abs_attribution_score": abs(pred_delta_m),
        "obstruction_only_score": -pred_delta_O,
        "beta": beta,
        "gamma": gamma,
        "response_score": response_score,
        "margin_gradient_mode": margin_gradient_mode,
    }


def off_target_kl_next_token(logits_before: Any, logits_after: Any, token_pos: int = -1) -> float:
    import torch.nn.functional as F

    log_p_after = logits_after[:, token_pos, :].log_softmax(dim=-1)
    p_before = logits_before[:, token_pos, :].softmax(dim=-1)
    return float(F.kl_div(log_p_after, p_before, reduction="batchmean").detach().cpu())


def observed_patch_eval(
    model: Any,
    example: dict,
    site: ComponentSite,
    clean_value: Any,
    artifacts: dict[str, Any],
    compute_kl: bool = True,
) -> dict:
    import torch
    from src.cache import resid_site_names

    conflict_prompt = example["conflict_prompt"]
    token_pos = final_token_pos(model, conflict_prompt)
    tokens = model.to_tokens([conflict_prompt], prepend_bos=True)
    with torch.no_grad():
        logits_before, cache_before = model.run_with_cache(
            tokens, names_filter=resid_site_names(model), remove_batch_dim=False
        )
        margin_before = multi_token_margin_from_logits_or_model(
            model, conflict_prompt, example["true_answer"], example["false_answer"]
        )
        margin_before = float(margin_before.detach().cpu())
        O_before = compute_low_rank_obstruction_from_resid_cache(model, cache_before, artifacts, token_pos=token_pos)

        logits_after, cache_after = patch_component_site(model, conflict_prompt, site, clean_value, token_pos=token_pos)
        # Recompute margin with patch hooks using continuation runs, since patched logits
        # from the prompt-only run do not include later answer-token contexts.
        hooks = [(site.hook_name, _patch_hook(site, clean_value, token_pos))]
        true_lp_tokens, prompt_len, true_ids = _tokens_for_prompt_answer(
            model, conflict_prompt, example["true_answer"]
        )
        false_lp_tokens, _, false_ids = _tokens_for_prompt_answer(model, conflict_prompt, example["false_answer"])
        logits_true = model.run_with_hooks(true_lp_tokens, fwd_hooks=hooks)
        logits_false = model.run_with_hooks(false_lp_tokens, fwd_hooks=hooks)
        margin_after = float(
            _continuation_logprob_from_logits(logits_true, prompt_len, true_ids)
            - _continuation_logprob_from_logits(logits_false, prompt_len, false_ids)
        )
        O_after = compute_low_rank_obstruction_from_resid_cache(model, cache_after, artifacts, token_pos=token_pos)
        kl = off_target_kl_next_token(logits_before, logits_after, token_pos=token_pos) if compute_kl else float("nan")
    return {
        "id": example["id"],
        **site.to_row(),
        "model": "",
        "variant": "",
        "rank": "",
        "margin_before": margin_before,
        "margin_after": float(margin_after),
        "actual_delta_m": float(margin_after - margin_before),
        "obstruction_before": float(O_before["normalized"][0].detach().cpu()),
        "obstruction_after": float(O_after["normalized"][0].detach().cpu()),
        "actual_delta_O": float((O_after["normalized"][0] - O_before["normalized"][0]).detach().cpu()),
        "actual_obstruction_reduction": float((O_before["normalized"][0] - O_after["normalized"][0]).detach().cpu()),
        "margin_increase_success": bool(margin_after > margin_before),
        "margin_flip_success": bool(margin_after > 0),
        "off_target_kl": kl,
        "kl_available": compute_kl,
    }
