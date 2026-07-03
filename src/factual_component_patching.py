from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.low_rank_germs import final_token_pos, project_activations
from src.vector_transports import compute_vector_obstruction


@dataclass(frozen=True)
class ComponentSite:
    kind: str
    layer: int
    head: int | None = None

    @property
    def id(self) -> str:
        head = "none" if self.head is None else str(self.head)
        return f"{self.kind}:L{self.layer}:H{head}:final_token"

    @property
    def hook_name(self) -> str:
        if self.kind == "head":
            return f"blocks.{self.layer}.attn.hook_z"
        if self.kind == "mlp":
            return f"blocks.{self.layer}.hook_mlp_out"
        if self.kind == "resid_pre":
            return f"blocks.{self.layer}.hook_resid_pre"
        raise ValueError(f"Unsupported site kind: {self.kind}")

    def to_row(self) -> dict:
        return asdict(self) | {"site_id": self.id}


def component_names_filter(model: Any, include_resid_controls: bool = False) -> list[str]:
    names = []
    for layer in range(model.cfg.n_layers):
        names.append(f"blocks.{layer}.attn.hook_z")
        names.append(f"blocks.{layer}.hook_mlp_out")
        if include_resid_controls:
            names.append(f"blocks.{layer}.hook_resid_pre")
    return names


def build_component_sites(
    model: Any,
    site_types: list[str] | None = None,
    include_resid_controls: bool = False,
) -> list[ComponentSite]:
    site_types = site_types or ["head", "mlp"]
    sites = []
    for layer in range(model.cfg.n_layers):
        if "head" in site_types:
            for head in range(model.cfg.n_heads):
                sites.append(ComponentSite("head", layer, head=head))
        if "mlp" in site_types:
            sites.append(ComponentSite("mlp", layer))
        if include_resid_controls or "resid_pre" in site_types:
            sites.append(ComponentSite("resid_pre", layer))
    return sites


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
    token_ids = [_bos_token_id(model)] + prompt_ids + answer_ids
    device = getattr(getattr(model, "cfg", None), "device", None)
    return torch.tensor([token_ids], device=device), len(prompt_ids), answer_ids


def site_activation(cache: Any, site: ComponentSite, row_idx: int, token_pos: int):
    act = cache[site.hook_name]
    if site.kind == "head":
        return act[row_idx, token_pos, site.head, :]
    return act[row_idx, token_pos, :]


def collect_clean_site_value(model: Any, clean_prompt: str, site: ComponentSite):
    from src.cache import run_with_resid_cache

    token_pos = final_token_pos(model, clean_prompt)
    _, _, cache = run_with_resid_cache(model, [clean_prompt], [site.hook_name])
    return site_activation(cache, site, 0, token_pos).detach().cpu()


def make_patch_hook(site: ComponentSite, token_pos: int, clean_value: Any):
    def patch_hook(act, hook):
        patched = act.clone()
        value = clean_value.to(patched.device, dtype=patched.dtype)
        if site.kind == "head":
            patched[0, token_pos, site.head, :] = value
        else:
            patched[0, token_pos, :] = value
        return patched

    return patch_hook


def continuation_logprob_with_hooks(
    model: Any,
    prompt: str,
    answer: str,
    hooks: list[tuple[str, Any]] | None = None,
) -> float:
    import torch

    tokens, prompt_len, answer_ids = _tokens_for_prompt_answer(model, prompt, answer)
    with torch.no_grad():
        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        total = 0.0
        for offset, answer_id in enumerate(answer_ids):
            total += float(log_probs[prompt_len + offset, answer_id].detach().cpu())
    return total


def margin_with_hooks(
    model: Any,
    prompt: str,
    true_answer: str,
    false_answer: str,
    hooks: list[tuple[str, Any]] | None = None,
) -> float:
    return continuation_logprob_with_hooks(model, prompt, true_answer, hooks) - continuation_logprob_with_hooks(
        model,
        prompt,
        false_answer,
        hooks,
    )


def resid_final_with_hooks(model: Any, prompt: str, hooks: list[tuple[str, Any]] | None = None):
    import torch
    from src.cache import resid_site_names

    site_names = resid_site_names(model)
    tokens = model.to_tokens([prompt], prepend_bos=True)
    with torch.no_grad():
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                _, cache = model.run_with_cache(tokens, names_filter=site_names, remove_batch_dim=False)
        else:
            _, cache = model.run_with_cache(tokens, names_filter=site_names, remove_batch_dim=False)
    token_pos = final_token_pos(model, prompt)
    return torch.stack([cache[name][0, token_pos, :].detach().cpu().float() for name in site_names], dim=0)


def low_rank_obstruction_for_resid(H_one: Any, projection: dict, transport: dict) -> dict:
    import torch

    H = H_one.unsqueeze(0) if H_one.ndim == 2 else H_one
    S = project_activations(H, projection["mu_by_layer"], projection["P_by_layer"])
    return compute_vector_obstruction(S, transport["R"], transport["weights"])


def evaluate_component_patch(
    model: Any,
    example: dict,
    site: ComponentSite,
    projection: dict,
    transport: dict,
) -> dict:
    clean_value = collect_clean_site_value(model, example["clean_prompt"], site)
    conflict_prompt = example["conflict_prompt"]
    token_pos = final_token_pos(model, conflict_prompt)
    hook = (site.hook_name, make_patch_hook(site, token_pos, clean_value))

    margin_conflict = margin_with_hooks(
        model,
        conflict_prompt,
        example["true_answer"],
        example["false_answer"],
        hooks=None,
    )
    margin_patched = margin_with_hooks(
        model,
        conflict_prompt,
        example["true_answer"],
        example["false_answer"],
        hooks=[hook],
    )

    H_conflict = resid_final_with_hooks(model, conflict_prompt, hooks=None)
    H_patched = resid_final_with_hooks(model, conflict_prompt, hooks=[hook])
    O_conflict = low_rank_obstruction_for_resid(H_conflict, projection, transport)
    O_patched = low_rank_obstruction_for_resid(H_patched, projection, transport)

    row = {
        **site.to_row(),
        "id": example["id"],
        "subject": example.get("subject", ""),
        "relation": example.get("relation", ""),
        "margin_conflict": margin_conflict,
        "margin_patched": margin_patched,
        "delta_margin": margin_patched - margin_conflict,
        "O_conflict": float(O_conflict["normalized"][0]),
        "O_patched": float(O_patched["normalized"][0]),
        "delta_O": float(O_patched["normalized"][0] - O_conflict["normalized"][0]),
        "obstruction_reduction": float(O_conflict["normalized"][0] - O_patched["normalized"][0]),
        "repair_success_margin_increase": margin_patched > margin_conflict,
        "repair_success_flip": margin_patched > 0,
    }
    return row
