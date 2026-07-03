from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


DEFAULT_FACTUAL_CANDIDATES = [
    "data/factual_conflict/factual_conflict_margin_matched.jsonl",
    "artifacts/results/factual_conflict_margin_matched.csv",
    "data/factual_conflict/factual_conflict_scored.jsonl",
]


def safe_model_name(model_name: str) -> str:
    return (
        model_name.replace("/", "__")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def find_candidate_input_files() -> list[str]:
    roots = [Path("data"), Path("artifacts")]
    out: list[str] = []
    for root in roots:
        if root.exists():
            out.extend(str(path) for path in root.rglob("*factual*") if path.suffix in {".jsonl", ".csv"})
    return sorted(out)


def autodetect_factual_input() -> str:
    for candidate in DEFAULT_FACTUAL_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = find_candidate_input_files()
    message = "Could not find default factual input. Candidate factual files found:\n"
    message += "\n".join(f"  - {path}" for path in found) if found else "  (none)"
    raise FileNotFoundError(message)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path: Path) -> list[dict]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_factual_example(row: dict) -> dict:
    """Normalize factual-conflict row fields with multi-token answer support."""

    out = dict(row)
    if not out.get("id"):
        parts = [
            out.get("subject", ""),
            out.get("relation", ""),
            out.get("true_object", ""),
            out.get("false_object", ""),
            out.get("template_id", "default"),
        ]
        out["id"] = "fact_" + "_".join(str(part).replace(" ", "_") for part in parts if part)

    if not out.get("clean_prompt"):
        raise ValueError(f"Example {out.get('id')} is missing clean_prompt")
    if not out.get("conflict_prompt"):
        out["conflict_prompt"] = out.get("correction_prompt") or out.get("false_context_prompt")
    if not out.get("conflict_prompt"):
        raise ValueError(f"Example {out.get('id')} is missing conflict/correction prompt")

    if not out.get("answer_a"):
        true_obj = out.get("true_object") or out.get("true_answer")
        if true_obj:
            out["answer_a"] = " " + str(true_obj).strip()
    if not out.get("answer_b"):
        false_obj = out.get("false_object") or out.get("false_answer")
        if false_obj:
            out["answer_b"] = " " + str(false_obj).strip()
    if not out.get("answer_a") or not out.get("answer_b"):
        raise ValueError(f"Example {out.get('id')} is missing answer_a/answer_b")

    out["true_answer"] = out.get("true_answer") or out["answer_a"]
    out["false_answer"] = out.get("false_answer") or out["answer_b"]
    out["subject"] = out.get("subject", "")
    out["relation"] = out.get("relation", "")
    return out


def load_factual_examples(path: str | None = None) -> list[dict]:
    if path is None:
        path = autodetect_factual_input()
    input_path = Path(path)
    if not input_path.exists():
        found = find_candidate_input_files()
        raise FileNotFoundError(
            f"Input file not found: {path}\nCandidate factual files:\n"
            + ("\n".join(f"  - {item}" for item in found) if found else "  (none)")
        )
    rows = _read_jsonl(input_path) if input_path.suffix == ".jsonl" else _read_csv(input_path)
    return [normalize_factual_example(row) for row in rows]


def split_examples(
    examples: list[dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 12345,
) -> dict[str, str]:
    """Split by fact key when possible, returning id -> split."""

    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for ex in examples:
        key_parts = [
            ex.get("subject"),
            ex.get("relation"),
            ex.get("true_object"),
            ex.get("false_object"),
        ]
        key = "||".join(str(part) for part in key_parts if part not in {None, ""}) or ex["id"]
        groups.setdefault(key, []).append(ex["id"])

    keys = sorted(groups)
    rng.shuffle(keys)
    n_train = int(round(len(keys) * train_frac))
    n_val = int(round(len(keys) * val_frac))
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    split_by_id: dict[str, str] = {}
    for key, ids in groups.items():
        split = "train" if key in train_keys else "val" if key in val_keys else "test"
        for ex_id in ids:
            split_by_id[ex_id] = split
    return split_by_id


def answer_token_ids(model: Any, answer: str) -> list[int]:
    toks = model.tokenizer.encode(answer, add_special_tokens=False)
    if not toks:
        raise ValueError(f"Answer {answer!r} tokenized to an empty sequence")
    return [int(tok) for tok in toks]


def sequence_logprob(model: Any, prompt: str, answer: str) -> float:
    from src.model import continuation_logprob

    return continuation_logprob(model, prompt, answer_token_ids(model, answer))


def batch_sequence_logprobs(model: Any, prompts: list[str], answers: list[str]) -> list[float]:
    return [sequence_logprob(model, prompt, answer) for prompt, answer in zip(prompts, answers)]


def multi_token_margin(model: Any, prompt: str, true_answer: str, false_answer: str) -> float:
    return sequence_logprob(model, prompt, true_answer) - sequence_logprob(model, prompt, false_answer)


def resid_site_names(model: Any) -> list[str]:
    from src.cache import resid_site_names as _resid_site_names

    return _resid_site_names(model)


def final_token_pos(model: Any, prompt: str) -> int:
    # TransformerLens prepends BOS in model.to_tokens, so this is the prompt-final
    # token index in cached activations.
    return len(model.tokenizer.encode(prompt, add_special_tokens=False))


def run_resid_final_token_cache(
    model: Any,
    prompts: list[str],
    batch_size: int = 8,
    device: str | None = None,
):
    """Return residual activations [n_examples, L+1, d_model] at final prompt token."""

    import torch
    from tqdm.auto import tqdm
    from src.cache import run_with_resid_cache

    site_names = resid_site_names(model)
    all_rows = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="cache residual final-token"):
        batch = prompts[start:start + batch_size]
        _, _, cache = run_with_resid_cache(model, batch)
        for i, prompt in enumerate(batch):
            token_pos = final_token_pos(model, prompt)
            hs = [cache[name][i, token_pos, :].detach().cpu().float() for name in site_names]
            all_rows.append(torch.stack(hs, dim=0))
    return torch.stack(all_rows, dim=0)


def fit_pca_basis(X: Any, rank: int):
    """Fit PCA/SVD basis; returns (mu, P, explained_variance). P is [d, r]."""

    import torch

    X = X.float()
    mu = X.mean(dim=0)
    centered = X - mu
    max_rank = min(rank, centered.shape[0], centered.shape[1])
    if max_rank < 1:
        raise ValueError("Cannot fit PCA basis with empty data")
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    P = vh[:max_rank].T.contiguous()
    if max_rank < rank:
        pad = torch.zeros(P.shape[0], rank - max_rank, dtype=P.dtype)
        P = torch.cat([P, pad], dim=1)
    denom = max(centered.shape[0] - 1, 1)
    explained = (s[:max_rank].pow(2) / denom).tolist()
    explained += [0.0] * (rank - max_rank)
    return mu, P[:, :rank], explained[:rank]


def orthonormalize_columns(P: Any):
    import torch

    q, _ = torch.linalg.qr(P.float(), mode="reduced")
    return q


def answer_contrast_vector(model: Any, true_answer: str, false_answer: str, mode: str = "mean_unembed"):
    """Approximate sequence answer contrast with mean unembedding direction."""

    import torch

    if mode != "mean_unembed":
        raise ValueError(f"Unsupported answer contrast mode: {mode}")
    true_ids = answer_token_ids(model, true_answer)
    false_ids = answer_token_ids(model, false_answer)
    true_vec = model.W_U[:, true_ids].mean(dim=1)
    false_vec = model.W_U[:, false_ids].mean(dim=1)
    du = true_vec - false_vec
    du = du / (du.norm() + 1e-8)
    return du.detach().cpu()


def _fit_layerwise_clean_pca(H_clean_train: Any, rank: int):
    import torch

    mus, bases, explained = [], [], []
    for layer_idx in range(H_clean_train.shape[1]):
        mu, P, ev = fit_pca_basis(H_clean_train[:, layer_idx, :], rank)
        mus.append(mu)
        bases.append(P)
        explained.append(torch.tensor(ev))
    return torch.stack(mus, dim=0), torch.stack(bases, dim=0), torch.stack(explained, dim=0)


def build_clean_pca_germs(H_clean_train: Any, rank: int):
    return _fit_layerwise_clean_pca(H_clean_train, rank)


def build_answer_augmented_clean_pca_germs(
    model: Any,
    examples: list[dict],
    H_clean_train: Any,
    rank: int,
):
    """Build layer bases with an average answer-contrast direction plus clean PCA."""

    import torch

    mus, pca_bases, explained = _fit_layerwise_clean_pca(H_clean_train, max(rank, 1))
    du = torch.stack(
        [
            answer_contrast_vector(model, ex["true_answer"], ex["false_answer"])
            for ex in examples
        ],
        dim=0,
    ).mean(dim=0)
    du = (du / (du.norm() + 1e-8)).to(pca_bases.device)
    bases = []
    for layer_idx in range(H_clean_train.shape[1]):
        if rank == 1:
            basis = du[:, None]
        else:
            candidates = torch.cat([du[:, None], pca_bases[layer_idx, :, :rank]], dim=1)
            basis = orthonormalize_columns(candidates)[:, :rank]
        bases.append(basis)
    return mus, torch.stack(bases, dim=0), explained[:, :rank]


def build_delta_pca_diagnostic_germs(H_clean_train: Any, H_conflict_train: Any, rank: int):
    return _fit_layerwise_clean_pca(H_clean_train - H_conflict_train, rank)


def project_activations(H: Any, mu_by_layer: Any, P_by_layer: Any):
    """Project activations H [n, L+1, d] to germs S [n, L+1, r]."""

    import torch

    centered = H.float() - mu_by_layer.float().unsqueeze(0)
    return torch.einsum("nld,ldr->nlr", centered, P_by_layer.float())
