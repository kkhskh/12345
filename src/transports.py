from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .cache import run_with_resid_cache
from .germs import scalar_germs


def fit_ridge_scalar_r(s: torch.Tensor, lambda_ridge: float = 1e-3):
    """
    s: [n_examples, L+1]
    returns:
      r: [L]
      r2: [L]
      w: [L]
    """
    x = s[:, :-1]
    y = s[:, 1:]

    numerator = (x * y).sum(dim=0)
    denominator = (x * x).sum(dim=0) + lambda_ridge
    r = numerator / denominator

    y_hat = x * r[None, :]
    ss_res = ((y_hat - y) ** 2).sum(dim=0)
    ss_tot = ((y - y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0) + 1e-8

    r2 = 1.0 - ss_res / ss_tot
    w = torch.clamp(r2, min=0.0)

    return r, r2, w


@torch.no_grad()
def collect_clean_germs(
    model,
    clean_examples,
    batch_size: int = 32,
    token_pos: int = -1,
):
    all_s = []

    for start in tqdm(range(0, len(clean_examples), batch_size)):
        batch = clean_examples[start : start + batch_size]
        prompts = [ex["clean_prompt"] for ex in batch]

        _, _, cache = run_with_resid_cache(model, prompts)

        batch_s = []
        for i, ex in enumerate(batch):
            s_i = scalar_germs(
                model=model,
                cache={key: value[i : i + 1] for key, value in cache.items()},
                answer_a_id=ex["repair_answer_a_id"],
                answer_b_id=ex["repair_answer_b_id"],
                token_pos=token_pos,
            )
            batch_s.append(s_i.squeeze(0))

        all_s.append(torch.stack(batch_s, dim=0).detach().cpu())

    return torch.cat(all_s, dim=0)


def fit_scalar_transports(
    model,
    train_examples,
    val_examples=None,
    lambda_ridge: float = 1e-3,
    batch_size: int = 32,
    save_path: str | Path | None = None,
):
    """
    Fit scalar transports r_l on clean train prompts.
    """
    s_train = collect_clean_germs(
        model,
        train_examples,
        batch_size=batch_size,
    )

    r, train_r2, w = fit_ridge_scalar_r(s_train, lambda_ridge=lambda_ridge)

    diagnostics = {
        "lambda_ridge": lambda_ridge,
        "train_r2": train_r2.tolist(),
        "weights": w.tolist(),
        "r": r.tolist(),
    }

    if val_examples is not None:
        s_val = collect_clean_germs(
            model,
            val_examples,
            batch_size=batch_size,
        )
        x = s_val[:, :-1]
        y = s_val[:, 1:]
        y_hat = x * r[None, :]

        ss_res = ((y_hat - y) ** 2).sum(dim=0)
        ss_tot = ((y - y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0) + 1e-8
        val_r2 = 1.0 - ss_res / ss_tot

        diagnostics["val_r2"] = val_r2.tolist()

    controls = {
        "identity_r": torch.ones_like(r).tolist(),
        "identity_w": torch.ones_like(w).tolist(),
        "random_r": torch.randn_like(r).tolist(),
        "random_w": torch.ones_like(w).tolist(),
    }

    out = {
        "r": r.tolist(),
        "w": w.tolist(),
        "diagnostics": diagnostics,
        "controls": controls,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)

    return out
