from __future__ import annotations

from typing import Any


def fit_ridge_matrix_transport(S_clean_train: Any, lambda_ridge: float = 1e-3):
    """Fit layerwise matrix ridge transports.

    Args:
        S_clean_train: tensor [n, L+1, r]
        lambda_ridge: ridge penalty

    Returns:
        R: tensor [L, r, r], mapping s_l to s_{l+1}
    """

    import torch

    S = S_clean_train.float()
    n_edges = S.shape[1] - 1
    rank = S.shape[2]
    eye = torch.eye(rank, dtype=S.dtype, device=S.device)
    transports = []
    for edge_idx in range(n_edges):
        x = S[:, edge_idx, :]
        y = S[:, edge_idx + 1, :]
        xtx = x.T @ x
        ytx = y.T @ x
        r_edge = ytx @ torch.linalg.inv(xtx + lambda_ridge * eye)
        transports.append(r_edge)
    return torch.stack(transports, dim=0)


def predict_next_germs(S: Any, R: Any):
    """Predict s_{l+1} from s_l using transports R."""

    import torch

    return torch.einsum("lrk,nlk->nlr", R, S[:, :-1, :])


def transport_r2(S: Any, R: Any):
    """Return edgewise R^2 for transports on S."""

    import torch

    y = S[:, 1:, :]
    y_hat = predict_next_germs(S, R)
    ss_res = (y_hat - y).pow(2).sum(dim=(0, 2))
    centered = y - y.mean(dim=0, keepdim=True)
    ss_tot = centered.pow(2).sum(dim=(0, 2)) + 1e-8
    return 1.0 - ss_res / ss_tot


def make_edge_weights(r2: Any, clamp: bool = True):
    """Convert edge R^2 diagnostics into obstruction weights."""

    import torch

    r2 = torch.as_tensor(r2).float()
    if clamp:
        return torch.clamp(r2, min=0.0)
    return r2


def compute_vector_obstruction(S: Any, R: Any, w: Any, eps: float = 1e-8):
    """Compute vector-valued residual-chain obstruction.

    Args:
        S: tensor [n, L+1, r]
        R: tensor [L, r, r]
        w: tensor [L]

    Returns dict with raw, normalized, edge_residuals, edge_terms.
    """

    import torch

    S = S.float()
    R = torch.as_tensor(R, dtype=S.dtype, device=S.device)
    w = torch.as_tensor(w, dtype=S.dtype, device=S.device)
    y_hat = predict_next_germs(S, R)
    edge_residuals = y_hat - S[:, 1:, :]
    edge_terms = w[None, :] * edge_residuals.pow(2).sum(dim=2)
    raw = edge_terms.sum(dim=1)
    denom = S.pow(2).sum(dim=(1, 2)) + eps
    normalized = raw / denom
    return {
        "raw": raw,
        "normalized": normalized,
        "edge_residuals": edge_residuals,
        "edge_terms": edge_terms,
    }


def identity_transports(L: int, r: int):
    """Return identity transports [L, r, r]."""

    import torch

    return torch.eye(r).unsqueeze(0).repeat(L, 1, 1)


def random_transports(L: int, r: int, seed: int = 0, orthogonal: bool = False):
    """Return random Gaussian or orthogonal transports."""

    import torch

    generator = torch.Generator().manual_seed(seed)
    mats = []
    for _ in range(L):
        raw = torch.randn(r, r, generator=generator)
        if orthogonal:
            q, _ = torch.linalg.qr(raw)
            mats.append(q)
        else:
            mats.append(raw / max(r ** 0.5, 1.0))
    return torch.stack(mats, dim=0)


def shuffled_transports(R: Any, seed: int = 0):
    """Return transports shuffled across edges."""

    import torch

    R = torch.as_tensor(R).clone()
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(R.shape[0], generator=generator)
    return R[perm]
