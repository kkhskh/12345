from __future__ import annotations

import torch


def compute_obstruction(
    s: torch.Tensor,
    r,
    w,
    eps: float = 1e-8,
):
    """
    Args:
      s: [batch, L+1] or [L+1]
      r: [L]
      w: [L]

    Returns:
      dict with:
        raw: [batch]
        normalized: [batch]
        edge_residuals: [batch, L]
        weighted_edge_terms: [batch, L]
    """
    if s.ndim == 1:
        s = s.unsqueeze(0)

    r = torch.as_tensor(r, device=s.device, dtype=s.dtype)
    w = torch.as_tensor(w, device=s.device, dtype=s.dtype)

    pred_next = s[:, :-1] * r[None, :]
    actual_next = s[:, 1:]

    edge_residuals = pred_next - actual_next
    weighted_edge_terms = w[None, :] * edge_residuals.pow(2)

    raw = weighted_edge_terms.sum(dim=1)
    denom = s.pow(2).sum(dim=1) + eps
    normalized = raw / denom

    return {
        "raw": raw,
        "normalized": normalized,
        "edge_residuals": edge_residuals,
        "weighted_edge_terms": weighted_edge_terms,
    }


def fragility(normalized_obstruction, margin, eps: float = 1e-8):
    return normalized_obstruction / (margin.abs() + eps)
