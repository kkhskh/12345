from __future__ import annotations

import numpy as np
import pytest

from src.toy_othello import pad_batch, softmax_np


def test_pad_batch_preserves_lengths():
    toks, lengths = pad_batch([[1, 2, 3], [4]], device=None)
    assert lengths == [3, 1]
    assert int(toks[0, 2]) == 3
    assert int(toks[1, 1]) == 0


def test_softmax_rows_sum_to_one():
    p = softmax_np(np.zeros((4, 61)))
    assert np.allclose(p.sum(axis=-1), 1.0)


def test_patch_hook_changes_and_noop_matches():
    pytest.importorskip("transformer_lens")
    import torch
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=32,
        d_head=8,
        n_heads=4,
        d_mlp=64,
        d_vocab=61,
        n_ctx=16,
        act_fn="gelu",
        normalization_type="LNPre",
        tokenizer_name=None,
        seed=0,
    )
    model = HookedTransformer(cfg)
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    hook = "blocks.0.hook_resid_post"
    with torch.no_grad():
        base = model(tokens)[0, -1]
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook)
        h = cache[hook].clone()

        def noop(act, hook):
            out = act.clone()
            out[:, -1, :] = h[:, -1, :]
            return out

        def shift(act, hook):
            out = act.clone()
            out[:, -1, 0] = h[:, -1, 0] + 25.0
            return out

        z0 = model.run_with_hooks(tokens, fwd_hooks=[(hook, noop)])[0, -1]
        z1 = model.run_with_hooks(tokens, fwd_hooks=[(hook, shift)])[0, -1]
    assert torch.allclose(z0, base, atol=1e-5)
    assert not torch.allclose(z1, base, atol=1e-4)
