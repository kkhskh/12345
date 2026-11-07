"""SYNTHETIC_UNIT_TEST only. These do not verify frozen Othello-GPT hooks."""

from __future__ import annotations

import numpy as np

from src.othello_r1 import FakeResidualForward, softmax
from src.toy_othello import hook_name


def test_hook_names_are_distinct():
    mid = hook_name("resid_mid", 6)
    post = hook_name("resid_post", 6)
    assert mid == "blocks.6.hook_resid_mid"
    assert post == "blocks.6.hook_resid_post"
    assert mid != post


def test_fake_mid_and_post_tensors_differ():
    model = FakeResidualForward()
    mid = model.cache("blocks.0.hook_resid_mid")
    post = model.cache("blocks.0.hook_resid_post")
    assert mid.shape == post.shape
    assert not np.allclose(mid, post)


def test_patching_each_hook_is_distinguishable_and_labeled():
    model = FakeResidualForward()
    donor_mid = model.resid_mid.copy()
    donor_post = model.resid_post.copy()
    donor_mid[:, 0] += 3.0
    donor_post[:, 1] += 4.0
    r_mid = model.run_patched("blocks.0.hook_resid_mid", donor_mid)
    r_post = model.run_patched("blocks.0.hook_resid_post", donor_post)
    assert r_mid["hook"] == "blocks.0.hook_resid_mid"
    assert r_post["hook"] == "blocks.0.hook_resid_post"
    assert r_mid["pert_norm_mid"] > 1.0
    assert r_post["pert_norm_post"] > 1.0
    assert r_mid["logit_max_abs_diff"] != r_post["logit_max_abs_diff"]
    assert not np.allclose(r_mid["logits_after"], r_post["logits_after"])


def test_noop_post_patch_reproduces_logits_after_final_norm():
    model = FakeResidualForward()
    out = model.run_patched("blocks.0.hook_resid_post", model.resid_post)
    assert np.allclose(out["logits_after"], out["logits_before"], atol=1e-10)


def test_full_post_patch_reproduces_donor_logits():
    model = FakeResidualForward()
    donor_post = model.resid_post[::-1].copy()
    donor_logits = model.logits_from_post(donor_post)
    out = model.run_patched("blocks.0.hook_resid_post", donor_post)
    assert np.allclose(out["logits_after"], donor_logits, atol=1e-10)
    # After softmax the donor last-token distribution is recovered.
    assert np.allclose(softmax(out["logits_after"]), softmax(donor_logits), atol=1e-10)
