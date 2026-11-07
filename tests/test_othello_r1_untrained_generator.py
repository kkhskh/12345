from __future__ import annotations

import numpy as np
import pytest

from src.othello_r1 import apply_cpu_normal_to_tensors, cpu_generator_fill


def test_cpu_generator_fill_is_deterministic_and_device_free():
    a = cpu_generator_fill([(3, 4), (2,)], seed=5)
    b = cpu_generator_fill([(3, 4), (2,)], seed=5)
    c = cpu_generator_fill([(3, 4), (2,)], seed=6)
    assert a[0].shape == (3, 4)
    assert np.allclose(a[0], b[0])
    assert not np.allclose(a[0], c[0])


def test_torch_cpu_generator_can_fill_cuda_shaped_tensors_without_device_generator():
    torch = pytest.importorskip("torch")
    p = torch.zeros(8, 8)
    apply_cpu_normal_to_tensors([p], seed=0)
    assert p.std() > 0
    # The historical bug used torch.Generator() (CPU) with p.normal_ on CUDA.
    if torch.cuda.is_available():
        q = torch.zeros(8, 8, device="cuda")
        apply_cpu_normal_to_tensors([q], seed=1)
        assert q.device.type == "cuda"
        assert float(q.std().cpu()) > 0
