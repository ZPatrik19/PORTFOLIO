import pytest

from rag_engine.platform.device import cuda_available


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_cuda_runtime_visible():
    import torch

    assert torch.cuda.is_available()
