import pytest

from rag_engine.platform.device import cuda_available


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_cpu_cuda_cosine_similarity_parity():
    import torch

    cpu_vectors = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.5, 1.0]], dtype=torch.float32)
    cpu_query = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
    cpu_score = torch.nn.functional.cosine_similarity(cpu_vectors, cpu_query.unsqueeze(0), dim=1)
    gpu_score = torch.nn.functional.cosine_similarity(
        cpu_vectors.cuda(), cpu_query.cuda().unsqueeze(0), dim=1
    ).cpu()
    assert torch.allclose(cpu_score, gpu_score, rtol=1e-4, atol=1e-5)
