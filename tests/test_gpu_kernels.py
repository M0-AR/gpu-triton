import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA GPU")


@cuda
def test_triton_add_matches_torch():
    from src.kernels.vector_add import triton_add, torch_add
    torch.manual_seed(0)
    for n in (98432, 1_000_000):  # 98432 = 96*1024+128 exercises the mask tail
        a = torch.rand(n, device="cuda")
        b = torch.rand(n, device="cuda")
        assert (triton_add(a, b) - torch_add(a, b)).abs().max().item() == 0.0


@cuda
def test_fused_matches_unfused():
    from src.kernels.fused_add_relu import triton_fused_add_relu, torch_unfused_add_relu
    torch.manual_seed(1)
    a = torch.rand(1_000_000, device="cuda") * 2 - 1
    b = torch.rand(1_000_000, device="cuda") * 2 - 1
    got = triton_fused_add_relu(a, b)
    assert (got - torch_unfused_add_relu(a, b)).abs().max().item() == 0.0
