import torch
import triton
import triton.language as tl

@triton.jit
def _sum2_kernel(x_ptr, y_ptr, z_ptr):
    x = tl.load(x_ptr)
    y = tl.load(y_ptr)
    z = x + y
    tl.store(z_ptr, z)


def sum2_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, 'Оба скаляра должны быть на CUDA'
    assert x.numel() == 1 and y.numel() == 1, 'Ожидаются скаляры'
    z = torch.empty_like(x)
    _sum2_kernel[(1,)](x, y, z)
    return z