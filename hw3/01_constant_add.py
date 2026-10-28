import torch
import triton
import triton.language as tl


@triton.jit
def _add_const_kernel(x_ptr, z_ptr, const_val, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    z = x + const_val
    tl.store(z_ptr + offs, z)


def add_triton(x: torch.Tensor, const_val: int) -> torch.Tensor:
    assert x.is_cuda, 'Тензор должен быть на GPU (CUDA).'
    N = x.numel()
    # по условию один блок полностью покрывает вектор
    grid = (1,)
    z = torch.empty_like(x)
    _add_const_kernel[grid](x, z, const_val, N)
    return z