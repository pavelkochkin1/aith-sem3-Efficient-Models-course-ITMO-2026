import torch
import triton
import triton.language as tl


@triton.jit
def _add_const_block_kernel(x_ptr, z_ptr, const_val, N: tl.constexpr, B0: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * B0 + tl.arange(0, B0)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    z = x + const_val
    tl.store(z_ptr + offs, z, mask=mask)


def add_block_triton(x: torch.Tensor, const_val: int) -> torch.Tensor:
    assert x.is_cuda, 'Тензор должен быть на GPU (CUDA).'
    N = x.numel()
    B0 = 128  # например, фиксированный размер блока
    grid = ((N + B0 - 1) // B0,)
    z = torch.empty_like(x)
    _add_const_block_kernel[grid](x, z, const_val, N, B0)
    return z