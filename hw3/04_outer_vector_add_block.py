import torch
import triton
import triton.language as tl

@triton.jit
def _outer_add_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs0 = pid0 * B0 + tl.arange(0, B0)
    offs1 = pid1 * B1 + tl.arange(0, B1)
    mask0 = offs0 < N0
    mask1 = offs1 < N1

    row = tl.load(y_ptr + offs1, mask=mask1)
    col = tl.load(x_ptr + offs0, mask=mask0)
    offs = offs1[:, None] * N0 + offs0[None, :]

    z = row[:, None] + col[None, :]
    tl.store(z_ptr + offs, z, mask=mask1[:, None] & mask0[None, :])


def add_vec_block_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    N0 = x.numel()
    N1 = y.numel()
    B0 = 128
    B1 = 128
    grid = ((N0 + B0 - 1) // B0, (N1 + B1 - 1) // B1)
    z = torch.empty((N1, N0), device=x.device, dtype=x.dtype)
    _outer_add_block_kernel[grid](x, y, z, N0, N1, B0, B1)
    return z
