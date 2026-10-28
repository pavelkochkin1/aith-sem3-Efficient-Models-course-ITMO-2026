import torch
import triton
import triton.language as tl

@triton.jit
def _mul_relu_block_back_kernel(x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs0 = pid0 * B0 + tl.arange(0, B0)
    offs1 = pid1 * B1 + tl.arange(0, B1)
    mask0 = offs0 < N0
    mask1 = offs1 < N1

    offs = offs1[:, None] * N0 + offs0[None, :]
    mask = mask1[:, None] & mask0[None, :]

    y = tl.load(y_ptr + offs1, mask=mask1, other=0.0)
    x = tl.load(x_ptr + offs0, mask=mask0, other=0.0)
    dz_tile = tl.load(dz_ptr + offs, mask=mask, other=0.0)

    u = y[:, None] * x[None, :]
    relu_mask = u > 0

    g = tl.where(relu_mask, y[:, None] * dz_tile, 0.0)
    g_sum = tl.sum(g, axis=0)

    tl.atomic_add(dx_ptr + offs0, g_sum, mask=mask0)
    


def mul_relu_block_back_triton(x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    N0 = x.numel()
    N1 = y.numel()
    B0 = 128  # размер блока по оси 0
    B1 = 128  # размер блока по оси 1
    grid = ((N0 + B0 - 1) // B0, (N1 + B1 - 1) // B1)
    dx = torch.zeros_like(x)
    _mul_relu_block_back_kernel[grid](x, y, dz, dx, N0, N1, B0, B1)
    return dx
