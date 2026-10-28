import torch
import triton
import triton.language as tl

@triton.jit
def _outer_add_kernel(x_ptr, y_ptr, z_ptr, N0: tl.constexpr, N1: tl.constexpr):
    row_ids = tl.arange(0, N1)
    col_ids = tl.arange(0, N0)
    row = tl.load(y_ptr + row_ids)
    col = tl.load(x_ptr + col_ids)

    offs = row_ids[:, None] * N0 + col_ids[None, :]
    z = row[:, None] + col[None, :]
    tl.store(z_ptr + offs, z)


def outer_vector_add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    N0 = x.numel()
    N1 = y.numel()
    z = torch.empty((N1, N0), device=x.device, dtype=x.dtype)
    grid = (N1, N0)
    _outer_add_kernel[grid](x, y, z, N0, N1)
    return z
