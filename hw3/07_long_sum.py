import torch
import triton
import triton.language as tl

@triton.jit
def _long_sum_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    row_start = pid0 * T
    cols = tl.arange(0, B1)

    row_sum = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, T, B1):
        offs = start + cols
        tile = tl.load(
            x_ptr + row_start + offs, mask=offs < T, other=0.0
        ).to(tl.float32)
        row_sum += tl.sum(tile, axis=0)

    tl.store(out_ptr + pid0, row_sum)


def sum_triton(x: torch.Tensor) -> torch.Tensor:
    N0, T = x.shape
    B1 = min(128, triton.next_power_of_2(T) // 2)  # степень двойки, B1 < T
    z = torch.empty((N0,), device=x.device, dtype=x.dtype)
    grid = (N0,)
    _long_sum_kernel[grid](x, z, N0, T, B1)
    return z
