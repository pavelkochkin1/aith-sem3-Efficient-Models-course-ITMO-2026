import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["n_elements", "BLOCK_SIZE"],
)
@triton.jit
def _quantize_rowwise(x_ptr, output_ptr, output_maxs, n_elements, BLOCK_SIZE: tl.constexpr, P2: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, P2)
    offs = row * BLOCK_SIZE + cols
    mask = (cols < BLOCK_SIZE) & (offs < n_elements)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(x), axis=0)
    denominator = tl.where(absmax == 0.0, 1.0, absmax)
    scaled = 127.0 * tl.div_rn(x, denominator)
    rounded = tl.extra.cuda.libdevice.nearbyint(scaled)

    tl.store(output_ptr + offs, rounded.to(tl.int8), mask=mask)
    tl.store(output_maxs + row, absmax)


def quantize_rowwise(x: torch.Tensor):
    n_rows, n_cols = x.shape
    q_int8 = torch.empty_like(x, dtype=torch.int8)
    absmaxs = torch.empty((n_rows,), device=x.device, dtype=torch.float16)
    P2 = triton.next_power_of_2(n_cols)

    _quantize_rowwise[(n_rows,)](x, q_int8, absmaxs, x.numel(), BLOCK_SIZE=n_cols, P2=P2)
    return q_int8, absmaxs
