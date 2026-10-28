import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(0)
    offs = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    absmax_inv = tl.load(absmax_inv_ptr).to(tl.float32)
    scaled = 127.0 * (x * absmax_inv)

    rounded = tl.extra.cuda.libdevice.nearbyint(scaled)

    tl.store(output_ptr + offs, rounded.to(tl.int8), mask=mask)


def quantize_global(x: torch.Tensor):
    x = x.contiguous()
    n_elements = x.numel()
    absmax = x.abs().max().reshape(1)
    absmax_inv = torch.where(absmax == 0, 0.0, absmax.float().reciprocal())
    q_int8 = torch.empty_like(x, dtype=torch.int8)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _quantize_global[grid](x, absmax_inv, q_int8, n_elements)
    return q_int8, absmax
