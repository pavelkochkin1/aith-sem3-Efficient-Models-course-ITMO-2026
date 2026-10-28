import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=2)
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _backward_dx_fused_kernel(dy_ptr, wq_ptr, w_scale_ptr, dx_ptr,
                              B, IN, OUT,
                              BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr,
                              BLOCK_K: tl.constexpr,
                              PER_CHANNEL: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_b = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in tl.range(0, OUT, BLOCK_K):
        offs_k = start + cols_k
        dy_tile = tl.load(
            dy_ptr + offs_b[:, None] * OUT + offs_k[None, :],
            mask=(offs_b[:, None] < B) & (offs_k[None, :] < OUT),
            other=0.0,
        ).to(tl.float32)
        w_tile = tl.load(
            wq_ptr + offs_i[None, :] * OUT + offs_k[:, None],
            mask=(offs_i[None, :] < IN) & (offs_k[:, None] < OUT),
            other=0,
        ).to(tl.float32)

        if PER_CHANNEL:
            scales = tl.load(
                w_scale_ptr + offs_k, mask=offs_k < OUT, other=0.0
            ).to(tl.float32)
            w_tile = w_tile * scales[:, None]

        acc = tl.dot(
            dy_tile, w_tile, acc, input_precision="ieee", out_dtype=tl.float32
        )

    if not PER_CHANNEL:
        scale = tl.load(w_scale_ptr).to(tl.float32)
        acc = acc * scale

    offs_dx = offs_b[:, None] * IN + offs_i[None, :]
    mask_dx = (offs_b[:, None] < B) & (offs_i[None, :] < IN)
    tl.store(dx_ptr + offs_dx, acc.to(tl.float16), mask=mask_dx)


def backward_dx_int8_fused(dy: torch.Tensor,
                           w_q: torch.Tensor,
                           w_scale: torch.Tensor,
                           *, per_channel: bool = True) -> torch.Tensor:
    """Вернуть dX = dY @ (dequant(W_q))^T, dtype fp16, shape (B, IN)."""
    B, OUT = dy.shape
    IN = w_q.shape[0]
    dx = torch.empty((B, IN), device=dy.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_M"]),
        triton.cdiv(IN, meta["BLOCK_N"]),
    )
    _backward_dx_fused_kernel[grid](
        dy, w_q, w_scale, dx, B, IN, OUT, PER_CHANNEL=per_channel
    )
    return dx
