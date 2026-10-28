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
def _forward_int8_fused_kernel(x_q_ptr, x_scale_ptr,
                               w_q_ptr, w_scale_ptr,
                               b_ptr, y_ptr,
                               B, IN, OUT,
                               BLOCK_M: tl.constexpr,
                               BLOCK_N: tl.constexpr,
                               BLOCK_K: tl.constexpr,
                               PER_CHANNEL: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for start in tl.range(0, IN, BLOCK_K):
        offs_k = start + cols_k
        x_tile = tl.load(
            x_q_ptr + offs_m[:, None] * IN + offs_k[None, :],
            mask=(offs_m[:, None] < B) & (offs_k[None, :] < IN),
            other=0,
        )
        w_tile = tl.load(
            w_q_ptr + offs_k[:, None] * OUT + offs_n[None, :],
            mask=(offs_k[:, None] < IN) & (offs_n[None, :] < OUT),
            other=0,
        )
        acc = tl.dot(x_tile, w_tile, acc, out_dtype=tl.int32)

    x_scale = tl.load(x_scale_ptr).to(tl.float32)
    result = acc.to(tl.float32)
    if PER_CHANNEL:
        w_scale = tl.load(
            w_scale_ptr + offs_n, mask=offs_n < OUT, other=0.0
        ).to(tl.float32)
        alpha = x_scale * w_scale
        result = result * alpha[None, :]
    else:
        w_scale = tl.load(w_scale_ptr).to(tl.float32)
        alpha = x_scale * w_scale
        result = result * alpha

    if b_ptr is not None:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < OUT, other=0.0).to(tl.float32)
        result = result + bias[None, :]

    offs_y = offs_m[:, None] * OUT + offs_n[None, :]
    mask_y = (offs_m[:, None] < B) & (offs_n[None, :] < OUT)
    tl.store(y_ptr + offs_y, result.to(tl.float16), mask=mask_y)


def matmul_int8_fused(x_q: torch.Tensor,
                      x_scale: torch.Tensor,
                      w_q: torch.Tensor,
                      w_scale: torch.Tensor,
                      bias: torch.Tensor | None = None,
                      *, per_channel: bool = True) -> torch.Tensor:
    """Вернуть Y = dequant(X_q) @ dequant(W_q) + bias, dtype fp16, shape (B, OUT)."""
    B, IN = x_q.shape
    OUT = w_q.shape[1]
    y = torch.empty((B, OUT), device=x_q.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_M"]),
        triton.cdiv(OUT, meta["BLOCK_N"]),
    )
    _forward_int8_fused_kernel[grid](
        x_q, x_scale, w_q, w_scale, bias, y, B, IN, OUT,
        PER_CHANNEL=per_channel,
    )
    return y
