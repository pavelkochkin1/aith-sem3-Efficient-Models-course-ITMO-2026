import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "GROUP_M": 8}, num_warps=4, num_stages=2)
    ],
    key=["M", "N"],
)
@triton.jit
def _quantize_global_transpose(A, absmax_inv_ptr, B,
                              stride_am, stride_an,
                              stride_bn, stride_bm,
                              M, N,
                              BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr,
                              GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    x = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
    absmax_inv = tl.load(absmax_inv_ptr).to(tl.float32)
    scaled = 127.0 * (x * absmax_inv)
    rounded = tl.extra.cuda.libdevice.nearbyint(scaled).to(tl.int8)

    b_ptrs = B + offs_n[None, :] * stride_bn + offs_m[:, None] * stride_bm
    tl.store(b_ptrs, rounded, mask=mask)


def quantize_global_transpose(x: torch.Tensor):
    """Return (q_T:int8[N,M], absmax:fp32[1])."""
    M, N = x.shape
    absmax = x.abs().max().to(torch.float32).reshape(1)
    denom = absmax.clamp_min(127.0 * 1e-8)
    absmax_inv = denom.reciprocal()
    q_T = torch.empty((N, M), device=x.device, dtype=torch.int8)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    _quantize_global_transpose[grid](
        x, absmax_inv, q_T,
        x.stride(0), x.stride(1), q_T.stride(0), q_T.stride(1),
        M, N,
    )
    return q_T, absmax
