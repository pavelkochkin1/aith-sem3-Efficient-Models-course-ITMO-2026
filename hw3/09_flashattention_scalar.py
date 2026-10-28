import torch
import triton
import triton.language as tl


@triton.jit
def _flashatt_kernel(q_ptr, k_ptr, v_ptr, out_ptr, T, B1: tl.constexpr, LOG2E: tl.constexpr):
    pid0 = tl.program_id(0)
    q = tl.load(q_ptr + pid0).to(tl.float32)
    cols = tl.arange(0, B1)

    m = tl.full((), -float("inf"), tl.float32)
    l = tl.full((), 0.0, tl.float32)
    acc = tl.full((), 0.0, tl.float32)

    for start in tl.range(0, T, B1):
        offs = start + cols
        mask = offs < T
        k = tl.load(k_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        s = tl.where(mask, q * k * LOG2E, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(s - m_new)

        l = l * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p * v, axis=0)
        m = m_new

    tl.store(out_ptr + pid0, acc / l)


def flashatt_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    T = k.numel()
    B1 = min(128, max(1, triton.next_power_of_2(T) // 2))
    LOG2E = 1.4426950408889634  # log2(e)
    out = torch.empty((T,), device=v.device, dtype=v.dtype)
    _flashatt_kernel[(T,)](q, k, v, out, T, B1, LOG2E)
    return out
