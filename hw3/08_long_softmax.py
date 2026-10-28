import torch
import triton
import triton.language as tl


LOG2E = tl.constexpr(1.4426950408889634)  # log2(e)


@triton.jit
def _long_softmax_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    row_start = pid0 * T
    cols = tl.arange(0, B1)

    # Проход 1: максимум по всей строке.
    row_max = tl.full((), -float("inf"), tl.float32)
    for start in tl.range(0, T, B1):
        offs = start + cols
        tile = tl.load(
            x_ptr + row_start + offs, mask=offs < T, other=-float("inf")
        ).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(tile, axis=0))

    # Проход 2: сумма экспонент с общим максимумом строки.
    row_sum = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, T, B1):
        offs = start + cols
        tile = tl.load(
            x_ptr + row_start + offs, mask=offs < T, other=-float("inf")
        ).to(tl.float32)
        row_exp = tl.exp2((tile - row_max) * LOG2E)
        row_sum += tl.sum(row_exp, axis=0)

    # Проход 3: нормализация и запись каждого тайла.
    for start in tl.range(0, T, B1):
        offs = start + cols
        tile = tl.load(
            x_ptr + row_start + offs, mask=offs < T, other=-float("inf")
        ).to(tl.float32)
        row_exp = tl.exp2((tile - row_max) * LOG2E)
        out = row_exp / row_sum
        tl.store(out_ptr + row_start + offs, out, mask=offs < T)

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    N0, T = x.shape
    B1 = min(128, triton.next_power_of_2(T) // 2)  # степень двойки, B1 < T
    z = torch.empty_like(x)
    grid = (N0,)
    _long_softmax_kernel[grid](x, z, N0, T, B1)
    return z
