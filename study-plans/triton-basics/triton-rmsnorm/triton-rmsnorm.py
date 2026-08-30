import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, gamma_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_row = x_ptr + row * stride_x_row + cols
    out_row = out_ptr + row * stride_out_row + cols

    x = tl.load(x_row, mask=mask, other=0.0)
    mean_square = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_square + eps)

    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)
    y = x * rstd * gamma
    tl.store(out_row, y, mask=mask)


def solve(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](
        x, gamma, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
