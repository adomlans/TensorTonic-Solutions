import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_row_ptr = x_ptr + row * stride_x_row + cols
    out_row_ptr = out_ptr + row * stride_out_row + cols

    x = tl.load(x_row_ptr, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0)

    y = centered * rstd * gamma + beta
    tl.store(out_row_ptr, y, mask=mask)


def solve(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    layernorm_fwd_kernel[grid](
        x, gamma, beta, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
