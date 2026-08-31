import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    row_start = row * N

    m = -1.0e30
    l = 0.0

    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        chunk = tl.load(x_ptr + row_start + cols, mask=mask, other=-1.0e30)
        m_chunk = tl.max(chunk, axis=0)
        m_new = tl.maximum(m, m_chunk)
        masked_chunk = tl.where(mask, chunk, -1.0e30)
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(masked_chunk - m_new), axis=0)
        m = m_new

    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        chunk = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        y = tl.exp(chunk - m) / l
        tl.store(out_ptr + row_start + cols, y, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    M, N = x.shape
    BLOCK_SIZE = 1024
    grid = (M,)
    online_softmax_kernel[grid](x, out, M, N, BLOCK_SIZE=BLOCK_SIZE)
