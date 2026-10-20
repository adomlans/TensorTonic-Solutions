import torch
import triton
import triton.language as tl


@triton.jit
def tiled_row_sum_kernel(x_ptr, out_ptr, n_cols, row_stride, col_stride, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_SIZE)
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + lanes
        values = tl.load(x_ptr + row * row_stride + cols * col_stride, mask=cols < n_cols, other=0.0)
        accumulator += values.to(tl.float32)
    tl.store(out_ptr + row, tl.sum(accumulator, axis=0))


def solve(x: torch.Tensor, out: torch.Tensor, tile_size: int) -> None:
    if x.ndim != 2:
        raise ValueError("x must be a matrix")
    if tile_size < 32 or tile_size > 1024 or tile_size & (tile_size - 1):
        raise ValueError("tile_size must be a power of two from 32 through 1024")
    if x.shape[0] == 0:
        return
    if x.shape[1] == 0:
        out.zero_()
        return
    kernel_x = x.to(torch.float32) if x.dtype == torch.bfloat16 else x
    tiled_row_sum_kernel[(x.shape[0],)](
        kernel_x,
        out,
        x.shape[1],
        kernel_x.stride(0),
        kernel_x.stride(1),
        BLOCK_SIZE=tile_size,
    )