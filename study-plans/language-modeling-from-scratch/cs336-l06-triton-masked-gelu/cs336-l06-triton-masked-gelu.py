import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def masked_gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    result = 0.5 * x * (1.0 + libdevice.tanh(inner))
    tl.store(out_ptr + offsets, result, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    if x.numel() == 0:
        return
    kernel_x = x
    kernel_out = out
    if x.dtype == torch.bfloat16:
        kernel_x = x.to(torch.float32)
        kernel_out = torch.empty_like(x, dtype=torch.float32)
    block_size = 256
    grid = (triton.cdiv(x.numel(), block_size),)
    masked_gelu_kernel[grid](kernel_x, kernel_out, x.numel(), BLOCK_SIZE=block_size)
    if kernel_out is not out:
        out.copy_(kernel_out)