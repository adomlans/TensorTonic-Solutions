import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, loss_out_ptr,
    stride_logits_row,
    B, C,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C

    row_ptr = logits_ptr + row * stride_logits_row + cols
    logits = tl.load(row_ptr, mask=mask, other=-float('inf'))

    row_max = tl.max(logits, axis=0)
    shifted = logits - row_max
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)
    lse = tl.log(sum_exp) + row_max

    target_id = tl.load(target_ptr + row)
    target_logit = tl.load(logits_ptr + row * stride_logits_row + target_id)

    row_loss = lse - target_logit
    tl.atomic_add(loss_out_ptr, row_loss)


def solve(logits: torch.Tensor, target: torch.Tensor, loss_out: torch.Tensor) -> None:
    B, C = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(C)
    loss_out.zero_()
    grid = (B,)
    cross_entropy_kernel[grid](
        logits, target, loss_out,
        logits.stride(0),
        B, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    loss_out.div_(B)
