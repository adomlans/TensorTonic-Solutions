import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, out_ptr,
    N_Q, N_KV, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    q_row_mask = offs_m < N_Q
    d_mask = offs_d < D
    q_mask = q_row_mask[:, None] & d_mask[None, :]
    q = tl.load(Q_ptr + offs_m[:, None] * D + offs_d[None, :], mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for start_n in range(0, N_KV, BLOCK_N):
        cur_n = start_n + offs_n
        kv_row_mask = cur_n < N_KV
        kv_mask = kv_row_mask[:, None] & d_mask[None, :]
        k = tl.load(K_ptr + cur_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V_ptr + cur_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)

        # manual (BLOCK_M, D) @ (BLOCK_N, D)^T -> (BLOCK_M, BLOCK_N)
        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        s = tl.where(kv_row_mask[None, :], s, -1.0e30)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        # manual (BLOCK_M, BLOCK_N) @ (BLOCK_N, D) -> (BLOCK_M, D)
        o_i = o_i * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        m_i = m_new

    o_i = o_i / l_i[:, None]
    tl.store(out_ptr + offs_m[:, None] * D + offs_d[None, :], o_i, mask=q_mask)


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, out: torch.Tensor) -> None:
    N_Q, D = Q.shape
    N_KV = K.shape[0]
    scale = 1.0 / (D ** 0.5)
    BLOCK_M = 1
    BLOCK_N = 16
    # Round BLOCK_DMODEL up to nearest power of two >= D
    BLOCK_DMODEL = 1
    while BLOCK_DMODEL < D:
        BLOCK_DMODEL *= 2
    grid = ((N_Q + BLOCK_M - 1) // BLOCK_M,)
    flash_attention_fwd_kernel[grid](
        Q, K, V, out,
        N_Q, N_KV, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )