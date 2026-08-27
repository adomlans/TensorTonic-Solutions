#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void cross_entropy_row_kernel(const float* logits, const int* target, float* loss, int B, int C) {
    int row = blockIdx.x;
    if (row >= B) return;

    const float* row_logits = logits + row * C;
    int t = target[row];

    __shared__ float s_buf[256];
    __shared__ float s_max;
    __shared__ float s_lse;

    int tid = threadIdx.x;

    // Pass 1: row max
    float local_max = -FLT_MAX;
    for (int j = tid; j < C; j += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[j]);
    }
    s_buf[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_buf[tid] = fmaxf(s_buf[tid], s_buf[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) s_max = s_buf[0];
    __syncthreads();

    float row_max = s_max;

    // Pass 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        local_sum += expf(row_logits[j] - row_max);
    }
    s_buf[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_buf[tid] += s_buf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) s_lse = logf(s_buf[0]);
    __syncthreads();

    // Thread 0 contributes this row's loss
    if (tid == 0) {
        float target_logit = row_logits[t];
        float row_loss = -(target_logit - row_max - s_lse);
        atomicAdd(loss, row_loss);
    }
}

__global__ void cross_entropy_finalize_kernel(float* loss, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        loss[0] = loss[0] / (float)B;
    }
}

extern "C" void solve(const float* logits, const int* target, float* loss, int B, int C) {
    cudaMemset(loss, 0, sizeof(float));

    int threads = 256;
    dim3 blocks(B);
    cross_entropy_row_kernel<<<blocks, threads>>>(logits, target, loss, B, C);
    cross_entropy_finalize_kernel<<<1, 1>>>(loss, B);
    cudaDeviceSynchronize();
}
