#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    __shared__ float s_sum[256];
    __shared__ float s_sqsum[256];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        float v = row_in[j];
        local_sum += v;
        local_sqsum += v * v;
    }
    s_sum[tid] = local_sum;
    s_sqsum[tid] = local_sqsum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sqsum[tid] += s_sqsum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / (float)N;
        float var = s_sqsum[0] / (float)N - mean * mean;
        if (var < 0.0f) var = 0.0f;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    for (int j = tid; j < N; j += blockDim.x) {
        float v = row_in[j];
        row_out[j] = (v - mean) * inv_std * gamma[j] + beta[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}

