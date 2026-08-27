#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* x = input + row * N;
    float* y = output + row * N;
    int tid = threadIdx.x;

    __shared__ float buf[256];
    __shared__ float s_inv_rms;

    float local = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = x[j];
        local += v * v;
    }
    buf[tid] = local;
    __syncthreads();

    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) buf[tid] += buf[tid + st];
        __syncthreads();
    }

    if (tid == 0) {
        float mean_sq = buf[0] / (float)N;
        s_inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float inv = s_inv_rms;
    for (int j = tid; j < N; j += blockDim.x) {
        y[j] = x[j] * inv * gamma[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}
