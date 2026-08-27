#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = input[i];
        local += v * v;
    }
    s[tid] = local;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) *sumv = s[0];
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float inv = rsqrtf(*sumv);
        output[i] = input[i] * inv;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_sq_sum<<<1, 256>>>(input, d_sum, N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}
