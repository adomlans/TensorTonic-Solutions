#include <cuda_runtime.h>
#include <float.h>

__global__ void block_max_kernel(const float* input, float* partials, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float v = -FLT_MAX;
    if (i < N) {
        v = input[i];
    }
    s[tid] = v;
    __syncthreads();

    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) {
            s[tid] = fmaxf(s[tid], s[tid + st]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = s[0];
    }
}

__global__ void final_max_kernel(const float* partials, float* result, int M) {
    __shared__ float s[256];
    int tid = threadIdx.x;

    float v = -FLT_MAX;
    for (int i = tid; i < M; i += blockDim.x) {
        v = fmaxf(v, partials[i]);
    }
    s[tid] = v;
    __syncthreads();

    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) {
            s[tid] = fmaxf(s[tid], s[tid + st]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = s[0];
    }
}

__global__ void max_kernel(const float* input, float* result, int N) {
    // Stub kept so the kernel name remains in the translation unit; real work
    // is split across block_max_kernel and final_max_kernel below.
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* partials = nullptr;
    cudaMalloc(&partials, blocks * sizeof(float));

    block_max_kernel<<<blocks, threads>>>(input, partials, N);
    final_max_kernel<<<1, 256>>>(partials, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(partials);
}
