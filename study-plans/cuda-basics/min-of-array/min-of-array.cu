#include <cuda_runtime.h>
#include <float.h>

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_i = (int*) addr;
    int old = *addr_i;
    int assumed;
    do {
        assumed = old;
        float current = __int_as_float(assumed);
        if (value >= current) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void min_kernel(const float* input, float* result, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float v = FLT_MAX;
    if (i < N) {
        v = input[i];
    }
    s[tid] = v;
    __syncthreads();

    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) {
            s[tid] = fminf(s[tid], s[tid + st]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMinFloat(result, s[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    init_result<<<1, 1>>>(result);
    min_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
