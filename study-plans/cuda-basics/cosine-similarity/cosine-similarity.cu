#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_partials_kernel(const float* A, const float* B, float* scratch, int N) {
    __shared__ float sdot[256];
    __shared__ float sa[256];
    __shared__ float sb[256];

    int tid = threadIdx.x;
    float d = 0.0f, a2 = 0.0f, b2 = 0.0f;

    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        float a = A[i], b = B[i];
        d += a * b;
        a2 += a * a;
        b2 += b * b;
    }
    sdot[tid] = d; sa[tid] = a2; sb[tid] = b2;
    __syncthreads();

    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) {
            sdot[tid] += sdot[tid + st];
            sa[tid]   += sa[tid + st];
            sb[tid]   += sb[tid + st];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&scratch[0], sdot[0]);
        atomicAdd(&scratch[1], sa[0]);
        atomicAdd(&scratch[2], sb[0]);
    }
}

__global__ void cosine_finalize_kernel(const float* scratch, float* result) {
    float dot = scratch[0];
    float na = sqrtf(scratch[1]);
    float nb = sqrtf(scratch[2]);
    *result = dot / (na * nb);
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    float* scratch;
    cudaMalloc(&scratch, 3 * sizeof(float));
    cudaMemset(scratch, 0, 3 * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    cosine_partials_kernel<<<blocks, threads>>>(A, B, scratch, N);
    cosine_finalize_kernel<<<1, 1>>>(scratch, result);

    cudaDeviceSynchronize();
    cudaFree(scratch);
}
