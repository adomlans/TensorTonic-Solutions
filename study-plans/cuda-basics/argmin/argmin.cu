#include <cuda_runtime.h>
#include <float.h>

__global__ void argmin_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    float v = FLT_MAX;
    int   j = N;
    if (i < N) {
        v = input[i];
        j = i;
    }
    s_val[tid] = v;
    s_idx[tid] = j;
    __syncthreads();

    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) {
            float va = s_val[tid];
            float vb = s_val[tid + st];
            int   ia = s_idx[tid];
            int   ib = s_idx[tid + st];
            bool take_b = (vb < va) || (vb == va && ib < ia);
            if (take_b) {
                s_val[tid] = vb;
                s_idx[tid] = ib;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = s_val[0];
        block_idxs[blockIdx.x] = s_idx[0];
    }
}

__global__ void argmin_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    int tid = threadIdx.x;

    float best_v = FLT_MAX;
    int   best_i = INT_MAX;
    for (int k = tid; k < num_blocks; k += blockDim.x) {
        float v = block_vals[k];
        int   j = block_idxs[k];
        if (v < best_v || (v == best_v && j < best_i)) {
            best_v = v;
            best_i = j;
        }
    }
    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) {
            float va = s_val[tid];
            float vb = s_val[tid + st];
            int   ia = s_idx[tid];
            int   ib = s_idx[tid + st];
            bool take_b = (vb < va) || (vb == va && ib < ia);
            if (take_b) {
                s_val[tid] = vb;
                s_idx[tid] = ib;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = s_idx[0];
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmin_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmin_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
