#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) {
            acc += Q[i * D + d] * K[j * D + d];
        }
        scores[i * N + j] = acc / sqrtf((float)D);
    }
}

__global__ void softmax_rows_kernel(float* scores, int N) {
    int row = blockIdx.x;
    if (row >= N) return;

    float* r = scores + row * N;
    int tid = threadIdx.x;

    __shared__ float buf[256];
    __shared__ float s_max;
    __shared__ float s_sum;

    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) local_max = fmaxf(local_max, r[j]);
    buf[tid] = local_max;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) buf[tid] = fmaxf(buf[tid], buf[tid + st]);
        __syncthreads();
    }
    if (tid == 0) s_max = buf[0];
    __syncthreads();
    float m = s_max;

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float e = expf(r[j] - m);
        r[j] = e;
        local_sum += e;
    }
    buf[tid] = local_sum;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) buf[tid] += buf[tid + st];
        __syncthreads();
    }
    if (tid == 0) s_sum = buf[0];
    __syncthreads();
    float inv = 1.0f / s_sum;

    for (int j = tid; j < N; j += blockDim.x) r[j] *= inv;
}

__global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && d < D) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) {
            acc += attn[i * N + j] * V[j * D + d];
        }
        output[i * D + d] = acc;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int D) {
    float* scores;
    cudaMalloc(&scores, (size_t)N * N * sizeof(float));

    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);
    scores_kernel<<<sBlocks, sThreads>>>(Q, K, scores, N, D);

    softmax_rows_kernel<<<N, 256>>>(scores, N);

    dim3 oThreads(16, 16);
    dim3 oBlocks((D + 15) / 16, (N + 15) / 16);
    av_kernel<<<oBlocks, oThreads>>>(scores, V, output, N, D);

    cudaDeviceSynchronize();
    cudaFree(scores);
}
