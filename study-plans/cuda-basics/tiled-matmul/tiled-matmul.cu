#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float Asub[TILE_DIM][TILE_DIM];
    __shared__ float Bsub[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_DIM + threadIdx.x;
        int bRow = t * TILE_DIM + threadIdx.y;

        Asub[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            acc += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
