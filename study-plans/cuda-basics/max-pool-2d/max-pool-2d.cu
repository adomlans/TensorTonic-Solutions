#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_2d_kernel(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;
    if (i < outH && j < outW) {
        float m = -FLT_MAX;
        for (int a = 0; a < kH; a++) {
            for (int b = 0; b < kW; b++) {
                float v = input[(i * sH + a) * W + (j * sW + b)];
                m = fmaxf(m, v);
            }
        }
        output[i * outW + j] = m;
    }
}

extern "C" void solve(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    max_pool_2d_kernel<<<blocks, threads>>>(input, output, H, W, kH, kW, sH, sW);
    cudaDeviceSynchronize();
}
