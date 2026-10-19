#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

template <typename T>
__device__ float load_float(const T* ptr, long long index);

template <>
__device__ float load_float<float>(const float* ptr, long long index) {
    return ptr[index];
}

template <>
__device__ float load_float<half>(const half* ptr, long long index) {
    return __half2float(ptr[index]);
}

template <>
__device__ float load_float<__nv_bfloat16>(const __nv_bfloat16* ptr, long long index) {
    return __bfloat162float(ptr[index]);
}

template <typename T>
__device__ void store_float(T* ptr, long long index, float value);

template <>
__device__ void store_float<float>(float* ptr, long long index, float value) {
    ptr[index] = value;
}

template <>
__device__ void store_float<half>(half* ptr, long long index, float value) {
    ptr[index] = __float2half(value);
}

template <>
__device__ void store_float<__nv_bfloat16>(__nv_bfloat16* ptr, long long index, float value) {
    ptr[index] = __float2bfloat16(value);
}

template <typename T>
__global__ void online_softmax_kernel(
    const T* input,
    T* output,
    int rows,
    int cols,
    long long input_row_stride,
    long long input_col_stride,
    long long output_row_stride,
    long long output_col_stride
) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    extern __shared__ float shared[];
    float* maxima = shared;
    float* sums = shared + blockDim.x;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int col = lane; col < cols; col += blockDim.x) {
        float value = load_float(input, row * input_row_stride + col * input_col_stride);
        float next_max = fmaxf(local_max, value);
        local_sum = expf(local_max - next_max) * local_sum + expf(value - next_max);
        local_max = next_max;
    }
    maxima[lane] = local_max;
    sums[lane] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            float left_max = maxima[lane];
            float right_max = maxima[lane + offset];
            float merged_max = fmaxf(left_max, right_max);
            float left_scale = left_max == -FLT_MAX ? 0.0f : expf(left_max - merged_max);
            float right_scale = right_max == -FLT_MAX ? 0.0f : expf(right_max - merged_max);
            sums[lane] = left_scale * sums[lane] + right_scale * sums[lane + offset];
            maxima[lane] = merged_max;
        }
        __syncthreads();
    }

    float row_max = maxima[0];
    float row_sum = sums[0];
    for (int col = lane; col < cols; col += blockDim.x) {
        float value = load_float(input, row * input_row_stride + col * input_col_stride);
        store_float(output, row * output_row_stride + col * output_col_stride, expf(value - row_max) / row_sum);
    }
}

extern "C" void solve(
    const void* input,
    void* output,
    int rows,
    int cols,
    long long input_row_stride,
    long long input_col_stride,
    long long output_row_stride,
    long long output_col_stride,
    int block_size,
    int dtype_code
) {
    if (
        rows < 0 ||
        cols < 0 ||
        block_size < 32 ||
        block_size > 1024 ||
        (block_size & (block_size - 1)) != 0 ||
        dtype_code < 0 ||
        dtype_code > 2
    ) {
        cudaDeviceSynchronize();
        return;
    }
    if (rows == 0 || cols == 0) {
        cudaDeviceSynchronize();
        return;
    }
    size_t shared_bytes = 2 * block_size * sizeof(float);
    if (dtype_code == 0) {
        online_softmax_kernel<<<rows, block_size, shared_bytes>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            rows,
            cols,
            input_row_stride,
            input_col_stride,
            output_row_stride,
            output_col_stride
        );
    } else if (dtype_code == 1) {
        online_softmax_kernel<<<rows, block_size, shared_bytes>>>(
            static_cast<const half*>(input),
            static_cast<half*>(output),
            rows,
            cols,
            input_row_stride,
            input_col_stride,
            output_row_stride,
            output_col_stride
        );
    } else {
        online_softmax_kernel<<<rows, block_size, shared_bytes>>>(
            static_cast<const __nv_bfloat16*>(input),
            static_cast<__nv_bfloat16*>(output),
            rows,
            cols,
            input_row_stride,
            input_col_stride,
            output_row_stride,
            output_col_stride
        );
    }
    cudaDeviceSynchronize();
}