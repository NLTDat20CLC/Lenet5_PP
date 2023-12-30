#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define UNROLL_FACTOR 4  // You can adjust the unroll factor based on performance testing

__global__ void conv_forward_kernel_op(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);

    float accum = 0.0f;

    // Shared memory for tiles
    __shared__ float shared_x[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_k[TILE_WIDTH][TILE_WIDTH];

    // Loop over the input channels with loop unrolling
    for (int c = 0; c < C; c += UNROLL_FACTOR * TILE_WIDTH)
    {
        // Load tiles into shared memory with loop unrolling
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR * TILE_WIDTH; ++i)
        {
            int idx = c + i;
            if (idx < C)
            {
                shared_x[ty][i] = x[(bx * C + idx + ty) * (H * W) + (by * TILE_WIDTH + i)];
            }
            else
            {
                shared_x[ty][i] = 0.0f;  // Padding for out-of-bounds access
            }

            shared_k[ty][i] = k[(by * M + idx + ty) * (K) + i];
        }

        __syncthreads();

        // Compute tile multiplication with loop unrolling
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR * TILE_WIDTH; ++i)
        {
            int idx = c + i;
            if (idx < C)
            {
                accum += shared_x[ty][i] * shared_k[i][tx];
            }
        }

        __syncthreads();
    }

    int h = by * TILE_WIDTH + ty;
    int w = bx * TILE_WIDTH + tx;

    if (h < H_out && w < W_out)
    {
        y[(bx * M * H_out * W_out) + (by * H_out * W_out) + h * W_out + w] = accum;
    }
}
	
__host__ void GPUInterface2::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int inputSize  = B * C * H * W * sizeof(float);  // input features map is C
    int outputSize = B * M * H_out * W_out * sizeof(float); // output feature map is M
    int maskSize = M * C * K * K * sizeof(float); // C * M filter Maps of size K*K

    cudaMalloc((void **) device_x_ptr, inputSize);
    cudaMalloc((void **) device_y_ptr, outputSize);
    cudaMalloc((void **) device_k_ptr, maskSize);

    // Copy Inout data to device
    cudaMemcpy(*device_x_ptr, host_x, inputSize, cudaMemcpyHostToDevice);
    // Copy Mask data to device
    cudaMemcpy(*device_k_ptr, host_k, maskSize, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface2::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int H_grid = ceil(1.0*H_out / TILE_WIDTH);
    int W_grid = ceil(1.0*W_out / TILE_WIDTH);
    int Z = H_grid * W_grid;

    // Block dimensions = #of threads in the block
    dim3 numThreadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Grid Dimension = #of Blocks: Batch Size * Num_Output_Features *
    dim3 numBlocksInGrid(B, M, Z);


    //launch the kernel
    conv_forward_kernel_op<<<numBlocksInGrid, numThreadsPerBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}


__host__ void GPUInterface2::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int outputSize = B * M * H_out * W_out * sizeof(float);

    cudaMemcpy(host_y, device_y, outputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
}

