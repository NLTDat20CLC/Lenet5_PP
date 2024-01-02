#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define UNROLL_FACTOR 8  // You can adjust the unroll factor based on performance testing

__global__ void conv_forward_kernel_op(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);


    int b = blockIdx.x;           // batch number
    int m = blockIdx.y;           // output feature
    int tile_row = blockIdx.z / W_grid;  // tile row
    int tile_col = blockIdx.z % W_grid;  // tile column

    int h_start = tile_row * TILE_WIDTH;
    int w_start = tile_col * TILE_WIDTH;

    int h = h_start + threadIdx.y; // row of the image matrix within the tile
    int w = w_start + threadIdx.x; // col of the image matrix within the tile

    __shared__ float shared_x[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_k[TILE_WIDTH][TILE_WIDTH];

    float accum[UNROLL_FACTOR] = {0.0f};

    for (int c = 0; c < C; c++)
    {
        // Load tile from global memory to shared memory
        shared_x[threadIdx.y][threadIdx.x] = x4d(b, c, h, w);
        shared_k[threadIdx.y][threadIdx.x] = k4d(m, c, h - h_start, w - w_start);

        __syncthreads();

        // Unrolled loop for computing partial result
#pragma unroll
        for (int p = 0; p < TILE_WIDTH; p += UNROLL_FACTOR)
        {
#pragma unroll
            for (int q = 0; q < UNROLL_FACTOR; ++q)
            {
                accum[q] += shared_x[threadIdx.y][p + q] * shared_k[p + q][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (h < H_out && w < W_out)
    {
#pragma unroll
        for (int q = 0; q < UNROLL_FACTOR; ++q)
        {
            y4d(b, m, h, w) += accum[q];
        }
    }
    #undef y4d
    #undef x4d
    #undef k4d
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
