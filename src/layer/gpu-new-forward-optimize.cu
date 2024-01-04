#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 12 // You can adjust the tile width based on performance testing
#define UNROLL_FACTOR 8  // You can adjust the unroll factor based on performance testing

//fused the memory accesses for the input and kernel matrices
__global__ void conv_forward_kernel_op(float *__restrict__ y, const float *__restrict__ x, const float *__restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

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

    __shared__ float shared_k[TILE_WIDTH][TILE_WIDTH * UNROLL_FACTOR];  // Increased shared memory size for loop unrolling

    float accum[UNROLL_FACTOR] = {0.0f};

    for (int c_start = 0; c_start < C; c_start += TILE_WIDTH)
    {
        // Load tile from global memory to shared memory for input channels with loop unrolling
        for (int q = 0; q < TILE_WIDTH; q += UNROLL_FACTOR)
        {
            int c = c_start + threadIdx.x + q;  // Adjusted index calculation
            int x_index = b * (C * H * W) + c * (H * W) + (h - h_start) * W + (w - w_start);
            int k_index = m * (C * K * K) + c * (K * K);

            // Loop unrolling
#pragma unroll
            for (int i = 0; i < UNROLL_FACTOR; ++i)
            {
                shared_k[threadIdx.y][q + i] = x[x_index + i * W];  // Adjusted shared memory indexing
                accum[i] += shared_k[threadIdx.y][q + i] * k[k_index + i];
            }
        }

        __syncthreads();
    }

    if (h < H_out && w < W_out)
    {
#pragma unroll
        for (int q = 0; q < UNROLL_FACTOR; ++q)
        {
            atomicAdd(&y4d(b, m, h, w), accum[q]);
        }
    }

    #undef y4d
}
	
__host__ void GPUInterface2::conv_forward_gpu_prolog(const float *__restrict__ host_y, const float *__restrict__ host_x, const float *__restrict__ host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
     const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int inputSize = B * C * H * W * sizeof(float);
    int outputSize = B * M * H_out * W_out * sizeof(float);
    int maskSize = M * C * K * K * sizeof(float);

    cudaMalloc((void **)device_x_ptr, inputSize);
    cudaMalloc((void **)device_y_ptr, outputSize);
    cudaMalloc((void **)device_k_ptr, maskSize);

    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
   

    // Asynchronously copy data to the GPU using streams
    cudaMemcpyAsync(*device_x_ptr, host_x, inputSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(*device_k_ptr, host_k, maskSize, cudaMemcpyHostToDevice, stream2);

    // Synchronize to ensure memory copies are completed before launching the kernel
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Destroy the streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

}


__host__ void GPUInterface2::conv_forward_gpu(float *__restrict__ device_y, const float *__restrict__ device_x, const float *__restrict__ device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
     const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int Z = H_grid * W_grid;

    dim3 numThreadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 numBlocksInGrid(B, M, Z);

    // Create a stream for the kernel execution
    cudaStream_t kernelStream;
    cudaStreamCreate(&kernelStream);

    // Launch the kernel asynchronously in the specified stream
    conv_forward_kernel_op<<<numBlocksInGrid, numThreadsPerBlock, 0, kernelStream>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    // Destroy the stream
    cudaStreamDestroy(kernelStream);
}


__host__ void GPUInterface2::conv_forward_gpu_epilog(float *__restrict__ host_y, float *__restrict__ device_y, float *__restrict__ device_x, float *__restrict__ device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int outputSize = B * M * H_out * W_out * sizeof(float);

    // Create a stream for the final data transfer
    cudaStream_t finalTransferStream;
    cudaStreamCreate(&finalTransferStream);

    // Asynchronously copy the result back to the host using the final transfer stream
    cudaMemcpyAsync(host_y, device_y, outputSize, cudaMemcpyDeviceToHost, finalTransferStream);

    // Destroy the stream
    cudaStreamDestroy(finalTransferStream);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
}
