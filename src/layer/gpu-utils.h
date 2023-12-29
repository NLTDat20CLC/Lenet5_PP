#ifndef SRC_LAYER_GPU_UTILS_H
#define SRC_LAYER_GPU_UTILS_H
#include <cuda_runtime.h>

// Use this class to define GPU functions that students don't need access to. 
class GPU_Utils
{
    public:
    /* For creating a dummy kernel call so that we can distinguish between kernels launched for different layers
     * in the Nsight Compute CLI for measuring per layer Op Times
     */
    void insert_post_barrier_kernel();
    // For inserting a marker visible in Nsys so that we can time total student function time
    void insert_pre_barrier_kernel();
};

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
#endif 