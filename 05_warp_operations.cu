// Demonstrating warp-level operations and reductions

#include <stdio.h>

// CUDA warp size is typically 32
#define WARP_SIZE 32

// Simple reduction using shared memory
__global__ void reduction_shared(float *input, float *output, int n) {
    // Shared memory for the block
    __shared__ float sdata[256];

    // Each thread loads one element
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Advanced reduction using warp intrinsics
__global__ void reduction_warp(float *input, float *output, int n) {
    // Shared memory for the block
    __shared__ float sdata[256];

    // Each thread loads one element
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Using warp-level intrinsics for the final reduction
    if (tid < 32) {
        // Perform the warp reduce without any __syncthreads()
        volatile float *smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }

    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Modern warp reduction using intrinsics
__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduction_modern(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element
    float sum = (gid < n) ? input[gid] : 0;

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // First thread in each warp writes result
    if (tid % WARP_SIZE == 0) {
        atomicAdd(&output[blockIdx.x], sum);
    }
}

int main() {
    int n = 1000000;  // Size of the array
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(sizeof(float));

    // Initialize input data
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // Sum will be n
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float) * 1024);  // Space for block results

    // Copy data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Setup execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;  // Limit for this example

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Zero out the output array
    cudaMemset(d_output, 0, sizeof(float) * 1024);

    // Launch shared memory reduction kernel
    cudaEventRecord(start);
    reduction_shared<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_shared = 0;
    cudaEventElapsedTime(&milliseconds_shared, start, stop);

    // Copy result back to host for verification
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Shared Memory Reduction Result: %f\n", h_output[0]);
    printf("Shared Memory Time: %.3f ms\n", milliseconds_shared);

    // Zero out the output array
    cudaMemset(d_output, 0, sizeof(float) * 1024);

    // Launch warp intrinsic reduction kernel
    cudaEventRecord(start);
    reduction_warp<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_warp = 0;
    cudaEventElapsedTime(&milliseconds_warp, start, stop);

    // Copy result back to host for verification
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Warp Intrinsic Reduction Result: %f\n", h_output[0]);
    printf("Warp Intrinsic Time: %.3f ms\n", milliseconds_warp);

    // Zero out the output array
    cudaMemset(d_output, 0, sizeof(float) * 1024);

    // Launch modern reduction kernel
    cudaEventRecord(start);
    reduction_modern<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_modern = 0;
    cudaEventElapsedTime(&milliseconds_modern, start, stop);

    // Copy result back to host for verification
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Modern Reduction Result: %f\n", h_output[0]);
    printf("Modern Reduction Time: %.3f ms\n", milliseconds_modern);

    printf("Expected result: %d\n", n);

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

/* 
Compilation and Run:
$ nvcc -o warp_operations 05_warp_operations.cu
$ ./warp_operations

Experiment Ideas:
1. Try different array sizes and block sizes
2. Experiment with other warp primitives like __shfl_xor_sync
3. Implement a two-level reduction for very large arrays
4. Try other operations like min, max, or average
*/
