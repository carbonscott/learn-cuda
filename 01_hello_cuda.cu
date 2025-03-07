// The simplest CUDA program to help understand the basic structure

#include <stdio.h>

// This is a kernel function that runs on the GPU
__global__ void hello_kernel() {
    // Get the unique ID for this thread
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread prints its ID
    printf("Hello from GPU thread %d\n", thread_id);
}

int main() {
    // Print from the CPU (host)
    printf("Hello from CPU!\n");

    // Launch the kernel with 1 block of 10 threads
    // The <<<1, 10>>> syntax specifies the grid and block dimensions
    hello_kernel<<<1, 10>>>();

    // Wait for GPU to finish before accessing results from host
    cudaDeviceSynchronize();

    printf("All GPU threads have completed!\n");

    return 0;
}

/* 
Compilation and Run:
$ nvcc -o hello_cuda 01_hello_cuda.cu
$ ./hello_cuda

Experiment Ideas:
1. Change the number of threads (second parameter in <<<1, 10>>>)
2. Change to use multiple blocks (first parameter in <<<1, 10>>>)
3. See what happens with different grid/block combinations
*/
