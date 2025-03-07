// Understanding the CUDA thread hierarchy (grids, blocks, threads)

#include <stdio.h>

// This kernel demonstrates the thread hierarchy
__global__ void thread_hierarchy_kernel() {
    // Calculate unique thread ID
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Print 3D info about this thread
    printf("Thread ID: %d | Block: (%d, %d, %d) | Thread: (%d, %d, %d)\n",
            thread_id,
            blockIdx.x, blockIdx.y, blockIdx.z,
            threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    // Launch with a 2D grid (2x2 blocks) and 2D blocks (4x2 threads)
    dim3 grid_dim(2, 2, 1);    // 2x2 grid of blocks (4 blocks total)
    dim3 block_dim(4, 2, 1);   // 4x2 threads per block (8 threads per block)

    printf("Launching kernel with:\n");
    printf("  Grid dimensions: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("  Block dimensions: (%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);
    printf("  Total threads: %d\n", grid_dim.x * grid_dim.y * grid_dim.z * 
                                   block_dim.x * block_dim.y * block_dim.z);

    // Launch kernel
    thread_hierarchy_kernel<<<grid_dim, block_dim>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}

/* 
Compilation and Run:
$ nvcc -o thread_hierarchy 03_thread_hierarchy.cu
$ ./thread_hierarchy

Experiment Ideas:
1. Try different grid and block dimensions (1D, 2D, 3D)
2. See what happens when you create more threads than your GPU supports
3. Calculate thread indices manually and compare with the output
4. Experiment with different ways to assign work to threads
*/
