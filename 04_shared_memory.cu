// Demonstrating the use of shared memory for faster access

#include <stdio.h>
#include <cuda_runtime.h>

// Matrix multiplication using global memory only (slow)
__global__ void matrix_mul_global(float *a, float *b, float *c, int width) {
    // Calculate row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            // Global memory access - slower
            sum += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = sum;
    }
}

// Matrix multiplication using shared memory (faster)
__global__ void matrix_mul_shared(float *a, float *b, float *c, int width) {
    // Define shared memory tiles
    __shared__ float s_a[16][16];
    __shared__ float s_b[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (width + 15) / 16; tile++) {
        // Load data into shared memory
        if (row < width && tile * 16 + tx < width) {
            s_a[ty][tx] = a[row * width + tile * 16 + tx];
        } else {
            s_a[ty][tx] = 0.0f;
        }

        if (col < width && tile * 16 + ty < width) {
            s_b[ty][tx] = b[(tile * 16 + ty) * width + col];
        } else {
            s_b[ty][tx] = 0.0f;
        }

        // Wait for all threads to finish loading
        __syncthreads();

        // Compute partial sum for this tile
        for (int i = 0; i < 16; i++) {
            sum += s_a[ty][i] * s_b[i][tx];
        }

        // Wait for all threads to finish computing
        __syncthreads();
    }

    // Write result
    if (row < width && col < width) {
        c[row * width + col] = sum;
    }
}

int main() {
    int width = 64; // Small matrix for demo
    size_t size = width * width * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c_global = (float*)malloc(size);
    float *h_c_shared = (float*)malloc(size);

    // Initialize matrices with simple values
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            h_a[i * width + j] = 1.0f;
            h_b[i * width + j] = 2.0f;
        }
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c_global, *d_c_shared;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_global, size);
    cudaMalloc(&d_c_shared, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch global memory kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 
                 (width + block_dim.y - 1) / block_dim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time global memory version
    cudaEventRecord(start);
    matrix_mul_global<<<grid_dim, block_dim>>>(d_a, d_b, d_c_global, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_global = 0;
    cudaEventElapsedTime(&milliseconds_global, start, stop);

    // Time shared memory version
    cudaEventRecord(start);
    matrix_mul_shared<<<grid_dim, block_dim>>>(d_a, d_b, d_c_shared, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_shared = 0;
    cudaEventElapsedTime(&milliseconds_shared, start, stop);

    // Copy results back
    cudaMemcpy(h_c_global, d_c_global, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_shared, d_c_shared, size, cudaMemcpyDeviceToHost);

    // Verify results
    bool correct = true;
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_c_global[i] - h_c_shared[i]) > 1e-5) {
            printf("Results don't match at position %d: %f vs %f\n", 
                   i, h_c_global[i], h_c_shared[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Results match!\n");
        printf("Global memory version took: %.3f ms\n", milliseconds_global);
        printf("Shared memory version took: %.3f ms\n", milliseconds_shared);
        printf("Speedup: %.2fx\n", milliseconds_global / milliseconds_shared);
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_global);
    free(h_c_shared);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_global);
    cudaFree(d_c_shared);

    return 0;
}

/* 
Compilation and Run:
$ nvcc -o shared_memory 04_shared_memory.cu
$ ./shared_memory

Experiment Ideas:
1. Change the matrix size and observe performance differences
2. Modify the shared memory tile size (currently 16x16)
3. Compare performance with different block dimensions
4. Add error checking with cudaGetLastError() to catch subtle bugs
*/
