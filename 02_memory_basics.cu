// Understanding memory allocation and transfer between CPU and GPU

#include <stdio.h>

// Simple vector addition kernel
__global__ void vector_add(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Array size
    int N = 1000;
    size_t size = N * sizeof(float);

    // Allocate memory for host arrays
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory for device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy host arrays to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel with enough threads to cover the array
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the result (just check a few values)
    printf("Checking results...\n");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", (int)h_a[i], (int)h_b[i], (int)h_c[i]);
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

/* 
Compilation and Run:
$ nvcc -o memory_basics 02_memory_basics.cu
$ ./memory_basics

Experiment Ideas:
1. Change the array size to see how it affects performance
2. Change threadsPerBlock to different values (32, 64, 128, 512, 1024)
3. Try to measure the time it takes for memory transfers vs. computation
4. Try allocating too much memory to see how errors are handled
*/
