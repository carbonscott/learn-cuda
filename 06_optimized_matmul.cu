// Matrix multiplication with optimizations, similar to those in transformer code

#include <stdio.h>
#include <cublas_v2.h>

// Simple matrix multiplication kernel (naïve version)
__global__ void matmul_naive(float *c, float *a, float *b, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = a[row * K] * b[col];
        for (int i = 1; i < K; i++) {
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

// Optimized matrix multiplication kernel using shared memory
__global__ void matmul_shared(float *c, float *a, float *b, int M, int N, int K) {
    // Tile size (must be known at compile time)
    const int TILE_SIZE = 16;

    // Shared memory for tiles of A and B
    __shared__ float a_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float b_shared[TILE_SIZE][TILE_SIZE];

    // Thread coordinates within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Coordinates in the result matrix
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for the dot product
    float sum = 0.0f;

    // Iterate over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            a_shared[ty][tx] = a[row * K + t * TILE_SIZE + tx];
        } else {
            a_shared[ty][tx] = 0.0f;
        }

        if (t * TILE_SIZE + ty < K && col < N) {
            b_shared[ty][tx] = b[(t * TILE_SIZE + ty) * N + col];
        } else {
            b_shared[ty][tx] = 0.0f;
        }

        // Make sure tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += a_shared[ty][i] * b_shared[i][tx];
        }

        // Make sure all threads are done with the tiles
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

// Matrix multiplication using cuBLAS (most optimized)
void matmul_cublas(float *c, float *a, float *b, int M, int N, int K, cublasHandle_t handle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Note: cuBLAS uses column-major order, so we're computing B*A instead of A*B
    // M, N are swapped in the call because we're computing the transpose
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                b, N,   // B with leading dimension N
                a, K,   // A with leading dimension K
                &beta,
                c, N);  // C with leading dimension N
}

int main() {
    // Matrix dimensions
    int M = 1024;  // A: M x K
    int N = 1024;  // B: K x N, C: M x N
    int K = 1024;

    size_t bytes_a = M * K * sizeof(float);
    size_t bytes_b = K * N * sizeof(float);
    size_t bytes_c = M * N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes_a);
    float *h_b = (float*)malloc(bytes_b);
    float *h_c_naive = (float*)malloc(bytes_c);
    float *h_c_shared = (float*)malloc(bytes_c);
    float *h_c_cublas = (float*)malloc(bytes_c);

    // Initialize matrices with simple values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_a[i * K + j] = 1.0f;  // All 1's for simplicity
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_b[i * N + j] = 1.0f;  // All 1's for simplicity
        }
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c_naive, *d_c_shared, *d_c_cublas;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c_naive, bytes_c);
    cudaMalloc(&d_c_shared, bytes_c);
    cudaMalloc(&d_c_cublas, bytes_c);

    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configure kernel execution parameters
    dim3 block_dim(16, 16);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                 (M + block_dim.y - 1) / block_dim.y);

    printf("Grid dimensions: (%d, %d)\n", grid_dim.x, grid_dim.y);
    printf("Block dimensions: (%d, %d)\n", block_dim.x, block_dim.y);

    // Execute and time the naive version
    printf("\nRunning naive matrix multiplication...\n");
    cudaEventRecord(start);
    matmul_naive<<<grid_dim, block_dim>>>(d_c_naive, d_a, d_b, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_naive = 0;
    cudaEventElapsedTime(&milliseconds_naive, start, stop);
    printf("Naive execution time: %.3f ms\n", milliseconds_naive);

    // Execute and time the shared memory version
    printf("\nRunning shared memory matrix multiplication...\n");
    cudaEventRecord(start);
    matmul_shared<<<grid_dim, block_dim>>>(d_c_shared, d_a, d_b, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_shared = 0;
    cudaEventElapsedTime(&milliseconds_shared, start, stop);
    printf("Shared memory execution time: %.3f ms\n", milliseconds_shared);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Execute and time the cuBLAS version
    printf("\nRunning cuBLAS matrix multiplication...\n");
    cudaEventRecord(start);
    matmul_cublas(d_c_cublas, d_a, d_b, M, N, K, handle);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_cublas = 0;
    cudaEventElapsedTime(&milliseconds_cublas, start, stop);
    printf("cuBLAS execution time: %.3f ms\n", milliseconds_cublas);

    // Copy results back to host
    cudaMemcpy(h_c_naive, d_c_naive, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_shared, d_c_shared, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_cublas, d_c_cublas, bytes_c, cudaMemcpyDeviceToHost);

    // Verify results
    bool naive_correct = true;
    bool shared_correct = true;
    bool cublas_correct = true;

    // For large matrices, just check a few values
    for (int i = 0; i < 10; i++) {
        float expected = K;  // Since all elements are 1, result should be K

        if (fabs(h_c_naive[i] - expected) > 1e-5) {
            naive_correct = false;
            printf("Naive result incorrect at index %d: %f vs %f\n", i, h_c_naive[i], expected);
            break;
        }

        if (fabs(h_c_shared[i] - expected) > 1e-5) {
            shared_correct = false;
            printf("Shared memory result incorrect at index %d: %f vs %f\n", i, h_c_shared[i], expected);
            break;
        }

        if (fabs(h_c_cublas[i] - expected) > 1e-5) {
            cublas_correct = false;
            printf("cuBLAS result incorrect at index %d: %f vs %f\n", i, h_c_cublas[i], expected);
            break;
        }
    }

    // Print results summary
    printf("\nResults Summary:\n");
    printf("Naive Implementation: %s, %.3f ms\n",
           naive_correct ? "CORRECT" : "INCORRECT", milliseconds_naive);
    printf("Shared Memory Implementation: %s, %.3f ms\n",
           shared_correct ? "CORRECT" : "INCORRECT", milliseconds_shared);
    printf("cuBLAS Implementation: %s, %.3f ms\n",
           cublas_correct ? "CORRECT" : "INCORRECT", milliseconds_cublas);

    if (naive_correct && shared_correct && cublas_correct) {
        printf("\nSpeedup Shared vs Naive: %.2fx\n", milliseconds_naive / milliseconds_shared);
        printf("Speedup cuBLAS vs Naive: %.2fx\n", milliseconds_naive / milliseconds_cublas);
        printf("Speedup cuBLAS vs Shared: %.2fx\n", milliseconds_shared / milliseconds_cublas);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c_naive);
    free(h_c_shared);
    free(h_c_cublas);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_naive);
    cudaFree(d_c_shared);
    cudaFree(d_c_cublas);

    cublasDestroy(handle);

    return 0;
}

/*
Compilation and Run:
$ nvcc -o optimized_matmul 06_optimized_matmul.cu -lcublas
$ ./optimized_matmul

Experiment Ideas:
1. Try different matrix sizes to see how performance scales
2. Change the tile size in the shared memory implementation
3. Test with different data patterns instead of all 1's
4. Try a more complex formula for matrix initialization
*/
