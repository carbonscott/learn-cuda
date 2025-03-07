// Understanding memory coalescing for optimal GPU performance

#include <stdio.h>
#include <cuda_runtime.h>

// Define array size
#define N 16000000  // 16M elements
#define THREADS_PER_BLOCK 256

// Kernel demonstrating coalesced memory access
__global__ void coalesced_access(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Threads with consecutive IDs access consecutive memory locations
        // This is COALESCED access - optimal performance
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel demonstrating non-coalesced (strided) memory access
__global__ void strided_access(float* input, float* output, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx * stride < N) {
        // Threads with consecutive IDs access memory locations that are 'stride' elements apart
        // This is NON-COALESCED access - poor performance
        output[idx] = input[idx * stride] * 2.0f;
    }
}

// Kernel demonstrating Array of Structures (AoS) - typically not coalesced
__global__ void aos_access(float4* input, float4* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N/4) {
        // Each thread processes one float4 structure
        float4 val = input[idx];

        // Process each component
        val.x *= 2.0f;
        val.y *= 2.0f;
        val.z *= 2.0f;
        val.w *= 2.0f;

        output[idx] = val;
    }
}

// Kernel demonstrating Structure of Arrays (SoA) - coalesced
__global__ void soa_access(float* input_x, float* input_y, float* input_z, float* input_w,
                          float* output_x, float* output_y, float* output_z, float* output_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N/4) {
        // Each thread accesses the same component from different structures
        // This results in coalesced memory access
        output_x[idx] = input_x[idx] * 2.0f;
        output_y[idx] = input_y[idx] * 2.0f;
        output_z[idx] = input_z[idx] * 2.0f;
        output_w[idx] = input_w[idx] * 2.0f;
    }
}

// Kernel demonstrating matrix transpose with non-coalesced memory access
__global__ void matrix_transpose_naive(float* input, float* output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < width) {
        // Reading is coalesced (consecutive threads read consecutive elements)
        // Writing is NOT coalesced (consecutive threads write to elements that are 'width' apart)
        output[x * width + y] = input[y * width + x];
    }
}

// Kernel demonstrating matrix transpose with shared memory to improve coalescing
__global__ void matrix_transpose_shared(float* input, float* output, int width) {
    __shared__ float tile[32][32];  // Shared memory tile

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Local indices within the block
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    if (x < width && y < width) {
        // Load data from input (coalesced read)
        tile[local_y][local_x] = input[y * width + x];
    }

    __syncthreads();  // Ensure all threads have loaded data

    // Determine output coordinates (transposed)
    int out_x = blockIdx.y * blockDim.y + local_x;
    int out_y = blockIdx.x * blockDim.x + local_y;

    if (out_x < width && out_y < width) {
        // Store to output (now coalesced write because we're reading from shared memory transposed)
        output[out_y * width + out_x] = tile[local_x][local_y];
    }
}

// Kernel demonstrating vectorized memory access (most optimal)
__global__ void vectorized_access(float4* input, float4* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N/4) {
        // Load 4 floats at once - this is both coalesced AND uses vector load instructions
        float4 val = input[idx];

        // Process 4 values at once
        val.x *= 2.0f;
        val.y *= 2.0f;
        val.z *= 2.0f;
        val.w *= 2.0f;

        // Store 4 floats at once
        output[idx] = val;
    }
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    float4 *h_input4, *h_output4;
    float4 *d_input4, *d_output4;
    float *d_input_x, *d_input_y, *d_input_z, *d_input_w;
    float *d_output_x, *d_output_y, *d_output_z, *d_output_w;

    // Allocate host memory
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Calculate grid dimensions
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Testing memory access patterns with array size: %d\n", N);
    printf("-------------------------------------------------------\n");

    // --------------------------------------------------------------------
    // Test 1: Coalesced access
    cudaEventRecord(start);

    coalesced_access<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float coalesced_time = 0;
    cudaEventElapsedTime(&coalesced_time, start, stop);

    // --------------------------------------------------------------------
    // Test 2: Strided access (non-coalesced)
    int stride = 32;  // Stride of 32 elements - terrible for coalescing!
    blocks = ((N / stride) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start);

    strided_access<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, stride);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float strided_time = 0;
    cudaEventElapsedTime(&strided_time, start, stop);

    // --------------------------------------------------------------------
    // Test 3: AoS vs SoA
    // Allocate memory for AoS test
    h_input4 = (float4*)malloc(N/4 * sizeof(float4));
    h_output4 = (float4*)malloc(N/4 * sizeof(float4));

    // Initialize input for AoS test
    for (int i = 0; i < N/4; i++) {
        h_input4[i].x = (float)(i * 4);
        h_input4[i].y = (float)(i * 4 + 1);
        h_input4[i].z = (float)(i * 4 + 2);
        h_input4[i].w = (float)(i * 4 + 3);
    }

    // Allocate device memory for AoS test
    cudaMalloc(&d_input4, N/4 * sizeof(float4));
    cudaMalloc(&d_output4, N/4 * sizeof(float4));

    // Copy input data to device for AoS test
    cudaMemcpy(d_input4, h_input4, N/4 * sizeof(float4), cudaMemcpyHostToDevice);

    // Allocate device memory for SoA test
    cudaMalloc(&d_input_x, N/4 * sizeof(float));
    cudaMalloc(&d_input_y, N/4 * sizeof(float));
    cudaMalloc(&d_input_z, N/4 * sizeof(float));
    cudaMalloc(&d_input_w, N/4 * sizeof(float));
    cudaMalloc(&d_output_x, N/4 * sizeof(float));
    cudaMalloc(&d_output_y, N/4 * sizeof(float));
    cudaMalloc(&d_output_z, N/4 * sizeof(float));
    cudaMalloc(&d_output_w, N/4 * sizeof(float));

    // Extract components from AoS and copy to device for SoA test
    float *h_comp_x = (float*)malloc(N/4 * sizeof(float));
    float *h_comp_y = (float*)malloc(N/4 * sizeof(float));
    float *h_comp_z = (float*)malloc(N/4 * sizeof(float));
    float *h_comp_w = (float*)malloc(N/4 * sizeof(float));

    for (int i = 0; i < N/4; i++) {
        h_comp_x[i] = h_input4[i].x;
        h_comp_y[i] = h_input4[i].y;
        h_comp_z[i] = h_input4[i].z;
        h_comp_w[i] = h_input4[i].w;
    }

    cudaMemcpy(d_input_x, h_comp_x, N/4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_y, h_comp_y, N/4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_z, h_comp_z, N/4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_w, h_comp_w, N/4 * sizeof(float), cudaMemcpyHostToDevice);

    // Test AoS
    blocks = (N/4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start);

    aos_access<<<blocks, THREADS_PER_BLOCK>>>(d_input4, d_output4);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float aos_time = 0;
    cudaEventElapsedTime(&aos_time, start, stop);

    // Test SoA
    cudaEventRecord(start);

    soa_access<<<blocks, THREADS_PER_BLOCK>>>(d_input_x, d_input_y, d_input_z, d_input_w,
                                         d_output_x, d_output_y, d_output_z, d_output_w);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float soa_time = 0;
    cudaEventElapsedTime(&soa_time, start, stop);

    // --------------------------------------------------------------------
    // Test 4: Matrix transpose

    // Matrix size (we'll use a square matrix)
    int width = 4096;
    int matrix_size = width * width * sizeof(float);

    // Allocate memory for matrix transpose test
    float *h_matrix = (float*)malloc(matrix_size);
    float *h_transposed = (float*)malloc(matrix_size);
    float *d_matrix, *d_transposed;

    // Initialize matrix
    for (int i = 0; i < width * width; i++) {
        h_matrix[i] = (float)i;
    }

    // Allocate device memory for matrix transpose test
    cudaMalloc(&d_matrix, matrix_size);
    cudaMalloc(&d_transposed, matrix_size);

    // Copy matrix to device
    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    // Grid and block dimensions for matrix transpose
    dim3 block_dim(32, 32);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 
                  (width + block_dim.y - 1) / block_dim.y);

    // Test naive transpose (non-coalesced writes)
    cudaEventRecord(start);

    matrix_transpose_naive<<<grid_dim, block_dim>>>(d_matrix, d_transposed, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naive_transpose_time = 0;
    cudaEventElapsedTime(&naive_transpose_time, start, stop);

    // Test shared memory transpose (improved coalescing)
    cudaEventRecord(start);

    matrix_transpose_shared<<<grid_dim, block_dim>>>(d_matrix, d_transposed, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float shared_transpose_time = 0;
    cudaEventElapsedTime(&shared_transpose_time, start, stop);

    // --------------------------------------------------------------------
    // Test 5: Vectorized access
    blocks = (N/4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start);

    vectorized_access<<<blocks, THREADS_PER_BLOCK>>>(d_input4, d_output4);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float vectorized_time = 0;
    cudaEventElapsedTime(&vectorized_time, start, stop);

    // --------------------------------------------------------------------
    // Print results
    printf("Test Results:\n");
    printf("1. Coalesced access:         %.3f ms\n", coalesced_time);
    printf("2. Strided access (stride=%d): %.3f ms\n", stride, strided_time);
    printf("   Slowdown vs. coalesced:   %.2fx\n", strided_time / coalesced_time);
    printf("\n");
    printf("3. Array of Structures:      %.3f ms\n", aos_time);
    printf("   Structure of Arrays:      %.3f ms\n", soa_time);
    printf("   AoS vs. SoA speedup:      %.2fx\n", aos_time / soa_time);
    printf("\n");
    printf("4. Matrix transpose (naive): %.3f ms\n", naive_transpose_time);
    printf("   Matrix transpose (shared): %.3f ms\n", shared_transpose_time);
    printf("   Shared memory speedup:    %.2fx\n", naive_transpose_time / shared_transpose_time);
    printf("\n");
    printf("5. Vectorized access:        %.3f ms\n", vectorized_time);
    printf("   Speedup vs. coalesced:    %.2fx\n", coalesced_time / vectorized_time);

    // --------------------------------------------------------------------
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input4);
    cudaFree(d_output4);
    cudaFree(d_input_x);
    cudaFree(d_input_y);
    cudaFree(d_input_z);
    cudaFree(d_input_w);
    cudaFree(d_output_x);
    cudaFree(d_output_y);
    cudaFree(d_output_z);
    cudaFree(d_output_w);
    cudaFree(d_matrix);
    cudaFree(d_transposed);

    free(h_input);
    free(h_output);
    free(h_input4);
    free(h_output4);
    free(h_comp_x);
    free(h_comp_y);
    free(h_comp_z);
    free(h_comp_w);
    free(h_matrix);
    free(h_transposed);

    return 0;
}

/* 
Compilation and Run:
$ nvcc -o memory_coalescing 05_memory_coalescing.cu
$ ./memory_coalescing

Memory Coalescing Principles:

1. What is Memory Coalescing?
   - Memory coalescing is when threads in the same warp access contiguous memory
   - The GPU can combine multiple memory requests into a single transaction
   - Coalesced access can be 10-30x faster than non-coalesced

2. Rules for Coalesced Access:
   - Threads with consecutive IDs should access consecutive memory locations
   - Memory should be properly aligned (e.g., 128-byte boundary for optimal access)
   - Access patterns should generally follow the structure of the thread indexing

3. Common Non-Coalesced Patterns to Avoid:
   - Strided access (threads access memory with gaps between elements)
   - Misaligned access (starting address not aligned to appropriate boundary)
   - Irregular access patterns (threads access random memory locations)
   - Array of Structures (AoS) instead of Structure of Arrays (SoA)
   - Column-major access in row-major stored arrays (or vice versa)

4. Advanced Techniques:
   - Use shared memory to reorganize non-coalesced patterns
   - Use vectorized loads/stores (float4, int4) for higher throughput
   - Transpose matrices in shared memory when necessary
   - Use padding to ensure alignment when needed
   - Consider SoA layout instead of AoS for better coalescing

Experiment Ideas:
1. Try different stride values to see the impact on performance
2. Experiment with different matrix sizes for the transpose test
3. Try implementing a 2D stencil computation with and without coalescing
4. Analyze the impact of memory coalescing on different GPU architectures
5. Combine memory coalescing with other optimization techniques
*/
