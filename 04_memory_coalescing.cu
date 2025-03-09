// Understanding memory coalescing for optimal GPU performance

#include <stdio.h>
#include <cuda_runtime.h>

// Define array size
#define N (32 * 1024 * 1024)  // 32M elements
#define THREADS_PER_BLOCK 256
#define BLOCKS ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

// Utility function for checking CUDA errors
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

// =========================================================================
// KERNEL 1: Perfectly coalesced memory access
// All threads in a warp access consecutive memory locations
// =========================================================================
__global__ void coalesced_access(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Each thread accesses consecutive memory locations
        // All threads in a warp (32 threads) access a contiguous chunk of memory
        // This allows the GPU hardware to combine these accesses into fewer transactions
        output[idx] = input[idx] * 2.0f;
    }
}

// =========================================================================
// KERNEL 2: Strided memory access (non-coalesced)
// Threads access memory with a stride, causing inefficient memory access patterns
// =========================================================================
__global__ void strided_access(float* input, float* output, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Calculate strided access pattern
        // This causes threads within the same warp to access memory locations far apart
        // The hardware cannot coalesce these into fewer transactions
        int strided_idx = (idx * stride) % N;
        output[idx] = input[strided_idx] * 2.0f;
    }
}

// =========================================================================
// KERNEL 3: Bank conflict demonstration with shared memory
// =========================================================================
__global__ void shared_memory_bank_conflicts(float* input, float* output, int stride) {
    __shared__ float shared_data[THREADS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory - coalesced global memory access
    if (idx < N) {
        shared_data[threadIdx.x] = input[idx];
    }
    __syncthreads();

    // Access shared memory with potential bank conflicts depending on stride
    if (idx < N) {
        int bank_idx = (threadIdx.x * stride) % THREADS_PER_BLOCK;
        float value = shared_data[bank_idx];
        output[idx] = value * 2.0f;
    }
}

// =========================================================================
// KERNEL 4: Misaligned access
// Starting from a non-aligned address, causing partial coalescing
// =========================================================================
__global__ void misaligned_access(float* input, float* output, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N - offset) {
        // Offset causes misalignment at the warp level
        output[idx] = input[idx + offset] * 2.0f;
    }
}

// =========================================================================
// KERNEL 5: Structure of Arrays (SoA) vs Array of Structures (AoS)
// =========================================================================
typedef struct {
    float x, y, z, w;  // 4 floats representing a vector4
} Vector4;

__global__ void aos_access(Vector4* input, Vector4* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N/4) {  // We have N/4 Vector4 elements
        // Each thread processes one Vector4
        // BUT: when the 32 threads in a warp access their x components,
        // they are accessing non-contiguous memory locations
        Vector4 v = input[idx];
        v.x *= 2.0f;
        v.y *= 2.0f;
        v.z *= 2.0f;
        v.w *= 2.0f;
        output[idx] = v;
    }
}

__global__ void soa_access(float* x, float* y, float* z, float* w,
                          float* x_out, float* y_out, float* z_out, float* w_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N/4) {
        // Each thread processes one element from each array
        // When the 32 threads in a warp access their x components,
        // they're accessing contiguous memory locations
        x_out[idx] = x[idx] * 2.0f;
        y_out[idx] = y[idx] * 2.0f;
        z_out[idx] = z[idx] * 2.0f;
        w_out[idx] = w[idx] * 2.0f;
    }
}

int main() {
    size_t size = N * sizeof(float);
    float *h_input, *h_output;
    float *d_input, *d_output;

    // Allocate host memory
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, size));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    printf("=================================================================\n");
    printf("CUDA Memory Coalescing Demonstration\n");
    printf("Array size: %d elements (%zu MB)\n", N, size / (1024 * 1024));
    printf("=================================================================\n\n");

    // =========================================================================
    // Test 1: Coalesced access
    // =========================================================================
    printf("Running Kernel 1: Coalesced access\n");

    cudaEventRecord(start);
    coalesced_access<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output);
    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Effective bandwidth: %.2f GB/s\n\n",
           (2 * size) / (milliseconds * 1.0e6));

    // =========================================================================
    // Test 2: Strided access pattern
    // =========================================================================
    printf("Running Kernel 2: Strided access (stride = 32)\n");

    // Stride of 32 means threads in the same warp access memory 32 elements apart
    // This is extremely inefficient for memory coalescing
    int stride = 32;

    cudaEventRecord(start);
    strided_access<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, stride);
    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Effective bandwidth: %.2f GB/s\n\n",
           (2 * size) / (milliseconds * 1.0e6));

    // =========================================================================
    // Test 3: Shared memory bank conflicts
    // =========================================================================
    printf("Running Kernel 3: Shared memory bank conflicts (stride = 32)\n");

    cudaEventRecord(start);
    shared_memory_bank_conflicts<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, stride);
    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Effective bandwidth: %.2f GB/s\n\n",
           (2 * size) / (milliseconds * 1.0e6));

    // =========================================================================
    // Test 4: Misaligned access
    // =========================================================================
    printf("Running Kernel 4: Misaligned access (offset = 1)\n");

    // Offset of 1 causes memory accesses to be misaligned at warp boundaries
    int offset = 1;

    cudaEventRecord(start);
    misaligned_access<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, offset);
    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Effective bandwidth: %.2f GB/s\n\n",
           (2 * size) / (milliseconds * 1.0e6));

    // =========================================================================
    // Test 5: Structure of Arrays (SoA) vs Array of Structures (AoS)
    // =========================================================================

    // Free previous allocations
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    // Setup for AoS test
    Vector4 *h_aos_input = (Vector4*)malloc(N/4 * sizeof(Vector4));
    Vector4 *h_aos_output = (Vector4*)malloc(N/4 * sizeof(Vector4));

    for (int i = 0; i < N/4; i++) {
        h_aos_input[i].x = (float)i;
        h_aos_input[i].y = (float)i + 0.1f;
        h_aos_input[i].z = (float)i + 0.2f;
        h_aos_input[i].w = (float)i + 0.3f;
    }

    Vector4 *d_aos_input, *d_aos_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_aos_input, N/4 * sizeof(Vector4)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_aos_output, N/4 * sizeof(Vector4)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_aos_input, h_aos_input, N/4 * sizeof(Vector4), cudaMemcpyHostToDevice));

    printf("Running Kernel 5a: Array of Structures (AoS) access\n");

    cudaEventRecord(start);
    aos_access<<<BLOCKS/4, THREADS_PER_BLOCK>>>(d_aos_input, d_aos_output);
    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Effective bandwidth: %.2f GB/s\n\n",
           (2 * N/4 * sizeof(Vector4)) / (milliseconds * 1.0e6));

    // Setup for SoA test
    float *h_x, *h_y, *h_z, *h_w;
    float *h_x_out, *h_y_out, *h_z_out, *h_w_out;

    h_x = (float*)malloc(N/4 * sizeof(float));
    h_y = (float*)malloc(N/4 * sizeof(float));
    h_z = (float*)malloc(N/4 * sizeof(float));
    h_w = (float*)malloc(N/4 * sizeof(float));
    h_x_out = (float*)malloc(N/4 * sizeof(float));
    h_y_out = (float*)malloc(N/4 * sizeof(float));
    h_z_out = (float*)malloc(N/4 * sizeof(float));
    h_w_out = (float*)malloc(N/4 * sizeof(float));

    for (int i = 0; i < N/4; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)i + 0.1f;
        h_z[i] = (float)i + 0.2f;
        h_w[i] = (float)i + 0.3f;
    }

    float *d_x, *d_y, *d_z, *d_w;
    float *d_x_out, *d_y_out, *d_z_out, *d_w_out;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, N/4 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_y, N/4 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_z, N/4 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w, N/4 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x_out, N/4 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_y_out, N/4 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_z_out, N/4 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w_out, N/4 * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, N/4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, h_y, N/4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_z, h_z, N/4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_w, h_w, N/4 * sizeof(float), cudaMemcpyHostToDevice));

    printf("Running Kernel 5b: Structure of Arrays (SoA) access\n");

    cudaEventRecord(start);
    soa_access<<<BLOCKS/4, THREADS_PER_BLOCK>>>(d_x, d_y, d_z, d_w, d_x_out, d_y_out, d_z_out, d_w_out);
    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Effective bandwidth: %.2f GB/s\n\n",
           (2 * 4 * N/4 * sizeof(float)) / (milliseconds * 1.0e6));

    // =========================================================================
    // Cleanup
    // =========================================================================

    // Free CUDA event resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory for AoS test
    CHECK_CUDA_ERROR(cudaFree(d_aos_input));
    CHECK_CUDA_ERROR(cudaFree(d_aos_output));

    // Free device memory for SoA test
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    CHECK_CUDA_ERROR(cudaFree(d_z));
    CHECK_CUDA_ERROR(cudaFree(d_w));
    CHECK_CUDA_ERROR(cudaFree(d_x_out));
    CHECK_CUDA_ERROR(cudaFree(d_y_out));
    CHECK_CUDA_ERROR(cudaFree(d_z_out));
    CHECK_CUDA_ERROR(cudaFree(d_w_out));

    // Free host memory
    free(h_input);
    free(h_output);
    free(h_aos_input);
    free(h_aos_output);
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_w);
    free(h_x_out);
    free(h_y_out);
    free(h_z_out);
    free(h_w_out);

    printf("Tests complete!\n");
    return 0;
}
