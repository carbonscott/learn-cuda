// Implementation of the GELU activation function used in transformers

#include <stdio.h>
#include <math.h>

// Simple GELU activation kernel
__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = inp[idx];
    float cube = 0.044715f * x * x * x;
    out[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + cube)));
}

// GELU kernel with vectorized load/store operations for improved memory throughput
// This is similar to the Packed128 optimization in the transformer code
__global__ void gelu_vectorized_kernel(float4* out, const float4* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load 4 elements at once
    float4 in_vec = inp[idx];
    float4 out_vec;

    // Process each element
    float x = in_vec.x;
    float cube = 0.044715f * x * x * x;
    out_vec.x = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + cube)));

    x = in_vec.y;
    cube = 0.044715f * x * x * x;
    out_vec.y = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + cube)));

    x = in_vec.z;
    cube = 0.044715f * x * x * x;
    out_vec.z = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + cube)));

    x = in_vec.w;
    cube = 0.044715f * x * x * x;
    out_vec.w = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + cube)));

    // Store 4 elements at once
    out[idx] = out_vec;
}

// Alternative GELU approximation that's faster but slightly less accurate
__global__ void gelu_fast_kernel(float* out, const float* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = inp[idx];
    // Faster approximation: 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))
    // Further simplified with constants pre-computed and fewer operations
    out[idx] = x * 0.5f * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
}

// Vectorized fast GELU
__global__ void gelu_fast_vectorized_kernel(float4* out, const float4* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load 4 elements at once
    float4 in_vec = inp[idx];
    float4 out_vec;

    // Process each element with fast GELU
    float x = in_vec.x;
    out_vec.x = x * 0.5f * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));

    x = in_vec.y;
    out_vec.y = x * 0.5f * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));

    x = in_vec.z;
    out_vec.z = x * 0.5f * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));

    x = in_vec.w;
    out_vec.w = x * 0.5f * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));

    // Store 4 elements at once
    out[idx] = out_vec;
}

// Host wrapper functions
void gelu_forward(float* out, const float* inp, int N, cudaStream_t stream = 0) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    gelu_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(out, inp, N);
}

void gelu_forward_vectorized(float* out, const float* inp, int N, cudaStream_t stream = 0) {
    int vec_size = 4; // float4 contains 4 floats
    int vec_N = N / vec_size;
    int blockSize = 256;
    int numBlocks = (vec_N + blockSize - 1) / blockSize;

    // Reinterpret cast pointers as float4
    float4* out_vec = reinterpret_cast<float4*>(out);
    const float4* inp_vec = reinterpret_cast<const float4*>(inp);

    gelu_vectorized_kernel<<<numBlocks, blockSize, 0, stream>>>(out_vec, inp_vec, vec_N);

    // Handle remaining elements (if N is not a multiple of 4)
    int remaining = N % vec_size;
    if (remaining > 0) {
        gelu_forward_kernel<<<1, remaining, 0, stream>>>(
            out + (N - remaining), inp + (N - remaining), remaining);
    }
}

void gelu_forward_fast(float* out, const float* inp, int N, cudaStream_t stream = 0) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    gelu_fast_kernel<<<numBlocks, blockSize, 0, stream>>>(out, inp, N);
}

void gelu_forward_fast_vectorized(float* out, const float* inp, int N, cudaStream_t stream = 0) {
    int vec_size = 4;
    int vec_N = N / vec_size;
    int blockSize = 256;
    int numBlocks = (vec_N + blockSize - 1) / blockSize;

    float4* out_vec = reinterpret_cast<float4*>(out);
    const float4* inp_vec = reinterpret_cast<const float4*>(inp);

    gelu_fast_vectorized_kernel<<<numBlocks, blockSize, 0, stream>>>(out_vec, inp_vec, vec_N);

    int remaining = N % vec_size;
    if (remaining > 0) {
        gelu_fast_kernel<<<1, remaining, 0, stream>>>(
            out + (N - remaining), inp + (N - remaining), remaining);
    }
}

// CPU reference implementation for validation
void gelu_cpu(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + cube)));
    }
}

int main() {
    // Test data size (should be a multiple of 4 for vectorized version)
    int N = 10000000;

    // Allocate host memory
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    float* h_output_vec = (float*)malloc(N * sizeof(float));
    float* h_output_fast = (float*)malloc(N * sizeof(float));
    float* h_output_fast_vec = (float*)malloc(N * sizeof(float));
    float* h_output_cpu = (float*)malloc(N * sizeof(float));

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 2000 - 1000) / 100.0f;  // Random values in [-10, 10]
    }

    // Allocate device memory
    float *d_input, *d_output, *d_output_vec, *d_output_fast, *d_output_fast_vec;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_output_vec, N * sizeof(float));
    cudaMalloc(&d_output_fast, N * sizeof(float));
    cudaMalloc(&d_output_fast_vec, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run standard GELU
    cudaEventRecord(start);
    gelu_forward(d_output, d_input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_standard = 0;
    cudaEventElapsedTime(&milliseconds_standard, start, stop);

    // Run vectorized GELU
    cudaEventRecord(start);
    gelu_forward_vectorized(d_output_vec, d_input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_vectorized = 0;
    cudaEventElapsedTime(&milliseconds_vectorized, start, stop);

    // Run fast GELU
    cudaEventRecord(start);
    gelu_forward_fast(d_output_fast, d_input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_fast = 0;
    cudaEventElapsedTime(&milliseconds_fast, start, stop);

    // Run fast vectorized GELU
    cudaEventRecord(start);
    gelu_forward_fast_vectorized(d_output_fast_vec, d_input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_fast_vec = 0;
    cudaEventElapsedTime(&milliseconds_fast_vec, start, stop);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_vec, d_output_vec, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fast, d_output_fast, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fast_vec, d_output_fast_vec, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    gelu_cpu(h_output_cpu, h_input, N);

    // Validate results
    double error_standard = 0.0;
    double error_vectorized = 0.0;
    double error_fast = 0.0;
    double error_fast_vec = 0.0;

    for (int i = 0; i < N; i++) {
        error_standard += fabs(h_output[i] - h_output_cpu[i]);
        error_vectorized += fabs(h_output_vec[i] - h_output_cpu[i]);
        error_fast += fabs(h_output_fast[i] - h_output_cpu[i]);
        error_fast_vec += fabs(h_output_fast_vec[i] - h_output_cpu[i]);
    }

    error_standard /= N;
    error_vectorized /= N;
    error_fast /= N;
    error_fast_vec /= N;

    // Print results
    printf("GELU Performance and Accuracy:\n");
    printf("-------------------------------\n");
    printf("Standard GELU:         %.3f ms, error: %.9f\n", milliseconds_standard, error_standard);
    printf("Vectorized GELU:       %.3f ms, error: %.9f\n", milliseconds_vectorized, error_vectorized);
    printf("Fast GELU:             %.3f ms, error: %.9f\n", milliseconds_fast, error_fast);
    printf("Fast Vectorized GELU:  %.3f ms, error: %.9f\n", milliseconds_fast_vec, error_fast_vec);

    // Print speedups
    printf("\nSpeedups:\n");
    printf("Vectorized vs Standard: %.2fx\n", milliseconds_standard / milliseconds_vectorized);
    printf("Fast vs Standard:       %.2fx\n", milliseconds_standard / milliseconds_fast);
    printf("Fast Vectorized vs Standard: %.2fx\n", milliseconds_standard / milliseconds_fast_vec);

    // Clean up
    free(h_input);
    free(h_output);
    free(h_output_vec);
    free(h_output_fast);
    free(h_output_fast_vec);
    free(h_output_cpu);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_vec);
    cudaFree(d_output_fast);
    cudaFree(d_output_fast_vec);

    return 0;
}

/*
Compilation and Run:
$ nvcc -o gelu_activation 08_gelu_activation.cu
$ ./gelu_activation

Experiment Ideas:
1. Test with different input distributions (uniform, normal, etc.)
2. Measure how the performance changes with array size
3. Try implementing a fused GELU+bias kernel
4. Experiment with other GELU approximations that might be faster
*/
