// Implementation of the Layer Normalization operation for transformers

#include <stdio.h>
#include <math.h>

// Layer Normalization kernel
__global__ void layernorm_kernel(float* out, float* mean, float* rstd,
                              const float* inp, const float* weight,
                              const float* bias, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    // The row of input that this thread is responsible for
    const float* x = inp + idx * C;

    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < C; i++) {
        sum += x[i];
    }
    float m = sum / C;
    if(mean != nullptr) {
        mean[idx] = m;
    }

    // Calculate standard deviation
    sum = 0.0f;
    for (int i = 0; i < C; i++) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    float s = rsqrtf(sum / C + 1e-5f);  // Reciprocal of sqrt with epsilon
    if(rstd != nullptr) {
        rstd[idx] = s;
    }

    // Final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = 0; c < C; c++) {
        float n = s * (x[c] - m);
        o[c] = n * weight[c] + bias[c];
    }
}

// Fused layer normalization kernel with shared memory optimization
__global__ void layernorm_shared_kernel(float* out, float* mean, float* rstd,
                                       const float* inp, const float* weight,
                                       const float* bias, int N, int C) {
    extern __shared__ float shared_data[];  // Dynamic shared memory

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    // Get row of input data
    const float* x = inp + idx * C;

    // Each thread calculates part of the sum for mean
    float local_sum = 0.0f;
    for (int i = threadIdx.y; i < C; i += blockDim.y) {
        local_sum += x[i];
    }

    // Use shared memory for reduction
    int shared_idx = threadIdx.x * blockDim.y + threadIdx.y;
    shared_data[shared_idx] = local_sum;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            shared_data[shared_idx] += shared_data[shared_idx + stride];
        }
        __syncthreads();
    }

    // Calculate mean
    float m = shared_data[threadIdx.x * blockDim.y] / C;
    if (mean != nullptr && threadIdx.y == 0) {
        mean[idx] = m;
    }

    // Calculate variance
    local_sum = 0.0f;
    for (int i = threadIdx.y; i < C; i += blockDim.y) {
        float diff = x[i] - m;
        local_sum += diff * diff;
    }

    // Store in shared memory
    shared_data[shared_idx] = local_sum;
    __syncthreads();

    // Reduce again for variance
    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            shared_data[shared_idx] += shared_data[shared_idx + stride];
        }
        __syncthreads();
    }

    // Calculate reciprocal standard deviation
    float s = rsqrtf(shared_data[threadIdx.x * blockDim.y] / C + 1e-5f);
    if (rstd != nullptr && threadIdx.y == 0) {
        rstd[idx] = s;
    }

    // Normalize and apply weight/bias
    float* o = out + idx * C;
    for (int i = threadIdx.y; i < C; i += blockDim.y) {
        float n = s * (x[i] - m);
        o[i] = n * weight[i] + bias[i];
    }
}

// Host function to call the appropriate kernel
void layernorm_forward(float* out, float* mean, float* rstd,
                     float* inp, const float* weight, const float* bias,
                     int B, int T, int C, cudaStream_t stream = 0) {
    const int block_size = 256;
    const int N = B * T;
    const int grid_size = (N + block_size - 1) / block_size;

    layernorm_kernel<<<grid_size, block_size, 0, stream>>>(
        out, mean, rstd, inp, weight, bias, N, C);
}

// Improved version with shared memory
void layernorm_forward_shared(float* out, float* mean, float* rstd,
                            float* inp, const float* weight, const float* bias,
                            int B, int T, int C, cudaStream_t stream = 0) {
    const int threads_x = 32;  // One thread per sequence element
    const int threads_y = 32;  // Multiple threads cooperate on each element
    dim3 block_dim(threads_x, threads_y);

    const int N = B * T;
    const int grid_size = (N + threads_x - 1) / threads_x;

    // Amount of shared memory needed
    size_t shared_mem_size = threads_x * threads_y * sizeof(float);

    layernorm_shared_kernel<<<grid_size, block_dim, shared_mem_size, stream>>>(
        out, mean, rstd, inp, weight, bias, N, C);
}

int main() {
    // Test dimensions
    int B = 32;    // Batch size
    int T = 512;   // Sequence length
    int C = 768;   // Hidden dimension

    size_t input_size = B * T * C;
    size_t weights_size = C;
    size_t stats_size = B * T;

    size_t input_bytes = input_size * sizeof(float);
    size_t output_bytes = input_bytes;
    size_t weights_bytes = weights_size * sizeof(float);
    size_t stats_bytes = stats_size * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(input_bytes);
    float *h_output = (float*)malloc(output_bytes);
    float *h_output_shared = (float*)malloc(output_bytes);
    float *h_weight = (float*)malloc(weights_bytes);
    float *h_bias = (float*)malloc(weights_bytes);
    float *h_mean = (float*)malloc(stats_bytes);
    float *h_rstd = (float*)malloc(stats_bytes);

    // Initialize input data
    for (int i = 0; i < input_size; i++) {
        h_input[i] = (float)(rand() % 1000) / 100.0f - 5.0f;  // Random values roughly between -5 and 5
    }

    // Initialize weights and biases
    for (int i = 0; i < weights_size; i++) {
        h_weight[i] = 1.0f;  // Simple weights for testing
        h_bias[i] = 0.0f;    // No bias for testing
    }

    // Allocate device memory
    float *d_input, *d_output, *d_output_shared, *d_weight, *d_bias, *d_mean, *d_rstd;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMalloc(&d_output_shared, output_bytes);
    cudaMalloc(&d_weight, weights_bytes);
    cudaMalloc(&d_bias, weights_bytes);
    cudaMalloc(&d_mean, stats_bytes);
    cudaMalloc(&d_rstd, stats_bytes);

    // Copy data to device
    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, weights_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, weights_bytes, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run the basic layer normalization
    cudaEventRecord(start);
    layernorm_forward(d_output, d_mean, d_rstd, d_input, d_weight, d_bias, B, T, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_basic = 0;
    cudaEventElapsedTime(&milliseconds_basic, start, stop);

    // Run the shared memory version
    cudaEventRecord(start);
    layernorm_forward_shared(d_output_shared, d_mean, d_rstd, d_input, d_weight, d_bias, B, T, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_shared = 0;
    cudaEventElapsedTime(&milliseconds_shared, start, stop);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_shared, d_output_shared, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mean, d_mean, stats_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rstd, d_rstd, stats_bytes, cudaMemcpyDeviceToHost);

    // Validate results (comparing first few elements)
    bool correct = true;
    for (int i = 0; i < 100; i++) {
        if (fabs(h_output[i] - h_output_shared[i]) > 1e-4) {
            printf("Results don't match at %d: %f vs %f\n", i, h_output[i], h_output_shared[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Results match between implementations.\n");
    }

    // Print performance results
    printf("Basic Layer Norm: %.3f ms\n", milliseconds_basic);
    printf("Shared Memory Layer Norm: %.3f ms\n", milliseconds_shared);
    printf("Speedup: %.2fx\n", milliseconds_basic / milliseconds_shared);

    // Free memory
    free(h_input);
    free(h_output);
    free(h_output_shared);
    free(h_weight);
    free(h_bias);
    free(h_mean);
    free(h_rstd);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_shared);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_mean);
    cudaFree(d_rstd);

    return 0;
}

/* 
Compilation and Run:
$ nvcc -o layer_normalization 07_layer_normalization.cu
$ ./layer_normalization

Experiment Ideas:
1. Try different batch sizes and sequence lengths
2. Compare performance as hidden dimension (C) changes
3. Explore the effect of thread block dimensions
4. Try a different epsilon value in the normalization
*/
