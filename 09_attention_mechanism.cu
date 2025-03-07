// Implementation of self-attention mechanism in transformers

#include <stdio.h>
#include <cublas_v2.h>

// Simplified QKV transformation kernel: compute query, key, value matrices
__global__ void qkv_transform_kernel(float* q, float* k, float* v,
                                    const float* inp,
                                    const float* q_weight, const float* k_weight, const float* v_weight,
                                    int B, int T, int C, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * NH * HS) return;

    // Determine position in output tensors
    int h = (idx / (T * HS)) % NH;  // Head index
    int t = (idx / HS) % T;         // Token/position index
    int b = idx / (NH * T * HS);    // Batch index
    int hs = idx % HS;              // Position within head

    // Compute Q, K, V projections as matrix multiplications
    float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;
    for (int c = 0; c < C; c++) {
        // Get input element
        float x = inp[b * T * C + t * C + c];

        // Multiply by weights
        sum_q += x * q_weight[h * HS * C + hs * C + c];
        sum_k += x * k_weight[h * HS * C + hs * C + c];
        sum_v += x * v_weight[h * HS * C + hs * C + c];
    }

    // Write to outputs
    q[b * NH * T * HS + h * T * HS + t * HS + hs] = sum_q;
    k[b * NH * T * HS + h * T * HS + t * HS + hs] = sum_k;
    v[b * NH * T * HS + h * T * HS + t * HS + hs] = sum_v;
}

// Basic attention kernel (naive, not optimized)
__global__ void attention_kernel(float* output, 
                               const float* q, const float* k, const float* v,
                               int B, int NH, int T, int HS) {
    int b = blockIdx.z / NH;                 // Batch index
    int h = blockIdx.z % NH;                 // Head index
    int t = blockIdx.y;                      // Query token index
    int hs = blockIdx.x * blockDim.x + threadIdx.x;  // Position within head

    if (hs >= HS) return;

    // Get query vector for this token
    float query = q[b * NH * T * HS + h * T * HS + t * HS + hs];

    // Compute attention scores for each key
    float scores[1024];  // Assume T <= 1024 for simplicity
    float max_score = -1e9f;

    for (int s = 0; s < T; s++) {
        scores[s] = 0.0f;

        // Compute attention score (dot product of q and k, scaled)
        for (int d = 0; d < HS; d++) {
            float key = k[b * NH * T * HS + h * T * HS + s * HS + d];
            if (hs == 0) {  // Only one thread computes the full score
                scores[s] += query * key;
            }
        }

        if (hs == 0) {
            scores[s] /= sqrtf(HS);  // Scale by sqrt(head_size)
            max_score = fmaxf(max_score, scores[s]);
        }
    }

    __syncthreads();  // Ensure max_score is computed

    // Compute softmax
    float softmax_denom = 0.0f;
    for (int s = 0; s < T; s++) {
        if (hs == 0) {  // Only one thread computes the exp and sum
            scores[s] = expf(scores[s] - max_score);
            softmax_denom += scores[s];
        }
    }

    __syncthreads();  // Ensure softmax_denom is computed

    // Compute weighted sum of values
    float result = 0.0f;
    for (int s = 0; s < T; s++) {
        if (hs == 0) {
            scores[s] /= softmax_denom;  // Normalize
        }

        float value = v[b * NH * T * HS + h * T * HS + s * HS + hs];
        if (hs == 0) {
            result += scores[s] * value;
        } else {
            result += 0; // This is just a placeholder, we'll handle this properly in flash attention
        }
    }

    // Write to output
    output[b * NH * T * HS + h * T * HS + t * HS + hs] = result;
}

// Flash attention kernel (more efficient attention computation)
__global__ void flash_attention_kernel(float* output, 
                                    const float* q, const float* k, const float* v,
                                    int B, int NH, int T, int HS) {
    int b = blockIdx.z / NH;          // Batch index
    int h = blockIdx.z % NH;          // Head index
    int t = blockIdx.y;               // Query position

    // Shared memory for keys, values, and scores
    extern __shared__ float shared_mem[];
    float* k_cache = shared_mem;
    float* v_cache = &k_cache[T * HS];
    float* scores = &v_cache[T * HS];

    // Each thread loads keys and values for its dimension
    for (int i = threadIdx.x; i < T * HS; i += blockDim.x) {
        int idx_t = i / HS;
        int idx_hs = i % HS;

        // Only load keys/values up to current token position (causal masking)
        if (idx_t <= t) {
            k_cache[i] = k[b * NH * T * HS + h * T * HS + idx_t * HS + idx_hs];
            v_cache[i] = v[b * NH * T * HS + h * T * HS + idx_t * HS + idx_hs];
        }
    }

    __syncthreads();  // Ensure all keys and values are loaded

    // Compute attention for each head dimension
    for (int hs = threadIdx.x; hs < HS; hs += blockDim.x) {
        float query = q[b * NH * T * HS + h * T * HS + t * HS + hs];
        float max_score = -1e9f;
        float sum_exp = 0.0f;
        float weighted_sum = 0.0f;

        // First pass: compute scores and find max
        for (int s = 0; s <= t; s++) {  // Causal masking
            float score = 0.0f;
            for (int d = 0; d < HS; d++) {
                score += query * k_cache[s * HS + d];
            }
            score /= sqrtf(HS);

            if (hs == 0) {  // Store scores in shared memory (just one thread)
                scores[s] = score;
            }

            max_score = fmaxf(max_score, score);
        }

        // Second pass: compute softmax and weighted sum
        for (int s = 0; s <= t; s++) {
            float score = (hs == 0) ? scores[s] : 0.0f;
            float exp_score = expf(score - max_score);
            sum_exp += exp_score;

            weighted_sum += exp_score * v_cache[s * HS + hs];
        }

        // Normalize and write output
        output[b * NH * T * HS + h * T * HS + t * HS + hs] = weighted_sum / sum_exp;
    }
}

// Project attention outputs back to model dimension
__global__ void attention_output_kernel(float* output,
                                      const float* attn_output,
                                      const float* output_weight,
                                      int B, int T, int C, int NH, int HS) {
    // Each thread computes one element of the output
    int b = blockIdx.z;
    int t = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= C) return;

    float sum = 0.0f;
    for (int h = 0; h < NH; h++) {
        for (int hs = 0; hs < HS; hs++) {
            int attn_idx = b * NH * T * HS + h * T * HS + t * HS + hs;
            int weight_idx = (h * HS + hs) * C + c;
            sum += attn_output[attn_idx] * output_weight[weight_idx];
        }
    }

    output[b * T * C + t * C + c] = sum;
}

// Host function for QKV transformation
void qkv_transform(float* q, float* k, float* v,
                 const float* inp,
                 const float* q_weight, const float* k_weight, const float* v_weight,
                 int B, int T, int C, int NH, int HS,
                 cudaStream_t stream = 0) {
    int total_elements = B * T * NH * HS;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    qkv_transform_kernel<<<grid_size, block_size, 0, stream>>>(
        q, k, v, inp, q_weight, k_weight, v_weight, B, T, C, NH, HS);
}

// Host function for attention computation
void attention_forward(float* output, 
                     const float* q, const float* k, const float* v,
                     int B, int T, int NH, int HS,
                     bool use_flash_attn = true,
                     cudaStream_t stream = 0) {
    if (use_flash_attn) {
        // Flash attention configuration
        int block_size = 256;
        size_t shared_mem_size = (2 * T * HS + T) * sizeof(float); // For k_cache, v_cache, and scores

        dim3 grid(1, T, B * NH);
        flash_attention_kernel<<<grid, block_size, shared_mem_size, stream>>>(
            output, q, k, v, B, NH, T, HS);
    } else {
        // Basic attention configuration
        int block_size = 256;
        int grid_x = (HS + block_size - 1) / block_size;

        dim3 grid(grid_x, T, B * NH);
        attention_kernel<<<grid, block_size, 0, stream>>>(
            output, q, k, v, B, NH, T, HS);
    }
}

// Host function for attention output projection
void attention_output_projection(float* output,
                               const float* attn_output,
                               const float* output_weight,
                               int B, int T, int C, int NH, int HS,
                               cudaStream_t stream = 0) {
    int block_size = 256;
    int grid_x = (C + block_size - 1) / block_size;

    dim3 grid(grid_x, T, B);
    attention_output_kernel<<<grid, block_size, 0, stream>>>(
        output, attn_output, output_weight, B, T, C, NH, HS);
}

// Alternative: use cuBLAS for the projections (more efficient)
void attention_with_cublas(float* output,
                         const float* inp,
                         const float* q_weight, const float* k_weight, const float* v_weight,
                         const float* output_weight,
                         int B, int T, int C, int NH, int HS,
                         cublasHandle_t handle,
                         cudaStream_t stream = 0) {
    // Temporary buffers
    float *q, *k, *v, *attn_out;
    cudaMalloc(&q, B * NH * T * HS * sizeof(float));
    cudaMalloc(&k, B * NH * T * HS * sizeof(float));
    cudaMalloc(&v, B * NH * T * HS * sizeof(float));
    cudaMalloc(&attn_out, B * NH * T * HS * sizeof(float));

    // Set cuBLAS stream
    cublasSetStream(handle, stream);

    // 1. Project input to Q, K, V
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Matrix multiplication for Q, K, V projections
    // (B*T) x C @ C x (NH*HS) -> (B*T) x (NH*HS)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                NH * HS, B * T, C,
                &alpha,
                q_weight, NH * HS,
                inp, C,
                &beta,
                q, NH * HS);

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                NH * HS, B * T, C,
                &alpha,
                k_weight, NH * HS,
                inp, C,
                &beta,
                k, NH * HS);

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                NH * HS, B * T, C,
                &alpha,
                v_weight, NH * HS,
                inp, C,
                &beta,
                v, NH * HS);

    // 2. Compute attention
    attention_forward(attn_out, q, k, v, B, T, NH, HS, true, stream);

    // 3. Project attention output back to model dimension
    // (B*T) x (NH*HS) @ (NH*HS) x C -> (B*T) x C
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                C, B * T, NH * HS,
                &alpha,
                output_weight, C,
                attn_out, NH * HS,
                &beta,
                output, C);

    // Free temporary buffers
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(attn_out);
}

int main() {
    // Model dimensions
    int B = 2;     // Batch size
    int T = 128;   // Sequence length
    int C = 512;   // Model dimension
    int NH = 8;    // Number of attention heads
    int HS = C / NH; // Head size (64)

    // Allocate host memory for input and weights
    float *h_input = (float*)malloc(B * T * C * sizeof(float));
    float *h_output_basic = (float*)malloc(B * T * C * sizeof(float));
    float *h_output_flash = (float*)malloc(B * T * C * sizeof(float));
    float *h_output_cublas = (float*)malloc(B * T * C * sizeof(float));

    float *h_q_weight = (float*)malloc(NH * HS * C * sizeof(float));
    float *h_k_weight = (float*)malloc(NH * HS * C * sizeof(float));
    float *h_v_weight = (float*)malloc(NH * HS * C * sizeof(float));
    float *h_out_weight = (float*)malloc(NH * HS * C * sizeof(float));

    // Initialize input with random data
    for (int i = 0; i < B * T * C; i++) {
        h_input[i] = (float)(rand() % 1000 - 500) / 500.0f;  // Random values in [-1, 1]
    }

    // Initialize weights (this would normally be loaded from a trained model)
    for (int i = 0; i < NH * HS * C; i++) {
        h_q_weight[i] = (float)(rand() % 1000 - 500) / (500.0f * sqrtf(C));
        h_k_weight[i] = (float)(rand() % 1000 - 500) / (500.0f * sqrtf(C));
        h_v_weight[i] = (float)(rand() % 1000 - 500) / (500.0f * sqrtf(C));
        h_out_weight[i] = (float)(rand() % 1000 - 500) / (500.0f * sqrtf(NH * HS));
    }

    // Allocate device memory
    float *d_input, *d_output_basic, *d_output_flash, *d_output_cublas;
    float *d_q, *d_k, *d_v, *d_attn_out;
    float *d_q_weight, *d_k_weight, *d_v_weight, *d_out_weight;

    cudaMalloc(&d_input, B * T * C * sizeof(float));
    cudaMalloc(&d_output_basic, B * T * C * sizeof(float));
    cudaMalloc(&d_output_flash, B * T * C * sizeof(float));
    cudaMalloc(&d_output_cublas, B * T * C * sizeof(float));

    cudaMalloc(&d_q, B * NH * T * HS * sizeof(float));
    cudaMalloc(&d_k, B * NH * T * HS * sizeof(float));
    cudaMalloc(&d_v, B * NH * T * HS * sizeof(float));
    cudaMalloc(&d_attn_out, B * NH * T * HS * sizeof(float));

    cudaMalloc(&d_q_weight, NH * HS * C * sizeof(float));
    cudaMalloc(&d_k_weight, NH * HS * C * sizeof(float));
    cudaMalloc(&d_v_weight, NH * HS * C * sizeof(float));
    cudaMalloc(&d_out_weight, NH * HS * C * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_weight, h_q_weight, NH * HS * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_weight, h_k_weight, NH * HS * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_weight, h_v_weight, NH * HS * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_weight, h_out_weight, NH * HS * C * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Run basic attention (manual implementation)
    cudaEventRecord(start);

    // 1. QKV transformation
    qkv_transform(d_q, d_k, d_v, d_input, d_q_weight, d_k_weight, d_v_weight, B, T, C, NH, HS);

    // 2. Basic attention
    attention_forward(d_attn_out, d_q, d_k, d_v, B, T, NH, HS, false);

    // 3. Output projection
    attention_output_projection(d_output_basic, d_attn_out, d_out_weight, B, T, C, NH, HS);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_basic = 0;
    cudaEventElapsedTime(&milliseconds_basic, start, stop);

    // Run flash attention
    cudaEventRecord(start);

    // 1. QKV transformation
    qkv_transform(d_q, d_k, d_v, d_input, d_q_weight, d_k_weight, d_v_weight, B, T, C, NH, HS);

    // 2. Flash attention
    attention_forward(d_attn_out, d_q, d_k, d_v, B, T, NH, HS, true);

    // 3. Output projection
    attention_output_projection(d_output_flash, d_attn_out, d_out_weight, B, T, C, NH, HS);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_flash = 0;
    cudaEventElapsedTime(&milliseconds_flash, start, stop);

    // Run attention with cuBLAS
    cudaEventRecord(start);

    attention_with_cublas(d_output_cublas, d_input, 
                        d_q_weight, d_k_weight, d_v_weight, d_out_weight,
                        B, T, C, NH, HS, handle);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_cublas = 0;
    cudaEventElapsedTime(&milliseconds_cublas, start, stop);

    // Copy results back to host
    cudaMemcpy(h_output_basic, d_output_basic, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_flash, d_output_flash, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_cublas, d_output_cublas, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    float max_diff_flash = 0.0f;
    float max_diff_cublas = 0.0f;

    for (int i = 0; i < B * T * C; i++) {
        max_diff_flash = fmaxf(max_diff_flash, fabsf(h_output_basic[i] - h_output_flash[i]));
        max_diff_cublas = fmaxf(max_diff_cublas, fabsf(h_output_basic[i] - h_output_cublas[i]));
    }

    // Print results
    printf("Attention Performance and Accuracy:\n");
    printf("-----------------------------------\n");
    printf("Basic Attention:       %.3f ms\n", milliseconds_basic);
    printf("Flash Attention:       %.3f ms (max diff: %.6f)\n", milliseconds_flash, max_diff_flash);
    printf("cuBLAS Attention:      %.3f ms (max diff: %.6f)\n", milliseconds_cublas, max_diff_cublas);

    // Print speedups
    printf("\nSpeedups:\n");
    printf("Flash vs Basic:        %.2fx\n", milliseconds_basic / milliseconds_flash);
    printf("cuBLAS vs Basic:       %.2fx\n", milliseconds_basic / milliseconds_cublas);

    // Clean up
    free(h_input);
    free(h_output_basic);
    free(h_output_flash);
    free(h_output_cublas);
    free(h_q_weight);
    free(h_k_weight);
    free(h_v_weight);
    free(h_out_weight);

    cudaFree(d_input);
    cudaFree(d_output_basic);
    cudaFree(d_output_flash);
    cudaFree(d_output_cublas);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_attn_out);
    cudaFree(d_q_weight);
    cudaFree(d_k_weight);
    cudaFree(d_v_weight);
    cudaFree(d_out_weight);

    cublasDestroy(handle);

    return 0;
}

/* 
Compilation and Run:
$ nvcc -o attention_mechanism 09_attention_mechanism.cu -lcublas
$ ./attention_mechanism

Experiment Ideas:
1. Test with different batch sizes and sequence lengths
2. Measure how attention complexity scales with sequence length
3. Implement a masked version for encoder-decoder attention
4. Implement a more sophisticated version of Flash Attention
*/
