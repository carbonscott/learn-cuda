// Complete transformer block implementation combining all components

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

// Utility function to check CUDA errors
#define cudaCheck(err) { cudaError_t err_ = (err); if (err_ != cudaSuccess) { \
    printf("CUDA error: %s at line %d\n", cudaGetErrorString(err_), __LINE__); \
    exit(1); } }

// Define a helper for ceiling division
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// ---------------------------------------------------------------------------
// Layer Normalization
// ---------------------------------------------------------------------------
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

// Host function to call the layer normalization kernel
void layernorm_forward(float* out, float* mean, float* rstd,
                     float* inp, const float* weight, const float* bias,
                     int B, int T, int C, cudaStream_t stream = 0) {
    const int block_size = 256;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size);

    layernorm_kernel<<<grid_size, block_size, 0, stream>>>(
        out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// GELU Activation
// ---------------------------------------------------------------------------
__global__ void gelu_kernel(float* out, const float* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = inp[idx];
    float cube = 0.044715f * x * x * x;
    out[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + cube)));
}

// Host function to call the GELU kernel
void gelu_forward(float* out, const float* inp, int N, cudaStream_t stream = 0) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);

    gelu_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Residual Connection
// ---------------------------------------------------------------------------
__global__ void residual_kernel(float* out, const float* inp1, const float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    out[idx] = inp1[idx] + inp2[idx];
}

// Host function to call the residual connection kernel
// void residual_forward(float* out, const float* inp1, const float* inp2, int N, cudaStream_t stream = 0) {
//     const int block_size = 256;
//     const int grid_size = CEIL_DIV(N, block_size);
// 
//     residual_kernel<<<grid_size, block_size, 0, stream>>>(out, inp1, inp2, N);
//     cudaCheck(cudaGetLastError());
// }
void residual_forward(float* out, const float* inp1, const float* inp2, int N, cudaStream_t stream = 0) {
    // Check for invalid parameters
    if (N <= 0) return;
    if (out == NULL || inp1 == NULL || inp2 == NULL) {
        printf("Error: NULL pointer passed to residual_forward\n");
        return;
    }

    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);

    // Check for valid grid dimensions
    if (grid_size <= 0) {
        printf("Error: Invalid grid size %d for N=%d\n", grid_size, N);
        return;
    }

    residual_kernel<<<grid_size, block_size, 0, stream>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// Self-Attention
// ---------------------------------------------------------------------------
// QKV Transformation
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

__global__ void simple_attention_kernel(float* output, const float* q, const float* k, const float* v,
                                     int B, int NH, int T, int HS) {
    int b = blockIdx.z / NH;
    int h = blockIdx.z % NH;
    int t = blockIdx.y;
    int hs = blockIdx.x * blockDim.x + threadIdx.x;

    if (hs >= HS) return;

    float query = q[b * NH * T * HS + h * T * HS + t * HS + hs];
    float max_score = -1e9f;

    // First pass - find max score for numerical stability
    for (int s = 0; s <= t; s++) {
        float score = 0.0f;
        for (int d = 0; d < HS; d++) {
            score += query * k[b * NH * T * HS + h * T * HS + s * HS + d];
        }
        score /= sqrtf(HS);
        max_score = fmaxf(max_score, score);
    }

    // Second pass - compute weighted sum
    float weighted_sum = 0.0f;
    float denom = 0.0f;

    for (int s = 0; s <= t; s++) {
        float score = 0.0f;
        for (int d = 0; d < HS; d++) {
            score += query * k[b * NH * T * HS + h * T * HS + s * HS + d];
        }
        score /= sqrtf(HS);

        float exp_score = expf(score - max_score);
        denom += exp_score;
        weighted_sum += exp_score * v[b * NH * T * HS + h * T * HS + s * HS + hs];
    }

    output[b * NH * T * HS + h * T * HS + t * HS + hs] = weighted_sum / denom;
}

__global__ void tiled_flash_attention_kernel(
    float* output, const float* q, const float* k, const float* v,
    int B, int NH, int T, int HS, int TILE_SIZE) {

    int b = blockIdx.z / NH;          // Batch index
    int h = blockIdx.z % NH;          // Head index
    int t = blockIdx.y;               // Query position

    // Shared memory for keys and values of the current tile
    extern __shared__ float shared_mem[];
    float* k_tile = shared_mem;
    float* v_tile = &k_tile[TILE_SIZE * HS];

    // Each thread handles one or more dimensions of the head
    float max_score = -1e9f;
    float scores[64];  // Temporary scores for each token (adjust size if needed)
    float weighted_sum[64] = {0.0f};  // Per-head dimension accumulators
    int tile_count = CEIL_DIV(t + 1, TILE_SIZE);

    // First pass to find maximum score (for numerical stability)
    for (int tile = 0; tile < tile_count; tile++) {
        int tile_start = tile * TILE_SIZE;
        int tile_end = min(tile_start + TILE_SIZE, t + 1);
        int tile_len = tile_end - tile_start;

        // Collaboratively load this tile's keys to shared memory
        for (int i = threadIdx.x; i < tile_len * HS; i += blockDim.x) {
            int local_t = i / HS;
            int local_hs = i % HS;
            int global_t = tile_start + local_t;

            k_tile[local_t * HS + local_hs] = k[b * NH * T * HS + h * T * HS + global_t * HS + local_hs];
        }

        __syncthreads();

        // Each thread computes max score for its head dimensions
        for (int hs = threadIdx.x; hs < HS; hs += blockDim.x) {
            float q_val = q[b * NH * T * HS + h * T * HS + t * HS + hs];

            for (int j = 0; j < tile_len; j++) {
                int src_pos = tile_start + j;
                if (src_pos <= t) {  // Apply causal mask
                    float score = 0.0f;

                    // Dot product between query and key
                    for (int d = 0; d < HS; d++) {
                        score += q_val * k_tile[j * HS + d];
                    }
                    score /= sqrtf(HS);

                    max_score = fmaxf(max_score, score);
                    scores[j] = score;  // Save for second pass
                }
            }
        }

        __syncthreads();
    }

    // Share max score among threads
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = max_score;
    }
    __syncthreads();
    max_score = shared_max;

    // Second pass to compute weighted sum with numerical stability
    float sum_exp = 0.0f;

    for (int tile = 0; tile < tile_count; tile++) {
        int tile_start = tile * TILE_SIZE;
        int tile_end = min(tile_start + TILE_SIZE, t + 1);
        int tile_len = tile_end - tile_start;

        // Load this tile's keys and values to shared memory
        for (int i = threadIdx.x; i < tile_len * HS; i += blockDim.x) {
            int local_t = i / HS;
            int local_hs = i % HS;
            int global_t = tile_start + local_t;

            v_tile[local_t * HS + local_hs] = v[b * NH * T * HS + h * T * HS + global_t * HS + local_hs];
        }

        __syncthreads();

        // Each thread accumulates weighted values for its head dimensions
        for (int hs = threadIdx.x; hs < HS; hs += blockDim.x) {
            float q_val = q[b * NH * T * HS + h * T * HS + t * HS + hs];

            for (int j = 0; j < tile_len; j++) {
                int src_pos = tile_start + j;
                if (src_pos <= t) {  // Apply causal mask
                    float score;
                    if (tile == 0) {
                        // Reuse scores from first pass if available
                        score = scores[j];
                    } else {
                        // Recompute score
                        score = 0.0f;
                        for (int d = 0; d < HS; d++) {
                            score += q_val * k_tile[j * HS + d];
                        }
                        score /= sqrtf(HS);
                    }

                    float exp_score = expf(score - max_score);
                    sum_exp += exp_score;
                    weighted_sum[hs] += exp_score * v_tile[j * HS + hs];
                }
            }
        }

        __syncthreads();
    }

    // Final normalization and output
    for (int hs = threadIdx.x; hs < HS; hs += blockDim.x) {
        output[b * NH * T * HS + h * T * HS + t * HS + hs] = weighted_sum[hs] / sum_exp;
    }
}

// Flash attention kernel (optimized attention computation)
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

            max_score = fmaxf(max_score, score);
        }

        // Second pass: compute softmax and weighted sum
        for (int s = 0; s <= t; s++) {
            float score = 0.0f;
            for (int d = 0; d < HS; d++) {
                score += query * k_cache[s * HS + d];
            }
            score /= sqrtf(HS);

            float exp_score = expf(score - max_score);
            sum_exp += exp_score;

            weighted_sum += exp_score * v_cache[s * HS + hs];
        }

        // Normalize and write output
        output[b * NH * T * HS + h * T * HS + t * HS + hs] = weighted_sum / sum_exp;
    }
}

// Attention output projection
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

// ---------------------------------------------------------------------------
// Feed-Forward MLP
// ---------------------------------------------------------------------------
// MLP forward pass (expansion layer)
__global__ void mlp_fc_kernel(float* output, const float* input, 
                             const float* weight, const float* bias,
                             int B, int T, int C, int intermediate_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * intermediate_size) return;

    int b = idx / (T * intermediate_size);
    int t = (idx / intermediate_size) % T;
    int j = idx % intermediate_size;

    float sum = bias[j];  // Start with bias

    // Matrix multiplication
    for (int i = 0; i < C; i++) {
        int input_idx = b * T * C + t * C + i;
        int weight_idx = i * intermediate_size + j;
        sum += input[input_idx] * weight[weight_idx];
    }

    output[idx] = sum;
}

// MLP projection back to model dimension
__global__ void mlp_proj_kernel(float* output, const float* input,
                               const float* weight, const float* bias,
                               int B, int T, int C, int intermediate_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;

    int b = idx / (T * C);
    int t = (idx / C) % T;
    int j = idx % C;

    float sum = bias[j];  // Start with bias

    // Matrix multiplication
    for (int i = 0; i < intermediate_size; i++) {
        int input_idx = b * T * intermediate_size + t * intermediate_size + i;
        int weight_idx = i * C + j;
        sum += input[input_idx] * weight[weight_idx];
    }

    output[idx] = sum;
}

// ---------------------------------------------------------------------------
// Helper function for matrix multiplication using cuBLAS
// ---------------------------------------------------------------------------
void matmul(float* C, const float* A, const float* B, 
          int M, int N, int K, cublasHandle_t handle, cudaStream_t stream = 0) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetStream(handle, stream);

    // Compute C = A * B
    // Using GEMM: M x K @ K x N -> M x N
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, 
                B, N,  // Note: cuBLAS uses column-major order
                A, K, 
                &beta, 
                C, N);

    cudaCheck(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Transformer Block Structure
// ---------------------------------------------------------------------------
typedef struct {
    // Dimensions
    int B;  // Batch size
    int T;  // Sequence length
    int C;  // Hidden size
    int NH; // Number of heads
    int HS; // Head size (C / NH)
    int intermediate_size; // MLP hidden size (typically 4*C)

    // Attention weights
    float* query_weight;
    float* key_weight;
    float* value_weight;
    float* attn_output_weight;

    // LayerNorm weights
    float* ln1_weight;
    float* ln1_bias;
    float* ln2_weight;
    float* ln2_bias;

    // MLP weights
    float* mlp_fc_weight;
    float* mlp_fc_bias;
    float* mlp_proj_weight;
    float* mlp_proj_bias;

    // cuBLAS handle
    cublasHandle_t cublas_handle;
} TransformerBlock;

// ---------------------------------------------------------------------------
// Initialize the transformer block
// ---------------------------------------------------------------------------
void transformer_block_init(TransformerBlock* block, int B, int T, int C, int NH) {
    // Set dimensions
    block->B = B;
    block->T = T;
    block->C = C;
    block->NH = NH;
    block->HS = C / NH;
    block->intermediate_size = 4 * C;  // Standard expansion factor

    // Initialize cuBLAS
    cublasCreate(&block->cublas_handle);

    // Allocate memory for weights
    size_t attn_weight_size = NH * (C / NH) * C * sizeof(float);
    size_t ln_weight_size = C * sizeof(float);
    size_t mlp_fc_weight_size = C * (4 * C) * sizeof(float);
    size_t mlp_proj_weight_size = (4 * C) * C * sizeof(float);
    size_t mlp_fc_bias_size = (4 * C) * sizeof(float);
    size_t mlp_proj_bias_size = C * sizeof(float);

    // Attention weights
    cudaMalloc(&block->query_weight, attn_weight_size);
    cudaMalloc(&block->key_weight, attn_weight_size);
    cudaMalloc(&block->value_weight, attn_weight_size);
    cudaMalloc(&block->attn_output_weight, attn_weight_size);

    // Layer norm weights
    cudaMalloc(&block->ln1_weight, ln_weight_size);
    cudaMalloc(&block->ln1_bias, ln_weight_size);
    cudaMalloc(&block->ln2_weight, ln_weight_size);
    cudaMalloc(&block->ln2_bias, ln_weight_size);

    // MLP weights
    cudaMalloc(&block->mlp_fc_weight, mlp_fc_weight_size);
    cudaMalloc(&block->mlp_fc_bias, mlp_fc_bias_size);
    cudaMalloc(&block->mlp_proj_weight, mlp_proj_weight_size);
    cudaMalloc(&block->mlp_proj_bias, mlp_proj_bias_size);
}

// ---------------------------------------------------------------------------
// Free transformer block resources
// ---------------------------------------------------------------------------
void transformer_block_free(TransformerBlock* block) {
    // Free attention weights
    cudaFree(block->query_weight);
    cudaFree(block->key_weight);
    cudaFree(block->value_weight);
    cudaFree(block->attn_output_weight);

    // Free layer norm weights
    cudaFree(block->ln1_weight);
    cudaFree(block->ln1_bias);
    cudaFree(block->ln2_weight);
    cudaFree(block->ln2_bias);

    // Free MLP weights
    cudaFree(block->mlp_fc_weight);
    cudaFree(block->mlp_fc_bias);
    cudaFree(block->mlp_proj_weight);
    cudaFree(block->mlp_proj_bias);

    // Destroy cuBLAS handle
    cublasDestroy(block->cublas_handle);
}

// ---------------------------------------------------------------------------
// Forward pass through the transformer block
// ---------------------------------------------------------------------------
void transformer_block_forward(TransformerBlock* block, float* output, float* input, cudaStream_t stream = 0) {
    int B = block->B;
    int T = block->T;
    int C = block->C;
    int NH = block->NH;
    int HS = block->HS;
    int intermediate_size = block->intermediate_size;

    // Allocate temporary buffers
    float *x1, *x2, *q, *k, *v, *attn_out, *mlp_hidden, *mlp_out;
    cudaMalloc(&x1, B * T * C * sizeof(float));
    cudaMalloc(&x2, B * T * C * sizeof(float));
    cudaMalloc(&q, B * NH * T * HS * sizeof(float));
    cudaMalloc(&k, B * NH * T * HS * sizeof(float));
    cudaMalloc(&v, B * NH * T * HS * sizeof(float));
    cudaMalloc(&attn_out, B * NH * T * HS * sizeof(float));
    cudaMalloc(&mlp_hidden, B * T * intermediate_size * sizeof(float));
    cudaMalloc(&mlp_out, B * T * C * sizeof(float));

    // Layer Norm 1
    layernorm_forward(x1, nullptr, nullptr, input, block->ln1_weight, block->ln1_bias, B, T, C, stream);

    // Self-Attention
    // 1. QKV Transformation
    int block_size = 256;
    int total_elements = B * T * NH * HS;
    int grid_size = CEIL_DIV(total_elements, block_size);
    qkv_transform_kernel<<<grid_size, block_size, 0, stream>>>(
        q, k, v, x1, block->query_weight, block->key_weight, block->value_weight, B, T, C, NH, HS);

    // 2. Attention
    int TILE_SIZE = 16;  // Adjust based on your GPU's shared memory
    size_t shared_mem_size = 2 * TILE_SIZE * HS * sizeof(float);
    dim3 grid(1, T, B * NH);

    tiled_flash_attention_kernel<<<grid, block_size, shared_mem_size, stream>>>(
        attn_out, q, k, v, B, NH, T, HS, TILE_SIZE);
    // dim3 grid(CEIL_DIV(HS, block_size), T, B * NH);
    // simple_attention_kernel<<<grid, block_size, 0, stream>>>(attn_out, q, k, v, B, NH, T, HS);

    // size_t shared_mem_size = 2 * T * HS * sizeof(float); // For k_cache, v_cache
    // dim3 grid(1, T, B * NH);
    // flash_attention_kernel<<<grid, block_size, shared_mem_size, stream>>>(
    //     attn_out, q, k, v, B, NH, T, HS);

    // 3. Output Projection
    grid_size = CEIL_DIV(C, block_size);
    dim3 output_grid(grid_size, T, B);
    attention_output_kernel<<<output_grid, block_size, 0, stream>>>(
        x2, attn_out, block->attn_output_weight, B, T, C, NH, HS);

    // Residual Connection 1
    residual_forward(x1, input, x2, B * T * C, stream);

    // Layer Norm 2
    layernorm_forward(x2, nullptr, nullptr, x1, block->ln2_weight, block->ln2_bias, B, T, C, stream);

    // MLP
    // 1. FC Layer
    grid_size = CEIL_DIV(B * T * intermediate_size, block_size);
    mlp_fc_kernel<<<grid_size, block_size, 0, stream>>>(
        mlp_hidden, x2, block->mlp_fc_weight, block->mlp_fc_bias, B, T, C, intermediate_size);

    // 2. GELU Activation
    gelu_forward(mlp_hidden, mlp_hidden, B * T * intermediate_size, stream);

    // 3. Projection
    grid_size = CEIL_DIV(B * T * C, block_size);
    mlp_proj_kernel<<<grid_size, block_size, 0, stream>>>(
        mlp_out, mlp_hidden, block->mlp_proj_weight, block->mlp_proj_bias, B, T, C, intermediate_size);

    // Residual Connection 2
    residual_forward(output, x1, mlp_out, B * T * C, stream);

    // Free temporary buffers
    cudaFree(x1);
    cudaFree(x2);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(attn_out);
    cudaFree(mlp_hidden);
    cudaFree(mlp_out);
}

// ---------------------------------------------------------------------------
// Initialize weights with random values (for testing)
// ---------------------------------------------------------------------------
void init_random_weights(TransformerBlock* block) {
    int C = block->C;
    int NH = block->NH;
    int HS = block->HS;
    int intermediate_size = block->intermediate_size;

    // Temporary host buffers
    float *h_attn_weight = new float[NH * HS * C];
    float *h_ln_weight = new float[C];
    float *h_ln_bias = new float[C];
    float *h_mlp_fc_weight = new float[C * intermediate_size];
    float *h_mlp_fc_bias = new float[intermediate_size];
    float *h_mlp_proj_weight = new float[intermediate_size * C];
    float *h_mlp_proj_bias = new float[C];

    // Initialize with random values
    // For attention weights: normal distribution scaled by 1/sqrt(head_size)
    for (int i = 0; i < NH * HS * C; i++) {
        h_attn_weight[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) / sqrtf(HS);
    }

    // For layer norm: weights=1, bias=0 (identity transformation initially)
    for (int i = 0; i < C; i++) {
        h_ln_weight[i] = 1.0f;
        h_ln_bias[i] = 0.0f;
    }

    // For MLP weights: normal distribution scaled appropriately
    for (int i = 0; i < C * intermediate_size; i++) {
        h_mlp_fc_weight[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) / sqrtf(C);
    }

    for (int i = 0; i < intermediate_size; i++) {
        h_mlp_fc_bias[i] = 0.0f;  // Zero bias initially
    }

    for (int i = 0; i < intermediate_size * C; i++) {
        h_mlp_proj_weight[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) / sqrtf(intermediate_size);
    }

    for (int i = 0; i < C; i++) {
        h_mlp_proj_bias[i] = 0.0f;  // Zero bias initially
    }

    // Copy to device
    cudaMemcpy(block->query_weight, h_attn_weight, NH * HS * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(block->key_weight, h_attn_weight, NH * HS * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(block->value_weight, h_attn_weight, NH * HS * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(block->attn_output_weight, h_attn_weight, NH * HS * C * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(block->ln1_weight, h_ln_weight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(block->ln1_bias, h_ln_bias, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(block->ln2_weight, h_ln_weight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(block->ln2_bias, h_ln_bias, C * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(block->mlp_fc_weight, h_mlp_fc_weight, C * intermediate_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(block->mlp_fc_bias, h_mlp_fc_bias, intermediate_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(block->mlp_proj_weight, h_mlp_proj_weight, intermediate_size * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(block->mlp_proj_bias, h_mlp_proj_bias, C * sizeof(float), cudaMemcpyHostToDevice);

    // Free host memory
    delete[] h_attn_weight;
    delete[] h_ln_weight;
    delete[] h_ln_bias;
    delete[] h_mlp_fc_weight;
    delete[] h_mlp_fc_bias;
    delete[] h_mlp_proj_weight;
    delete[] h_mlp_proj_bias;
}

// ---------------------------------------------------------------------------
// Main Function
// ---------------------------------------------------------------------------
int main() {
    // Model dimensions
    int B = 1;      // Batch size
    int T = 128;    // Sequence length
    int C = 768;    // Hidden size
    int NH = 12;    // Number of heads

    // Allocate host memory for input and output
    size_t tensor_size = B * T * C * sizeof(float);
    float *h_input = new float[B * T * C];
    float *h_output = new float[B * T * C];

    // Initialize input with random values
    for (int i = 0; i < B * T * C; i++) {
        h_input[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f);  // Random values in [-1, 1]
    }

    // Allocate device memory for input and output
    float *d_input, *d_output;
    cudaMalloc(&d_input, tensor_size);
    cudaMalloc(&d_output, tensor_size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, tensor_size, cudaMemcpyHostToDevice);

    // Create and initialize transformer block
    TransformerBlock block;
    transformer_block_init(&block, B, T, C, NH);
    init_random_weights(&block);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Forward pass
    printf("Running transformer block forward pass...\n");
    cudaEventRecord(start);

    transformer_block_forward(&block, d_output, d_input);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output back to host
    cudaMemcpy(h_output, d_output, tensor_size, cudaMemcpyDeviceToHost);

    // Print timing information
    printf("Transformer block forward pass complete.\n");
    printf("Time taken: %.3f ms\n", milliseconds);
    printf("Batch size: %d, Sequence length: %d, Hidden size: %d, Heads: %d\n", B, T, C, NH);
    printf("Tokens per second: %.1f\n", (B * T * 1000.0f) / milliseconds);

    // Validate output (simple check for NaN or inf)
    bool output_valid = true;
    for (int i = 0; i < 100; i++) {
        if (isnan(h_output[i]) || isinf(h_output[i])) {
            printf("Warning: Invalid output value detected at index %d: %f\n", i, h_output[i]);
            output_valid = false;
            break;
        }
    }

    if (output_valid) {
        printf("Output validation passed (sample check).\n");

        // Print a few values from the output
        printf("Sample output values:\n");
        for (int i = 0; i < 5; i++) {
            printf("[%d]: %f\n", i, h_output[i]);
        }
    }

    // Clean up
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    transformer_block_free(&block);

    return 0;
}

/* 
Compilation and Run:
$ nvcc -o transformer_block 10_transformer_block.cu -lcublas
$ ./transformer_block

Experiment Ideas:
1. Change batch size, sequence length, hidden size and number of heads to see performance impact
2. Profile memory usage for different model sizes
3. Try implementing a version with half-precision (FP16) for better performance
4. Modify the attention mechanism to support different attention patterns (e.g., local attention)
5. Use cuDNN for some operations to potentially improve performance
*/
