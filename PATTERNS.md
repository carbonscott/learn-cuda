# The Definitive CUDA Programming Patterns and Anti-Patterns Reference Guide

## Part I: Anti-Patterns - What NOT to Do

### 1. Thread Execution Anti-Patterns

#### 1.1 Warp Divergence from Conditional Branching

**TLDR**: Avoid thread-ID-based or data-dependent branching that causes threads within a warp to execute different code paths.

**Problem Statement**: CUDA executes threads in groups of 32 (warps) using SIMT architecture. When threads diverge, both paths execute serially, wasting compute resources.

**Bad Code Example**:
```cuda
// ANTI-PATTERN: Thread divergence
__global__ void divergentKernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x % 2 == 0) {
        data[idx] = expensiveComputation1(data[idx]);
    } else {
        data[idx] = expensiveComputation2(data[idx]);
    }
}
```

**Good Code Example**:
```cuda
// PATTERN: Warp-aligned branching
__global__ void alignedKernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadIdx.x / 32;
    
    // All threads in warp take same branch
    if (warpId % 2 == 0) {
        data[idx] = computation1(data[idx]);
    } else {
        data[idx] = computation2(data[idx]);
    }
}
```

**Performance Impact**: 
- Worst case: **32x slowdown** when all threads take different paths
- Typical: **3x performance degradation** for 50/50 branch splits
- Detection: Monitor `branch_efficiency` metric in Nsight Compute

**When to Apply**: Always structure conditionals to minimize intra-warp divergence. Use predication for short branches.

**Related Patterns**: Warp-level primitives, cooperative groups

---

#### 1.2 Occupancy-Killing Resource Usage

**TLDR**: Excessive register or shared memory usage limits concurrent thread blocks, reducing latency hiding.

**Problem Statement**: Each SM has limited resources. Overuse prevents multiple blocks from running concurrently.

**Bad Code Example**:
```cuda
// ANTI-PATTERN: Excessive registers
__global__ void heavyKernel(float* data) {
    float localArray[64];  // Forces register spilling
    // Complex computation using many local variables
}
```

**Good Code Example**:
```cuda
// PATTERN: Controlled resource usage
__launch_bounds__(256, 2)  // Limit registers
__global__ void optimizedKernel(float* data) {
    __shared__ float sharedData[256];  // Use shared memory strategically
    // Balanced resource utilization
}
```

**Performance Impact**:
- Going from 32 to 33 registers per thread: **25% occupancy drop** on V100
- Optimal occupancy range: **50-66%** for most kernels
- Detection: Use `nvcc --ptxas-options=-v` to check register usage

**When to Apply**: Always monitor and control resource usage. Use `__launch_bounds__` for fine control.

---

### 2. Memory Access Anti-Patterns

#### 2.1 Uncoalesced Global Memory Access

**TLDR**: Non-consecutive memory accesses by consecutive threads destroy memory bandwidth.

**Problem Statement**: GPUs achieve peak bandwidth when consecutive threads access consecutive memory addresses.

**Bad Code Example**:
```cuda
// ANTI-PATTERN: Strided access
__global__ void stridedAccess(float* data, int stride) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride;
    data[idx] = data[idx] * 2.0f;  // Non-coalesced
}
```

**Good Code Example**:
```cuda
// PATTERN: Coalesced access
__global__ void coalescedAccess(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;  // Perfect coalescing
}
```

**Performance Impact**:
- Coalesced: **~790 GB/s** on V100
- Stride-2: **~400 GB/s** (50% efficiency)
- Random: **<100 GB/s** (>8x slowdown)

**When to Apply**: Always ensure coalesced access patterns. Use SoA instead of AoS for data structures.

---

#### 2.2 Shared Memory Bank Conflicts

**TLDR**: Multiple threads accessing different addresses in the same memory bank causes serialization.

**Problem Statement**: Shared memory has 32 banks. Conflicts occur when threads access same bank at different addresses.

**Bad Code Example**:
```cuda
// ANTI-PATTERN: Bank conflicts
__global__ void bankConflicts() {
    __shared__ float data[32][32];
    int tid = threadIdx.x;
    data[tid][0] = 1.0f;  // All threads hit bank 0
}
```

**Good Code Example**:
```cuda
// PATTERN: Padding to avoid conflicts
__global__ void noBankConflicts() {
    __shared__ float data[32][33];  // +1 padding
    int tid = threadIdx.x;
    data[tid][0] = 1.0f;  // Different banks
}
```

**Performance Impact**:
- 4-way conflict: **4x slowdown**
- 32-way conflict: **32x slowdown** (complete serialization)

**When to Apply**: Add padding to 2D shared memory arrays. Use different access patterns for columns vs rows.

---

### 3. Kernel Launch Anti-Patterns

#### 3.1 Poor Grid and Block Dimensions

**TLDR**: Suboptimal thread block sizes waste GPU resources and reduce performance.

**Problem Statement**: Block sizes must be multiples of 32 and balance resource usage with occupancy.

**Bad Code Example**:
```cuda
// ANTI-PATTERN: Poor dimensions
dim3 block(30, 1);  // Not warp-aligned
kernel<<<grid, block>>>(data);
```

**Good Code Example**:
```cuda
// PATTERN: Optimal dimensions
dim3 block(256, 1);  // 8 warps, good occupancy
kernel<<<grid, block>>>(data);
```

**Performance Impact**:
- Non-warp-aligned: Wastes threads, reduces efficiency
- Too small (<64 threads): Poor latency hiding
- Too large (>512 threads): May limit occupancy

**When to Apply**: Start with 128-256 threads per block. Always use multiples of 32.

---

## Part II: Performance Patterns - What TO Do

### 1. Warp Execution Patterns

#### 1.1 Warp-Level Primitives for Fast Communication

**TLDR**: Use shuffle instructions for register-to-register data exchange within warps.

**Problem Statement**: Intra-warp communication through shared memory adds latency and uses resources.

**Implementation**:
```cuda
// PATTERN: Warp reduction using shuffle
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

**Performance Impact**:
- **10x faster** than shared memory for warp-level operations
- Zero shared memory usage increases occupancy
- Hardware-accelerated on Kepler+ architectures

**When to Apply**: Use for reductions, scans, and any intra-warp communication.

**Related Patterns**: Cooperative groups, hierarchical reductions

---

### 2. Memory Coalescing Patterns

#### 2.1 Structure of Arrays (SoA) for Optimal Access

**TLDR**: Organize data so consecutive threads access consecutive memory locations.

**Problem Statement**: Array of Structures (AoS) causes strided memory access, destroying bandwidth.

**Implementation**:
```cuda
// PATTERN: SoA layout
struct ParticlesSoA {
    float *x, *y, *z;     // Positions
    float *vx, *vy, *vz;  // Velocities
};

__global__ void updateSoA(ParticlesSoA particles, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        particles.x[idx] += particles.vx[idx];  // Coalesced
        particles.y[idx] += particles.vy[idx];
        particles.z[idx] += particles.vz[idx];
    }
}
```

**Performance Impact**:
- SoA: **Full bandwidth utilization** (~790 GB/s on V100)
- AoS: **3x slower** due to strided access
- Enables vectorized loads (float4) for additional speedup

**When to Apply**: Always use SoA for performance-critical kernels accessing multiple fields.

---

### 3. Shared Memory Patterns

#### 3.1 Tiled Matrix Multiplication

**TLDR**: Use shared memory tiles to reduce global memory traffic and achieve near-peak performance.

**Problem Statement**: Naive matrix multiplication has O(n³) memory accesses for O(n³) compute.

**Implementation**:
```cuda
#define TILE_SIZE 16

__global__ void tiledMatMul(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; k++) {
        // Load tiles cooperatively
        if (row < N && k * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + k * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && k * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute on shared memory tiles
        for (int j = 0; j < TILE_SIZE; j++)
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

**Performance Impact**:
- **4-20x speedup** over naive implementation
- Achieves **80-95% of cuBLAS performance** with optimizations
- Reduces memory bandwidth requirement by tile factor

**When to Apply**: Use for any algorithm with data reuse patterns.

---

### 4. Reduction Patterns

#### 4.1 Hierarchical Reduction with Register Blocking

**TLDR**: Combine sequential work per thread with parallel tree reduction for optimal efficiency.

**Problem Statement**: Simple parallel reduction has O(n log n) work complexity instead of optimal O(n).

**Implementation**:
```cuda
// PATTERN: Work-efficient reduction
template<int BLOCK_SIZE>
__global__ void efficientReduce(float *input, float *output, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (BLOCK_SIZE * 2) + tid;
    
    // Sequential reduction in registers (register blocking)
    float sum = 0;
    while (idx < N) {
        sum += input[idx];
        if (idx + BLOCK_SIZE < N)
            sum += input[idx + BLOCK_SIZE];
        idx += gridDim.x * BLOCK_SIZE * 2;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int s = BLOCK_SIZE/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp reduction
    if (tid < 32) {
        volatile float *vdata = sdata;
        vdata[tid] += vdata[tid + 32];
        vdata[tid] += vdata[tid + 16];
        vdata[tid] += vdata[tid + 8];
        vdata[tid] += vdata[tid + 4];
        vdata[tid] += vdata[tid + 2];
        vdata[tid] += vdata[tid + 1];
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

**Performance Impact**:
- **30x improvement** over naive reduction
- Achieves **>60 GB/s** bandwidth utilization
- Work-efficient O(n) operations

**When to Apply**: Use for any commutative and associative operations (sum, max, min).

---

### 5. Scan/Prefix Sum Patterns

#### 5.1 Work-Efficient Parallel Scan (Blelloch Algorithm)

**TLDR**: Two-phase algorithm achieving O(n) work complexity for prefix sums.

**Problem Statement**: Naive scan has O(n log n) work complexity, inefficient for large arrays.

**Implementation Overview**:
```cuda
// PATTERN: Work-efficient scan
// Phase 1: Up-sweep (reduce)
for (int d = 0; d < log2n; d++) {
    // Parallel: Build partial sums in tree
}

// Phase 2: Down-sweep  
for (int d = log2n-1; d >= 0; d--) {
    // Parallel: Propagate sums down tree
}
```

**Performance Impact**:
- O(n) work complexity vs O(n log n) for naive
- CUB implementation approaches memory bandwidth limits
- Essential for stream compaction, sorting, and many parallel algorithms

**When to Apply**: Use CUB's DeviceScan for production code. Implement custom versions for specialized data types.

---

### 6. Stream Processing Patterns

#### 6.1 Asynchronous Execution with Multiple Streams

**TLDR**: Overlap computation with memory transfers using CUDA streams.

**Problem Statement**: Serial execution wastes GPU resources during memory transfers.

**Implementation**:
```cuda
// PATTERN: Multi-stream pipeline
void pipelinedExecution(float *h_data, float *d_data, size_t size, int chunks) {
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) 
        cudaStreamCreate(&streams[i]);
    
    size_t chunkSize = size / chunks;
    
    for (int i = 0; i < chunks; i++) {
        int stream = i % 4;
        size_t offset = i * chunkSize;
        
        // Async memory transfer
        cudaMemcpyAsync(d_data + offset, h_data + offset, 
                       chunkSize * sizeof(float),
                       cudaMemcpyHostToDevice, streams[stream]);
        
        // Launch kernel on same stream
        processKernel<<<grid, block, 0, streams[stream]>>>(d_data + offset);
        
        // Async copy back
        cudaMemcpyAsync(h_data + offset, d_data + offset,
                       chunkSize * sizeof(float),
                       cudaMemcpyDeviceToHost, streams[stream]);
    }
}
```

**Performance Impact**:
- **Up to 2x speedup** by hiding transfer latency
- Requires pinned memory for true async transfers
- Enables full GPU utilization during I/O

**When to Apply**: Use for any application with significant data transfer requirements.

---

### 7. Multi-GPU Patterns

#### 7.1 Peer-to-Peer Communication

**TLDR**: Direct GPU-to-GPU transfers without CPU involvement.

**Problem Statement**: Transferring through CPU memory adds latency and reduces bandwidth.

**Implementation**:
```cuda
// PATTERN: P2P GPU communication
void enableP2P(int gpu1, int gpu2) {
    cudaSetDevice(gpu1);
    cudaDeviceEnablePeerAccess(gpu2, 0);
    
    cudaSetDevice(gpu2);
    cudaDeviceEnablePeerAccess(gpu1, 0);
}

// Direct GPU-to-GPU copy
cudaMemcpyPeerAsync(d_dst, gpuDst, d_src, gpuSrc, size, stream);
```

**Performance Impact**:
- **2-4x faster** than routing through host memory
- NVLink provides **300 GB/s** bidirectional bandwidth
- PCIe provides **32 GB/s** bidirectional bandwidth

**When to Apply**: Essential for multi-GPU algorithms requiring data exchange.

---

## Part III: Golden Rules Summary

### The 10 Commandments of CUDA Programming

1. **Thou Shalt Coalesce Memory Access**
   - Consecutive threads MUST access consecutive memory addresses
   - Performance penalty: 2-10x for violations

2. **Thou Shalt Minimize Warp Divergence**
   - Structure conditionals to keep warps uniform
   - Performance penalty: Up to 32x for complete divergence

3. **Thou Shalt Use Shared Memory Wisely**
   - Cache frequently accessed data in shared memory
   - Avoid bank conflicts with proper access patterns

4. **Thou Shalt Balance Resources for Occupancy**
   - Target 50-66% occupancy for latency hiding
   - Control register and shared memory usage

5. **Thou Shalt Use Appropriate Memory Types**
   - Registers > Shared > L1/L2 > Global memory
   - Match memory type to access pattern

6. **Thou Shalt Profile Before Optimizing**
   - Use Nsight Compute for detailed analysis
   - Focus on bottlenecks, not assumptions

7. **Thou Shalt Use Architecture-Specific Features**
   - Tensor Cores for AI workloads
   - Warp-level primitives for reductions
   - Async copy for memory operations

8. **Thou Shalt Prefer SoA Over AoS**
   - Structure data for coalesced access
   - Performance gain: 3x typical

9. **Thou Shalt Use Existing Libraries When Possible**
   - cuBLAS, cuDNN, CUB, Thrust are highly optimized
   - Reinventing rarely beats library performance

10. **Thou Shalt Test Across Architectures**
    - Performance characteristics vary by GPU generation
    - Optimize for your target deployment hardware

### Performance Hierarchy (Most to Least Important)

1. **Memory Coalescing** (2-10x impact)
2. **Occupancy Optimization** (2-3x impact)
3. **Shared Memory Usage** (2-5x impact)
4. **Warp Divergence** (1.5-32x impact, application-dependent)
5. **Memory Transfer Optimization** (1.5-2x impact)
6. **Instruction-Level Optimizations** (1.1-1.3x impact)

### Quick Decision Tree

```
Is kernel memory-bound?
├─ YES → Focus on coalescing, shared memory, memory hierarchy
└─ NO → Is it compute-bound?
    ├─ YES → Optimize arithmetic intensity, use Tensor Cores
    └─ NO → Check for:
        ├─ Launch overhead → Batch operations
        ├─ Synchronization → Minimize sync points
        └─ Load imbalance → Dynamic scheduling
```

## Architecture-Specific Quick Reference

### Volta/Turing (CC 7.0-7.5)
- Tensor Cores: Use for FP16/FP32 mixed precision
- Independent Thread Scheduling: Safer but watch convergence
- 96KB shared memory per SM

### Ampere (CC 8.0-8.6)
- Async copy instructions: Use for pipelined operations
- 164KB shared memory per SM (A100)
- Third-gen Tensor Cores: Support for TF32, BF16

### Ada Lovelace (CC 8.9)
- 98MB L2 cache: Leverage for persistent data
- Fourth-gen Tensor Cores: FP8 support
- Enhanced ray tracing cores

### Hopper (CC 9.0)
- Thread Block Clusters: Inter-SM communication
- Distributed Shared Memory: 228KB per SM group
- Tensor Memory Accelerator: Hardware-accelerated tensor operations
- DPX instructions: 40x speedup for dynamic programming

## Conclusion

This comprehensive guide provides the essential patterns and anti-patterns for CUDA programming success. The key principles are:

1. **Understand the hardware**: SIMT execution, memory hierarchy, architectural features
2. **Measure everything**: Profile before and after optimizations
3. **Start with memory**: Most performance issues are memory-related
4. **Use the right abstraction level**: From low-level PTX to high-level libraries
5. **Iterate and refine**: Optimization is an iterative process

Following these patterns and avoiding the anti-patterns will help achieve 80-95% of theoretical peak performance for most CUDA applications. Remember: premature optimization is the root of all evil, but understanding these patterns from the start prevents fundamental design mistakes that are costly to fix later.
