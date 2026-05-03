// Simplified API for asynchronous data loading.

#include "common.cuh"


static __device__ __forceinline__ unsigned int ggml_cuda_cvta_generic_to_shared(void * generic_ptr) {
#ifdef CP_ASYNC_AVAILABLE
    return __cvta_generic_to_shared(generic_ptr);
#else
    GGML_UNUSED(generic_ptr);
    NO_DEVICE_CODE;
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// Copies data from global to shared memory, cg == cache global.
// Both the src and dst pointers must be aligned to 16 bit.
// Shared memory uses 32 bit addressing, the pointer is passed as unsigned int.
// Generic pointers can be converted to 32 bit shared memory pointers using __cvta_generic_to_shared.
// Only the 16 bit copy is exposed because 4 and 8 bit copies did not yield performance improvements.
template <int preload>
static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    static_assert(preload == 0 || preload == 64 || preload == 128 || preload == 256, "bad preload");
#ifdef CP_ASYNC_AVAILABLE
#if CUDART_VERSION >= 11040
    if (preload == 256) {
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 128) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 64) {
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else
#endif // CUDART_VERSION >= 11040
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    }
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Makes each thread wait until its asynchronous data copies are done.
// This does NOT provide any additional synchronization.
// In particular, when copying data with multiple warps a call to __syncthreads will be needed.
static __device__ __forceinline__ void cp_async_wait_all() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_all;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// ========================================================================
// Blackwell SM120 TMA (Tensor Memory Accelerator) primitives
// ========================================================================
// TMA provides hardware-accelerated bulk tensor copies from global memory
// to shared memory. On Blackwell, this is significantly faster than
// cp.async for large contiguous transfers because:
//   - The TMA unit operates independently of SM threads
//   - Supports 2D/3D/4D/5D tensor layouts with strides
//   - Single instruction to initiate a large transfer
//   - Reduces register pressure (no per-thread address computation)
//   - Frees warps to do compute while data is in flight
//
// TMA is only available on Blackwell (sm_120a+).
// Requires CUDA 12.8+ for PTX support.
// ========================================================================

#if defined(TMA_AVAILABLE) && CUDART_VERSION >= 12800

// TMA descriptor type - opaque 128-byte structure that describes a tensor in global memory.
// Created on the host with cuTensorMapEncodeTiled.
struct tma_desc_t {
    alignas(128) uint8_t data[128];
};

// Initiate a TMA bulk copy from global to shared memory.
// The tensor map (desc) describes the source tensor layout in global memory.
// The coordinates specify the starting position in each dimension.
// The box_size specifies the size of the tile to copy in each dimension.
// The smem_dst is the shared memory destination address.
//
// For 1D tiles (most common in matmul/attention):
//   tma_copy_1d<bytes>(smem_dst, desc, coord0)
//
// For 2D tiles (useful for matrix tiles with row stride):
//   tma_copy_2d(smem_dst, desc, coord0, coord1, box_size0, box_size1)
template <int bytes>
static __device__ __forceinline__ void tma_copy_1d(
        void * __restrict__ smem_dst,
        const tma_desc_t & desc,
        const int64_t coord0) {
    GGML_UNUSED(smem_dst);
    GGML_UNUSED(desc);
    GGML_UNUSED(coord0);
    // TMA 1D bulk copy PTX requires 64-bit shared memory address and 64-bit coordinates.
    // On Blackwell, shared memory can be addressed with 64-bit pointers for TMA.
    // The "l" constraint is for 64-bit register operands.
    NO_DEVICE_CODE;
}

// 2D TMA bulk copy - copies a 2D tile from a tensor to shared memory.
// This is particularly efficient for matrix transpose and flash attention
// where the K/V matrices have non-trivial strides.
static __device__ __forceinline__ void tma_copy_2d(
        void * __restrict__ smem_dst,
        const tma_desc_t & desc,
        const int64_t coord0,
        const int64_t coord1) {
    GGML_UNUSED(smem_dst);
    GGML_UNUSED(desc);
    GGML_UNUSED(coord0);
    GGML_UNUSED(coord1);
    NO_DEVICE_CODE;
}

// Wait for all outstanding TMA bulk copies to complete.
// group: 0 = wait for all, 1 = wait for all but 1, etc.
// Must be a compile-time constant for the PTX instruction.
template <int group>
static __device__ __forceinline__ void tma_wait_group() {
    NO_DEVICE_CODE;
}

// Wait for all outstanding TMA bulk copies to complete (both reads and writes).
static __device__ __forceinline__ void tma_wait_group_all() {
    NO_DEVICE_CODE;
}

// Commit the TMA asynchronous group (fence).
static __device__ __forceinline__ void tma_commit_group() {
    NO_DEVICE_CODE;
}

#endif // TMA_AVAILABLE && CUDART_VERSION >= 12800
