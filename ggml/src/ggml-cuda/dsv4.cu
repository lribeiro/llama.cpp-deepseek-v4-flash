#include "dsv4.cuh"
#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"

// ============================================================================
// DSV4 RoPE Tail - applies RoPE only to the tail (rotary) dimensions,
// leaving the non-RoPE prefix dimensions unchanged.
// Follows the Metal kernel structure closely for correctness.
// ============================================================================

struct rope_corr_dims {
    float v[2];
};

// Host-side YaRN correction dims computation (matches ggml.c)
static float dsv4_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2.0f * (float)M_PI)) / (2.0f * logf(base));
}

static void dsv4_rope_yarn_corr_dims(
        int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow,
        rope_corr_dims & dims) {
    float start = floorf(dsv4_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(dsv4_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims.v[0] = fmaxf(0.0f, start);
    dims.v[1] = fminf((float)(n_dims - 1), end);
}

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

template<bool forward>
static __device__ void rope_yarn(
        float theta_extrap, float freq_scale, const rope_corr_dims & corr_dims,
        int64_t i0, float ext_factor, float mscale,
        float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

// F32 kernel - follows the Metal dsv4_rope_tail_f32 structure exactly.
// Each thread processes one element i0 in [0, ne00), for a given (i1, i2, i3) row.
// Threadgroup: one threadgroup per (i1, i2, i3), threads iterate over i0.
template <bool forward, bool has_ff>
static __global__ void dsv4_rope_tail_f32(
        const float * x, float * dst,
        const int ne00, const int ne01, const int ne02,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const uint64_t nb0,  const uint64_t nb1,  const uint64_t nb2,  const uint64_t nb3,
        const int n_dims, const int n_nope,
        const int32_t * pos,
        const float freq_base, const float freq_scale,
        const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const bool is_neox,
        const float * freq_factors) {
    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.z;

    if (i1 >= ne01 || i2 >= ne02) return;

    const char * src_row = (const char *) x + i3 * nb03 + i2 * nb02 + i1 * nb01;
    char * dst_row = (char *) dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    const float inv_ndims = -1.0f / n_dims;
    const int p = pos[i2];

    for (int i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
        // Copy non-RoPE dimensions as-is
        if (i0 < n_nope) {
            const float val = *(const float *)(src_row + i0 * nb00);
            *(float *)(dst_row + i0 * nb0) = val;
            continue;
        }

        const int r = i0 - n_nope;

        if (is_neox) {
            const int n_half = n_dims / 2;
            if (r >= n_half) continue;  // second half handled by first half

            const int ic = r;
            const int rel_i0 = 2 * ic;  // dimension index for theta
            const float theta = p * powf(freq_base, inv_ndims * rel_i0);
            const float freq_factor = has_ff ? freq_factors[ic] : 1.0f;

            float cos_theta, sin_theta;
            rope_yarn<forward>(theta / freq_factor, freq_scale, corr_dims, rel_i0, ext_factor, attn_factor, cos_theta, sin_theta);

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = *(const float *)(src_row + j0 * nb00);
            const float x1 = *(const float *)(src_row + j1 * nb00);

            *(float *)(dst_row + j0 * nb0) = x0 * cos_theta - x1 * sin_theta;
            *(float *)(dst_row + j1 * nb0) = x0 * sin_theta + x1 * cos_theta;
        } else {
            // Normal (interleaved) RoPE
            if ((r & 1) != 0) continue;  // odd indices handled by their pair

            const float theta = p * powf(freq_base, inv_ndims * r);
            const int ic = r / 2;
            const float freq_factor = has_ff ? freq_factors[ic] : 1.0f;

            float cos_theta, sin_theta;
            rope_yarn<forward>(theta / freq_factor, freq_scale, corr_dims, r, ext_factor, attn_factor, cos_theta, sin_theta);

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = *(const float *)(src_row + j0 * nb00);
            const float x1 = *(const float *)(src_row + j1 * nb00);

            *(float *)(dst_row + j0 * nb0) = x0 * cos_theta - x1 * sin_theta;
            *(float *)(dst_row + j1 * nb0) = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

// F16 kernel - same logic but reads/writes half
template <bool forward, bool has_ff>
static __global__ void dsv4_rope_tail_f16(
        const half * x, half * dst,
        const int ne00, const int ne01, const int ne02,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const uint64_t nb0,  const uint64_t nb1,  const uint64_t nb2,  const uint64_t nb3,
        const int n_dims, const int n_nope,
        const int32_t * pos,
        const float freq_base, const float freq_scale,
        const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const bool is_neox,
        const float * freq_factors) {
    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.z;

    if (i1 >= ne01 || i2 >= ne02) return;

    const char * src_row = (const char *) x + i3 * nb03 + i2 * nb02 + i1 * nb01;
    char * dst_row = (char *) dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    const float inv_ndims = -1.0f / n_dims;
    const int p = pos[i2];

    for (int i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
        if (i0 < n_nope) {
            const half val = *(const half *)(src_row + i0 * nb00);
            *(half *)(dst_row + i0 * nb0) = val;
            continue;
        }

        const int r = i0 - n_nope;

        if (is_neox) {
            const int n_half = n_dims / 2;
            if (r >= n_half) continue;

            const int ic = r;
            const int rel_i0 = 2 * ic;
            const float theta = p * powf(freq_base, inv_ndims * rel_i0);
            const float freq_factor = has_ff ? freq_factors[ic] : 1.0f;

            float cos_theta, sin_theta;
            rope_yarn<forward>(theta / freq_factor, freq_scale, corr_dims, rel_i0, ext_factor, attn_factor, cos_theta, sin_theta);

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = __half2float(*(const half *)(src_row + j0 * nb00));
            const float x1 = __half2float(*(const half *)(src_row + j1 * nb00));

            *(half *)(dst_row + j0 * nb0) = __float2half(x0 * cos_theta - x1 * sin_theta);
            *(half *)(dst_row + j1 * nb0) = __float2half(x0 * sin_theta + x1 * cos_theta);
        } else {
            if ((r & 1) != 0) continue;

            const float theta = p * powf(freq_base, inv_ndims * r);
            const int ic = r / 2;
            const float freq_factor = has_ff ? freq_factors[ic] : 1.0f;

            float cos_theta, sin_theta;
            rope_yarn<forward>(theta / freq_factor, freq_scale, corr_dims, r, ext_factor, attn_factor, cos_theta, sin_theta);

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = __half2float(*(const half *)(src_row + j0 * nb00));
            const float x1 = __half2float(*(const half *)(src_row + j1 * nb00));

            *(half *)(dst_row + j0 * nb0) = __float2half(x0 * cos_theta - x1 * sin_theta);
            *(half *)(dst_row + j1 * nb0) = __float2half(x0 * sin_theta + x1 * cos_theta);
        }
    }
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const int n_dims     = ((int32_t *) dst->op_params)[0];
    const int mode       = ((int32_t *) dst->op_params)[1];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[2];
    const bool inverse   = ((int32_t *) dst->op_params)[3] != 0;

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 9, sizeof(float));

    const int n_nope = src0->ne[0] - n_dims;
    const bool is_neox = (mode == GGML_ROPE_TYPE_NEOX);

    // Compute corr_dims on the host
    rope_corr_dims corr_dims;
    dsv4_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    cudaStream_t stream = ctx.stream();

    // Grid: one block per (i1, i2, i3), threads iterate over i0
    dim3 block_dims(256, 1, 1);
    dim3 grid_dims(src0->ne[1], src0->ne[2], src0->ne[3]);

    const float * freq_factors = src2 ? (const float *) src2->data : nullptr;

    if (src0->type == GGML_TYPE_F32) {
        if (!inverse) {
            if (freq_factors) {
                dsv4_rope_tail_f32<true, true><<<grid_dims, block_dims, 0, stream>>>(
                    (const float *) src0->data, (float *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
                    n_dims, n_nope,
                    (const int32_t *) src1->data,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    corr_dims, is_neox, freq_factors);
            } else {
                dsv4_rope_tail_f32<true, false><<<grid_dims, block_dims, 0, stream>>>(
                    (const float *) src0->data, (float *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
                    n_dims, n_nope,
                    (const int32_t *) src1->data,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    corr_dims, is_neox, nullptr);
            }
        } else {
            if (freq_factors) {
                dsv4_rope_tail_f32<false, true><<<grid_dims, block_dims, 0, stream>>>(
                    (const float *) src0->data, (float *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
                    n_dims, n_nope,
                    (const int32_t *) src1->data,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    corr_dims, is_neox, freq_factors);
            } else {
                dsv4_rope_tail_f32<false, false><<<grid_dims, block_dims, 0, stream>>>(
                    (const float *) src0->data, (float *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
                    n_dims, n_nope,
                    (const int32_t *) src1->data,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    corr_dims, is_neox, nullptr);
            }
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (!inverse) {
            if (freq_factors) {
                dsv4_rope_tail_f16<true, true><<<grid_dims, block_dims, 0, stream>>>(
                    (const half *) src0->data, (half *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
                    n_dims, n_nope,
                    (const int32_t *) src1->data,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    corr_dims, is_neox, freq_factors);
            } else {
                dsv4_rope_tail_f16<true, false><<<grid_dims, block_dims, 0, stream>>>(
                    (const half *) src0->data, (half *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
                    n_dims, n_nope,
                    (const int32_t *) src1->data,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    corr_dims, is_neox, nullptr);
            }
        } else {
            if (freq_factors) {
                dsv4_rope_tail_f16<false, true><<<grid_dims, block_dims, 0, stream>>>(
                    (const half *) src0->data, (half *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
                    n_dims, n_nope,
                    (const int32_t *) src1->data,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    corr_dims, is_neox, freq_factors);
            } else {
                dsv4_rope_tail_f16<false, false><<<grid_dims, block_dims, 0, stream>>>(
                    (const half *) src0->data, (half *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
                    n_dims, n_nope,
                    (const int32_t *) src1->data,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    corr_dims, is_neox, nullptr);
            }
        }
    } else {
        GGML_ABORT("dsv4_rope_tail: unsupported type %d", src0->type);
    }
}

// ============================================================================
// DSV4 FP8 KV Quantize - quantize non-RoPE dimensions to FP8 (E4M3FN) and
// dequantize back to F32, in blocks of 64 elements per row.
// ============================================================================

static __device__ float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * exp2f(float(exp - 7));
}

static __device__ float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int best = 0;
    float best_diff = ax;
    for (int i = 1; i < 127; ++i) {
        const float val = dsv4_e4m3fn_value(i);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }

    return sign * dsv4_e4m3fn_value(best);
}

// Each threadblock handles one row; 64 threads per block for the 64-element blocks
static __global__ void dsv4_fp8_kv_quantize_f32(
        const char * src0, char * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        uint64_t nb00, uint64_t nb01, uint64_t nb02, uint64_t nb03,
        uint64_t nb0,  uint64_t nb1,  uint64_t nb2,  uint64_t nb3,
        int32_t n_rot) {
    const int64_t row = blockIdx.x;
    const int64_t n_rows = ne01 * ne02 * ne03;
    if (row >= n_rows) return;

    const int64_t i1 = row % ne01;
    const int64_t i2 = (row / ne01) % ne02;
    const int64_t i3 = row / (ne01 * ne02);

    const char * src_base = src0 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    char * dst_base = dst + i1 * nb1 + i2 * nb2 + i3 * nb3;

    const int64_t n_nope = ne00 - n_rot;
    const int tid = threadIdx.x;

    // Process 64-element blocks for the non-RoPE portion
    for (int64_t off = 0; off < n_nope; off += 64) {
        // Each thread reads one element and computes abs
        __shared__ float s_abs[64];
        float v = 0.0f;
        if (tid < 64) {
            v = *(const float *)(src_base + (off + tid) * nb00);
            s_abs[tid] = fabsf(v);
        }
        __syncthreads();

        // Reduction to find max
        for (int stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_abs[tid] = fmaxf(s_abs[tid], s_abs[tid + stride]);
            }
            __syncthreads();
        }

        const float amax = fmaxf(s_abs[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));

        if (tid < 64) {
            const float q = dsv4_e4m3fn_dequant(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
            *(float *)(dst_base + (off + tid) * nb0) = q;
        }
        __syncthreads();
    }

    // Copy the RoPE dimensions as-is
    for (int64_t i = n_nope + tid; i < ne00; i += 64) {
        *(float *)(dst_base + i * nb0) = *(const float *)(src_base + i * nb00);
    }
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const int32_t n_rot = ggml_get_op_params_i32(dst, 0);

    const int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    cudaStream_t stream = ctx.stream();

    dsv4_fp8_kv_quantize_f32<<<n_rows, 64, 0, stream>>>(
        (const char *) src0->data, (char *) dst->data,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
        n_rot);
}

// ============================================================================
// DSV4 HC Split Sinkhorn - hyper-connection weight processing with Sinkhorn
// normalization. Each row is processed independently by one thread.
// ============================================================================

static __global__ void dsv4_hc_split_sinkhorn_kernel(
        const float * mixes, const float * scale, const float * base,
        float * dst,
        int32_t n_hc, int32_t sinkhorn_iters, float eps,
        int64_t n_rows, int64_t mix_hc,
        uint64_t mix_nb1, uint64_t dst_nb1) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rows) return;

    constexpr int HC_MAX = 16;
    if (n_hc <= 0 || n_hc > HC_MAX) return;

    const float * mix = mixes + tid * (mix_nb1 / sizeof(float));
    float * out = dst + tid * (dst_nb1 / sizeof(float));

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    // Pre weights: sigmoid + eps
    for (int i = 0; i < n_hc; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    // Post weights: 2 * sigmoid
    for (int i = 0; i < n_hc; ++i) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    // Combination weights: softmax per row + Sinkhorn normalization
    float c[HC_MAX * HC_MAX];

    // Initial softmax over src_hc for each dst_hc
    for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }

    // Normalize over dst_hc for each src_hc
    for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            sum += c[src_hc + dst_hc * n_hc];
        }
        const float inv_denom = 1.0f / (sum + eps);
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            c[src_hc + dst_hc * n_hc] *= inv_denom;
        }
    }

    // Sinkhorn iterations
    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        // Normalize columns (over src_hc for each dst_hc)
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }

        // Normalize rows (over dst_hc for each src_hc)
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
    }

    // Write comb matrix to output
    for (int i = 0; i < n_hc * n_hc; ++i) {
        out[2 * n_hc + i] = c[i];
    }
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    const int32_t n_hc           = ggml_get_op_params_i32(dst, 0);
    const int32_t sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    float eps;
    memcpy(&eps, (int32_t *) dst->op_params + 2, sizeof(float));

    const int64_t n_rows  = ggml_nrows(mixes);
    const int64_t mix_hc  = mixes->ne[0];

    cudaStream_t stream = ctx.stream();

    const int block_size = 256;
    const int num_blocks = (n_rows + block_size - 1) / block_size;

    dsv4_hc_split_sinkhorn_kernel<<<num_blocks, block_size, 0, stream>>>(
        (const float *) mixes->data,
        (const float *) scale->data,
        (const float *) base->data,
        (float *) dst->data,
        n_hc, sinkhorn_iters, eps,
        n_rows, mix_hc,
        mixes->nb[1], dst->nb[1]);
}

// ============================================================================
// DSV4 HC Weighted Sum - y[d,t] = sum_h x[d,h,t] * w[h,t]
// ============================================================================

static __global__ void dsv4_hc_weighted_sum_kernel(
        const char * x, const char * weights, char * dst,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens,
        uint64_t nb_x0, uint64_t nb_x1, uint64_t nb_x2,
        uint64_t nb_w0, uint64_t nb_w1,
        uint64_t nb0,  uint64_t nb1) {
    const int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_elem = n_embd * n_tokens;
    if (gid >= n_elem) return;

    const int64_t d = gid % n_embd;
    const int64_t t = gid / n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < n_hc; ++h) {
        const float xv = *(const float *)(x + d * nb_x0 + h * nb_x1 + t * nb_x2);
        const float wv = *(const float *)(weights + h * nb_w0 + t * nb_w1);
        acc += xv * wv;
    }

    *(float *)(dst + d * nb0 + t * nb1) = acc;
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = x->ne[1];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_elem   = n_embd * n_tokens;

    cudaStream_t stream = ctx.stream();

    const int block_size = 256;
    const int num_blocks = (n_elem + block_size - 1) / block_size;

    dsv4_hc_weighted_sum_kernel<<<num_blocks, block_size, 0, stream>>>(
        (const char *) x->data,
        (const char *) weights->data,
        (char *) dst->data,
        n_embd, n_hc, n_tokens,
        x->nb[0], x->nb[1], x->nb[2],
        weights->nb[0], weights->nb[1],
        dst->nb[0], dst->nb[1]);
}

// ============================================================================
// DSV4 HC Expand - expand hidden state into per-hyper-connection segments
// dst[d, dst_hc, t] = block_out[d, t] * post[dst_hc, t]
//                      + sum_{src_hc} comb[dst_hc, src_hc, t] * residual[d, src_hc, t]
// ============================================================================

static __global__ void dsv4_hc_expand_kernel(
        const char * block_out, const char * residual,
        const char * post, const char * comb,
        char * dst,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens,
        uint64_t nb_block0, uint64_t nb_block1,
        uint64_t nb_res0, uint64_t nb_res1, uint64_t nb_res2,
        uint64_t nb_post0, uint64_t nb_post1,
        uint64_t nb_comb0, uint64_t nb_comb1, uint64_t nb_comb2,
        uint64_t nb0, uint64_t nb1, uint64_t nb2) {
    const int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_elem = n_embd * n_hc * n_tokens;
    if (gid >= n_elem) return;

    const int64_t d      = gid % n_embd;
    const int64_t tmp    = gid / n_embd;
    const int64_t dst_hc = tmp % n_hc;
    const int64_t t      = tmp / n_hc;

    const float block_v = *(const float *)(block_out + d * nb_block0 + t * nb_block1);
    const float post_v  = *(const float *)(post + dst_hc * nb_post0 + t * nb_post1);

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
        const float comb_v = *(const float *)(comb + dst_hc * nb_comb0 + src_hc * nb_comb1 + t * nb_comb2);
        const float res_v  = *(const float *)(residual + d * nb_res0 + src_hc * nb_res1 + t * nb_res2);
        acc += comb_v * res_v;
    }

    *(float *)(dst + d * nb0 + dst_hc * nb1 + t * nb2) = acc;
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = dst->ne[1];
    const int64_t n_tokens = dst->ne[2];
    const int64_t n_elem   = n_embd * n_hc * n_tokens;

    cudaStream_t stream = ctx.stream();

    const int block_size = 256;
    const int num_blocks = (n_elem + block_size - 1) / block_size;

    dsv4_hc_expand_kernel<<<num_blocks, block_size, 0, stream>>>(
        (const char *) block_out->data,
        (const char *) residual->data,
        (const char *) post->data,
        (const char *) comb->data,
        (char *) dst->data,
        n_embd, n_hc, n_tokens,
        block_out->nb[0], block_out->nb[1],
        residual->nb[0], residual->nb[1], residual->nb[2],
        post->nb[0], post->nb[1],
        comb->nb[0], comb->nb[1], comb->nb[2],
        dst->nb[0], dst->nb[1], dst->nb[2]);
}
