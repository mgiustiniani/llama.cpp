#include "dsv4-rope-tail.cuh"
#include "ggml-impl.h"
#include "ggml-cuda/common.cuh"

struct rope_corr_dims {
    float v[2];
};

// YaRN device helpers (inlined from ggml.c + rope.cu for device-side use)
__device__ static float rope_yarn_ramp(float low, float high, int64_t i0) {
    float y = (i0 / 2.0f - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ static float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

__device__ static void rope_yarn_corr_dims_dev(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    float start = floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf((float)(n_dims - 1), end);
}

// Half to float / float to half conversion helpers for HIP/CUDA
template<typename T>
__device__ inline float to_float(T v) { return (float)v; }
template<>
__device__ inline float to_float<half>(half v) { return __half2float(v); }

template<typename D>
__device__ inline D from_float(float v) { return (D)v; }
template<>
__device__ inline half from_float<half>(float v) { return __float2half(v); }

// DS4V RoPE tail kernel - only applies RoPE to the tail portion
template<typename T, typename D>
static __global__ void dsv4_rope_tail_kernel(
    const T *src,
    const int32_t *pos,
    const float *freq_factors,
    D *dst,
    int64_t n_nope,
    int64_t n_dims,
    int64_t head_dim,
    int64_t ne01,
    int64_t ne02,
    int64_t ne03,
    size_t nb0,
    size_t nb1,
    size_t nb2,
    size_t nb3,
    size_t d_nb0,
    size_t d_nb1,
    size_t d_nb2,
    size_t d_nb3,
    float freq_scale,
    float theta_scale,
    int n_ctx_orig,
    float beta_fast,
    float beta_slow,
    float ext_factor,
    float attn_factor,
    bool inverse) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = head_dim * ne01 * ne02 * ne03;
    if (idx >= total) return;

    int64_t i3 = idx / (head_dim * ne01 * ne02);
    int64_t tmp = idx - i3 * head_dim * ne01 * ne02;
    int64_t i2 = tmp / (head_dim * ne01);
    tmp -= i2 * head_dim * ne01;
    int64_t i1 = tmp / head_dim;
    int64_t i0 = tmp - i1 * head_dim;

    const T *src_ptr = (const T *)((const char *)src + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
    D *dst_ptr = (D *)((char *)dst + i0 * d_nb0 + i1 * d_nb1 + i2 * d_nb2 + i3 * d_nb3);

    // Copy the nope prefix unchanged
    if (n_nope > 0 && i0 < n_nope) {
        *dst_ptr = *src_ptr;
        return;
    }

    // Apply RoPE to the tail
    int64_t pos_val = pos[i2];
    int64_t tail_idx = i0 - n_nope;
    int64_t half_dims = n_dims / 2;
    int64_t freq_idx = tail_idx / 2;

    float theta_base = pos_val * powf(theta_scale, freq_idx);

    float freq = 1.0f;
    if (freq_factors != NULL) {
        freq = freq_factors[freq_idx];
    }

    float theta = theta_base * freq * freq_scale;

    // YaRN correction
    rope_corr_dims corr_dims;
    rope_yarn_corr_dims_dev(n_dims, n_ctx_orig, 1.0f, beta_fast, beta_slow, corr_dims.v);
    float mscale = 1.0f;
    float theta_extrap = theta / freq_scale;
    float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], tail_idx) * ext_factor;
    theta = theta * (1 - ramp_mix) + theta_extrap * ramp_mix;
    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    float attn_scale = attn_factor;
    mscale *= attn_scale;

    float cos_theta = cosf(theta) * mscale;
    float sin_theta = sinf(theta) * mscale;
    if (inverse) {
        sin_theta = -sin_theta;
    }

    // Apply 2D rotation (pairwise)
    int64_t pair_idx = tail_idx / 2;
    int offset = (tail_idx % 2) * 2;
    int i0_even = n_nope + pair_idx * 2 + offset;
    int i0_odd = n_nope + pair_idx * 2 + offset + 1;

    const T *src_ptr_even = (const T *)((const char *)src + i0_even * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
    const T *src_ptr_odd = (const T *)((const char *)src + i0_odd * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

    float x0, x1;
    x0 = to_float(*src_ptr_even);
    x1 = to_float(*src_ptr_odd);

    float r0 = x0 * cos_theta - x1 * sin_theta;
    float r1 = x0 * sin_theta + x1 * cos_theta;

    D r0_d, r1_d;
    r0_d = from_float<D>(r0);
    r1_d = from_float<D>(r1);

    ((D *)((char *)dst + i0_even * d_nb0 + i1 * d_nb1 + i2 * d_nb2 + i3 * d_nb3))[0] = r0_d;
    ((D *)((char *)dst + i0_odd * d_nb0 + i1 * d_nb1 + i2 * d_nb2 + i3 * d_nb3))[0] = r1_d;
}

// NeoX variant - interleaved pairing
template<typename T, typename D>
static __global__ void dsv4_rope_tail_neox_kernel(
    const T *src,
    const int32_t *pos,
    const float *freq_factors,
    D *dst,
    int64_t n_nope,
    int64_t n_dims,
    int64_t head_dim,
    int64_t ne01,
    int64_t ne02,
    int64_t ne03,
    size_t nb0,
    size_t nb1,
    size_t nb2,
    size_t nb3,
    size_t d_nb0,
    size_t d_nb1,
    size_t d_nb2,
    size_t d_nb3,
    float freq_scale,
    float theta_scale,
    int n_ctx_orig,
    float beta_fast,
    float beta_slow,
    float ext_factor,
    float attn_factor,
    bool inverse) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = head_dim * ne01 * ne02 * ne03;
    if (idx >= total) return;

    int64_t i3 = idx / (head_dim * ne01 * ne02);
    int64_t tmp = idx - i3 * head_dim * ne01 * ne02;
    int64_t i2 = tmp / (head_dim * ne01);
    tmp -= i2 * head_dim * ne01;
    int64_t i1 = tmp / head_dim;
    int64_t i0 = tmp - i1 * head_dim;

    // Copy the nope prefix unchanged
    if (n_nope > 0 && i0 < n_nope) {
        const T *src_ptr = (const T *)((const char *)src + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
        D *dst_ptr = (D *)((char *)dst + i0 * d_nb0 + i1 * d_nb1 + i2 * d_nb2 + i3 * d_nb3);
        *dst_ptr = *src_ptr;
        return;
    }

    // Apply RoPE to the tail (NeOX: pair elements with stride n_dims/2)
    int64_t pos_val = pos[i2];
    int64_t tail_idx = i0 - n_nope;
    int64_t half_dims = n_dims / 2;
    int64_t freq_idx = tail_idx % half_dims;

    float theta_base = pos_val * powf(theta_scale, freq_idx);

    float freq = 1.0f;
    if (freq_factors != NULL) {
        freq = freq_factors[freq_idx];
    }

    float theta = theta_base * freq * freq_scale;

    rope_corr_dims corr_dims;
    rope_yarn_corr_dims_dev(n_dims, n_ctx_orig, 1.0f, beta_fast, beta_slow, corr_dims.v);
    float mscale = 1.0f;
    float theta_extrap = theta / freq_scale;
    float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], tail_idx) * ext_factor;
    theta = theta * (1 - ramp_mix) + theta_extrap * ramp_mix;
    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    float attn_scale = attn_factor;
    mscale *= attn_scale;

    float cos_theta = cosf(theta) * mscale;
    float sin_theta = sinf(theta) * mscale;
    if (inverse) {
        sin_theta = -sin_theta;
    }

    // NeOX: pair element i with element (i + half_dims)
    int64_t pair_idx = tail_idx + half_dims;
    if (pair_idx >= n_dims) pair_idx -= n_dims;

    int64_t i0_a = n_nope + tail_idx;
    int64_t i0_b = n_nope + pair_idx;

    const T *src_ptr_a = (const T *)((const char *)src + i0_a * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
    const T *src_ptr_b = (const T *)((const char *)src + i0_b * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

    float x0, x1;
    x0 = to_float(*src_ptr_a);
    x1 = to_float(*src_ptr_b);

    float r0 = x0 * cos_theta - x1 * sin_theta;
    float r1 = x0 * sin_theta + x1 * cos_theta;

    D r0_d, r1_d;
    r0_d = from_float<D>(r0);
    r1_d = from_float<D>(r1);

    ((D *)((char *)dst + i0_a * d_nb0 + i1 * d_nb1 + i2 * d_nb2 + i3 * d_nb3))[0] = r0_d;
    ((D *)((char *)dst + i0_b * d_nb0 + i1 * d_nb1 + i2 * d_nb2 + i3 * d_nb3))[0] = r1_d;
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    const ggml_tensor *src2 = dst->src[2];

    int n_dims = ((int32_t *)dst->op_params)[0];
    int mode = ((int32_t *)dst->op_params)[1];
    int n_ctx_orig = ((int32_t *)dst->op_params)[2];
    bool inverse = ((int32_t *)dst->op_params)[3] != 0;

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *)dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (int32_t *)dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (int32_t *)dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (int32_t *)dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (int32_t *)dst->op_params + 9, sizeof(float));

    int64_t head_dim = src0->ne[0];
    int64_t n_nope = head_dim - n_dims;

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    int64_t ne01 = src0->ne[1];
    int64_t ne02 = src0->ne[2];
    int64_t ne03 = src0->ne[3];

    int threads = 256;
    int64_t total = head_dim * ne01 * ne02 * ne03;
    int blocks = (total + threads - 1) / threads;

    const float *freq_factors = nullptr;
    if (src2 != NULL) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (const float *)src2->data;
    }

    if (src0->type == GGML_TYPE_F16) {
        if (is_neox) {
            dsv4_rope_tail_neox_kernel<half, half><<<blocks, threads, 0, ctx.stream()>>>(
                (const half *)src0->data,
                (const int32_t *)src1->data,
                freq_factors,
                (half *)dst->data,
                n_nope, n_dims, head_dim,
                ne01, ne02, ne03,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                freq_scale, theta_scale, n_ctx_orig,
                beta_fast, beta_slow, ext_factor, attn_factor, inverse
            );
        } else {
            dsv4_rope_tail_kernel<half, half><<<blocks, threads, 0, ctx.stream()>>>(
                (const half *)src0->data,
                (const int32_t *)src1->data,
                freq_factors,
                (half *)dst->data,
                n_nope, n_dims, head_dim,
                ne01, ne02, ne03,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                freq_scale, theta_scale, n_ctx_orig,
                beta_fast, beta_slow, ext_factor, attn_factor, inverse
            );
        }
    } else { // F32
        if (is_neox) {
            dsv4_rope_tail_neox_kernel<float, float><<<blocks, threads, 0, ctx.stream()>>>(
                (const float *)src0->data,
                (const int32_t *)src1->data,
                freq_factors,
                (float *)dst->data,
                n_nope, n_dims, head_dim,
                ne01, ne02, ne03,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                freq_scale, theta_scale, n_ctx_orig,
                beta_fast, beta_slow, ext_factor, attn_factor, inverse
            );
        } else {
            dsv4_rope_tail_kernel<float, float><<<blocks, threads, 0, ctx.stream()>>>(
                (const float *)src0->data,
                (const int32_t *)src1->data,
                freq_factors,
                (float *)dst->data,
                n_nope, n_dims, head_dim,
                ne01, ne02, ne03,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                freq_scale, theta_scale, n_ctx_orig,
                beta_fast, beta_slow, ext_factor, attn_factor, inverse
            );
        }
    }
}
