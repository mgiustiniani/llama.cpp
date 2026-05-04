#include "dsv4-fp8-kv-quantize.cuh"
#include "ggml-impl.h"

// FP8 E4M3FN quantization lookup table helper
__device__ float dsv4_fp8_e4m3fn_quant(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);

    int best = 0;
    float best_diff = ax;

    for (int i = 1; i < 127; ++i) {
        int exp = (i >> 3) & 0x0f;
        int mant = i & 0x07;
        float val = exp == 0
            ? __ldg((float *)&mant) / 512.0f  // mant / 2^9
            : __ldg((float *)&mant) / 8.0f + 1.0f;
        val = ldexpf(val, exp - 7);

        float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }

    int exp = (best >> 3) & 0x0f;
    int mant = best & 0x07;
    float val = exp == 0
        ? __ldg((float *)&mant) / 512.0f
        : (__ldg((float *)&mant) / 8.0f + 1.0f);
    val = ldexpf(val, exp - 7);

    return sign * val;
}

__global__ void dsv4_fp8_kv_quantize_kernel(
    const float *src,
    float *dst,
    int64_t n_rot,
    int64_t head_dim,
    int64_t n_nope,
    int64_t n_rows) {

    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    for (int64_t i0 = 0; i0 < head_dim; ++i0) {
        if (i0 < n_rot) {
            // RoPE tail: pass through unchanged
            dst[i0] = src[i0];
        } else {
            // Nope part: FP8 E4M3FN quantization
            dst[i0] = dsv4_fp8_e4m3fn_quant(src[i0]);
        }
    }
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor *src0 = dst->src[0];

    GGML_TENSOR_BINARY_OP_LOCALS

    int64_t n_rot = ggml_get_op_params_i32(dst, 0);
    int64_t head_dim = src0->ne[0];
    int64_t n_nope = head_dim - n_rot;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int threads = 256;
    int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    int blocks = (n_rows + threads - 1) / threads;

    dsv4_fp8_kv_quantize_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *)src0->data,
        (float *)dst->data,
        n_rot,
        head_dim,
        n_nope,
        n_rows
    );
}
