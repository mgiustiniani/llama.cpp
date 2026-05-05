#include "dsv4-hc-split-sinkhorn.cuh"
#include "ggml-impl.h"

// DS4V HC Split Sinkhorn CUDA kernel
// n_hc <= 16, so the matrix fits in shared memory
__device__ void dsv4_hc_split_sinkhorn_row(
    const float *mix,
    const float *scale_data,
    const float *base_data,
    float *out,
    int n_hc,
    int sinkhorn_iters,
    float eps) {

    float pre_scale = scale_data[0];
    float post_scale = scale_data[1];
    float comb_scale = scale_data[2];

    // Apply sigmoid to pre-scaled inputs
    for (int i = 0; i < n_hc; ++i) {
        float z = mix[i] * pre_scale + base_data[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    // Apply sigmoid to post-scaled inputs
    for (int i = 0; i < n_hc; ++i) {
        int off = n_hc + i;
        float z = mix[off] * post_scale + base_data[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[256]; // n_hc <= 16, so max 16*16=256

    // Load combination matrix and apply softmax over src_hc for each dst_hc
    for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            int idx = src_hc + dst_hc * n_hc;
            int off = 2 * n_hc + idx;
            float v = mix[off] * comb_scale + base_data[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            int idx = src_hc + dst_hc * n_hc;
            float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            int idx = src_hc + dst_hc * n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }

    // Normalize over dst_hc for each src_hc
    for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            sum += c[src_hc + dst_hc * n_hc];
        }

        float inv_denom = 1.0f / (sum + eps);
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            c[src_hc + dst_hc * n_hc] *= inv_denom;
        }
    }

    // Sinkhorn-Knopp iterations
    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        // Normalize columns (over dst_hc)
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }

            float inv_denom = 1.0f / (sum + eps);
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }

        // Normalize rows (over src_hc)
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }

            float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
    }

    // Write combination matrix to output
    for (int i = 0; i < n_hc * n_hc; ++i) {
        out[2 * n_hc + i] = c[i];
    }
}

__global__ void dsv4_hc_split_sinkhorn_kernel(
    const float *mixes,
    const float *scale,
    const float *base,
    float *dst,
    int n_hc,
    int sinkhorn_iters,
    float eps,
    int64_t mix_hc,
    int64_t n_rows) {

    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;

    const float *mix = mixes + r * mix_hc;
    float *out = dst + r * mix_hc;

    dsv4_hc_split_sinkhorn_row(mix, scale, base, out, n_hc, sinkhorn_iters, eps);
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor *mixes = dst->src[0];
    const ggml_tensor *scale = dst->src[1];
    const ggml_tensor *base = dst->src[2];

    int64_t ne00 = mixes->ne[0];
    int64_t ne01 = mixes->ne[1];
    int64_t ne02 = mixes->ne[2];
    int64_t ne03 = mixes->ne[3];
    size_t nb00 = mixes->nb[0];
    size_t nb01 = mixes->nb[1];
    size_t nb02 = mixes->nb[2];
    size_t nb03 = mixes->nb[3];
    size_t mix_hc = ne01 * ne02 * ne03;

    int n_hc = ggml_get_op_params_i32(dst, 0);
    int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    float eps = ggml_get_op_params_f32(dst, 2);

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int threads = 256;
    int64_t n_rows = ggml_nrows(mixes);
    int blocks = (n_rows + threads - 1) / threads;

    dsv4_hc_split_sinkhorn_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *)mixes->data,
        (const float *)scale->data,
        (const float *)base->data,
        (float *)dst->data,
        n_hc,
        sinkhorn_iters,
        eps,
        mix_hc,
        n_rows
    );
}
