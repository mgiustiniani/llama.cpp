#include "dsv4-hc-weighted-sum.cuh"
#include "ggml-impl.h"

__global__ void dsv4_hc_weighted_sum_kernel(
    const float *x,
    const float *weights,
    float *dst,
    int64_t n_embd,
    int64_t n_hc,
    int64_t n_tokens,
    size_t nb0,
    size_t nb1,
    size_t w_nb0,
    size_t w_nb1) {

    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;

    for (int64_t t = blockIdx.y; t < n_tokens; t += gridDim.y) {
        float acc = 0.0f;
        for (int64_t h = 0; h < n_hc; ++h) {
            const float *x_ptr = (const float *)((const char *)x + d * nb0 + h * nb1);
            const float *w_ptr = (const float *)((const char *)weights + h * w_nb0 + t * w_nb1);
            acc += *x_ptr * *w_ptr;
        }

        *(float *)((char *)dst + d * nb0 + t * (size_t)dst->nb[1]) = acc;
    }
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor *x = dst->src[0];
    const ggml_tensor *weights = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[3] == 1);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(weights->ne[3] == 1);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    int threads = std::min((int)n_embd0, 512);
    dim3 blocks(n_embd0, (int)n_embd1);

    dsv4_hc_weighted_sum_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *)x->data,
        (const float *)weights->data,
        (float *)dst->data,
        ne00,
        ne01,
        ne02,
        nb01,
        nb02,
        nb11,
        nb12
    );
}
