#include "dsv4-hc-expand.cuh"
#include "ggml-impl.h"

__global__ void dsv4_hc_expand_kernel(
    const float *block_out,
    const float *residual,
    const float *post,
    const float *comb,
    float *dst,
    int64_t n_embd,
    int64_t n_hc,
    int64_t n_tokens,
    size_t block_nb0,
    size_t block_nb1,
    size_t res_nb0,
    size_t res_nb1,
    size_t res_nb2,
    size_t post_nb0,
    size_t post_nb1,
    size_t comb_nb0,
    size_t comb_nb1,
    size_t comb_nb2,
    size_t dst_nb0,
    size_t dst_nb1,
    size_t dst_nb2) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = n_embd * n_hc * n_tokens;
    if (idx >= total) return;

    int64_t d = idx % n_embd;
    int64_t tmp = idx / n_embd;
    int64_t dst_hc = tmp % n_hc;
    int64_t t = tmp / n_hc;

    const float *block_ptr = (const float *)((const char *)block_out + d * block_nb0 + t * block_nb1);
    const float *res_ptr = (const float *)((const char *)residual + d * res_nb0 + dst_hc * res_nb1 + t * res_nb2);
    const float *post_ptr = (const float *)((const char *)post + dst_hc * post_nb0 + t * post_nb1);
    const float *comb_ptr = (const float *)((const char *)comb + dst_hc * comb_nb0 + dst_hc * comb_nb1 + t * comb_nb2);
    float *dst_ptr = (float *)((char *)dst + d * dst_nb0 + dst_hc * dst_nb1 + t * dst_nb2);

    float acc = *block_ptr * *post_ptr;

    for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
        const float *c_ptr = (const float *)((const char *)comb + dst_hc * comb_nb0 + src_hc * comb_nb1 + t * comb_nb2);
        const float *r_ptr = (const float *)((const char *)residual + d * res_nb0 + src_hc * res_nb1 + t * res_nb2);
        acc += *c_ptr * *r_ptr;
    }

    *dst_ptr = acc;
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor *block_out = dst->src[0];
    const ggml_tensor *residual = dst->src[1];
    const ggml_tensor *post = dst->src[2];
    const ggml_tensor *comb = dst->src[3];

    GGML_TENSOR_LOCALS(int64_t, ne0, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, dst, nb)

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int64_t n_embd = dst->ne[0];
    int64_t n_hc = dst->ne[1];
    int64_t n_tokens = dst->ne[2];
    int64_t total = n_embd * n_hc * n_tokens;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    dsv4_hc_expand_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *)block_out->data,
        (const float *)residual->data,
        (const float *)post->data,
        (const float *)comb->data,
        (float *)dst->data,
        n_embd,
        n_hc,
        n_tokens,
        block_out->nb[0],
        block_out->nb[1],
        residual->nb[0],
        residual->nb[1],
        residual->nb[2],
        post->nb[0],
        post->nb[1],
        comb->nb[0],
        comb->nb[1],
        comb->nb[2],
        nb00,
        nb01,
        nb02
    );
}
