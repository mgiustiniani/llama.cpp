// vk_op_dsv4_hc_weighted_sum_push_constants
layout (push_constant) uniform parameter
{
    uint n_embd;
    uint n_hc;
    uint n_tokens;
    uint nb_x0, nb_x1, nb_x2;
    uint nb_w0, nb_w1;
    uint nb0, nb1;
} p;
