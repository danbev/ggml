#include "ggml.h"

#include <stdio.h>

int main(int argc, char ** argv) {

  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
  };
  struct ggml_context * ctx = ggml_init(params);

  int embd_dim = 128;
  int n_head = 32;
  int n_tokens = 6;

  struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, embd_dim, n_head, n_tokens);
  ggml_set_name(a, "a");
  ggml_set_param(ctx, a);

  struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_name(pos, "pos");
  ggml_set_param(ctx, pos);

  struct ggml_tensor * rope = ggml_rope(ctx, a, pos, embd_dim, 0);
  ggml_set_name(rope, "rope");

  struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, embd_dim, n_head, n_tokens);
  ggml_set_name(b, "b");
  ggml_set_param(ctx, b);

  struct ggml_tensor * mul = ggml_mul_mat(ctx, rope, b);
  ggml_set_name(mul, "mul_mat");

  struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
  ggml_build_forward_expand(gf, mul);
  ggml_graph_compute_with_ctx(ctx, gf, 1);

  struct ggml_cgraph * gb = ggml_graph_dup(ctx, gf);
  ggml_build_backward_expand(ctx, gf, gb, false, true);

  GGML_ASSERT(mul->src[0]->grad->grad == NULL);

  ggml_free(ctx);
  return 0;
}
