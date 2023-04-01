#include "ggml.h"
#include "stdio.h"


int main(){

    ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
    };
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_f32_1d(x, 0, 20);
   //*((float*)x->data) = 10.0f;
    printf("f = %f\n", ggml_get_f32_1d(x, 0));
    return 0;
}