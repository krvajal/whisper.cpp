#include "ggml.h"
#include "stdio.h"

struct ggml_context {
    size_t mem_size;
    void* mem_buffer;
};

ggml_context * ggml_init(ggml_init_params params){
    
    void* mem_buffer = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&mem_buffer, params.mem_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    }

    ggml_context* ctx = (ggml_context*)malloc(sizeof(struct ggml_context));
    ctx -> mem_size = params.mem_size;
    ctx -> mem_buffer = mem_buffer;

    return ctx;
}

ggml_tensor* ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0)  {
             return ggml_new_tensor(ctx, type, 1, &ne0);
}

ggml_tensor* ggml_new_tensor(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    n_dims,
        const int *ne) {
    return NULL;
}



float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i){
    return 1.0f;
}

void  ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value){

}