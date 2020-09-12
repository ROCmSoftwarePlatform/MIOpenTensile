#ifndef MIOPENTENSILE_GUARD_GEMM_H
#define MIOPENTENSILE_GUARD_GEMM_H

#include <stddef.h>
#include <stdbool.h>

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    miopen_tensile_status_success = 0, /*!< No errors */
    miopen_tensile_status_no_solution = 1, /*!< No solution found for configuration.. */
    miopen_tensile_status_unknown = 2, /*!< Unknown error occurred.. */
} miopen_tensile_status;

typedef enum {
    miopen_tensile_type_float = 0,
    miopen_tensile_type_half = 1,
    miopen_tensile_type_bfloat16 = 2,
    miopen_tensile_type_int8x4 = 3,
    miopen_tensile_type_int32 = 4,
} miopen_tensile_type;

typedef size_t miopen_tensile_2d[2];

typedef struct
{    
    size_t num;
    size_t stride;
} miopen_tensile_batch;

typedef struct
{
    miopen_tensile_2d lens;
    miopen_tensile_2d strides;
    miopen_tensile_batch batch;
    miopen_tensile_type type;
    bool is_mat_transposed;
    void* data;
} miopen_tensile_matrix;

miopen_tensile_status miopen_tensile_gemm_hip(hipStream_t stream, 
                                              miopen_tensile_matrix* a, 
                                              miopen_tensile_matrix* b, 
                                              miopen_tensile_matrix* c, 
                                              double alpha, 
                                              double beta);

#ifdef __cplusplus
}
#endif

#endif
