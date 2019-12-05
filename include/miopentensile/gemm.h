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
    void* data;
} miopen_tensile_matrix;

miopen_tensile_status miopen_tensile_gemm(hipStream_t stream, miopen_tensile_matrix* a, miopen_tensile_matrix* b, miopen_tensile_matrix* c);

#ifdef __cplusplus
}
#endif

#endif
