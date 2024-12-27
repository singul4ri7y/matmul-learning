#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H

void gemm_4x24kernel(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC);
void gemm_14x8kernel(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC);
void gemm_8x16kernel(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC);
void gemm_12x16kernel(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC);

void gemm_4x24kernel_packed(int k, float *packed_A, float *packed_B, float *C, int ldC);
void gemm_14x8kernel_packed(int k, float *packed_A, float *packed_B, float *C, int ldC);
void gemm_8x16kernel_packed(int k, float *packed_A, float *packed_B, float *C, int ldC);
void gemm_12x16kernel_packed(int k, float *packed_A, float *packed_B, float *C, int ldC);

#endif  // MATMUL_KERNEL_H
