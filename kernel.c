#include <immintrin.h>

#define alpha(i, j)    A[i * ldA + j]
#define beta(i, j)     B[i * ldB + j]
#define gamma(i, j)    C[i * ldC + j]

extern int NR, MR;

/* ================ UNPACKED KERNELS ================ */

void gemm_4x24kernel(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    register __m256 c_0 = _mm256_loadu_ps(&gamma(0, 0));
    register __m256 c_a = _mm256_loadu_ps(&gamma(0, 8));
    register __m256 c_A = _mm256_loadu_ps(&gamma(0, 16));

    register __m256 c_1 = _mm256_loadu_ps(&gamma(1, 0));
    register __m256 c_b = _mm256_loadu_ps(&gamma(1, 8));
    register __m256 c_B = _mm256_loadu_ps(&gamma(1, 16));

    register __m256 c_2 = _mm256_loadu_ps(&gamma(2, 0));
    register __m256 c_c = _mm256_loadu_ps(&gamma(2, 8));
    register __m256 c_C = _mm256_loadu_ps(&gamma(2, 16));

    register __m256 c_3 = _mm256_loadu_ps(&gamma(3, 0));
    register __m256 c_d = _mm256_loadu_ps(&gamma(3, 8));
    register __m256 c_D = _mm256_loadu_ps(&gamma(3, 16));

    for(int p = 0; p < k; p++) {
        register __m256 b = _mm256_loadu_ps(&beta(p, 0));
        register __m256 bl = _mm256_loadu_ps(&beta(p, 8));
        register __m256 bL = _mm256_loadu_ps(&beta(p, 16));

        register __m256 a = _mm256_set1_ps(alpha(0, p));
        c_0 = _mm256_fmadd_ps(a, b, c_0);
        c_a = _mm256_fmadd_ps(a, bl, c_a);
        c_A = _mm256_fmadd_ps(a, bL, c_A);

        a = _mm256_set1_ps(alpha(1, p));
        c_1 = _mm256_fmadd_ps(a, b, c_1);
        c_b = _mm256_fmadd_ps(a, bl, c_b);
        c_B = _mm256_fmadd_ps(a, bL, c_B);

        a = _mm256_set1_ps(alpha(2, p));
        c_2 = _mm256_fmadd_ps(a, b, c_2);
        c_c = _mm256_fmadd_ps(a, bl, c_c);
        c_C = _mm256_fmadd_ps(a, bL, c_C);

        a = _mm256_set1_ps(alpha(3, p));
        c_3 = _mm256_fmadd_ps(a, b, c_3);
        c_d = _mm256_fmadd_ps(a, bl, c_d);
        c_D = _mm256_fmadd_ps(a, bL, c_D);
    }

    /* Store the results. */
    _mm256_storeu_ps(&gamma(0, 0), c_0);
    _mm256_storeu_ps(&gamma(0, 8), c_a);
    _mm256_storeu_ps(&gamma(0, 16), c_A);

    _mm256_storeu_ps(&gamma(1, 0), c_1);
    _mm256_storeu_ps(&gamma(1, 8), c_b);
    _mm256_storeu_ps(&gamma(1, 16), c_B);

    _mm256_storeu_ps(&gamma(2, 0), c_2);
    _mm256_storeu_ps(&gamma(2, 8), c_c);
    _mm256_storeu_ps(&gamma(2, 16), c_C);

    _mm256_storeu_ps(&gamma(3, 0), c_3);
    _mm256_storeu_ps(&gamma(3, 8), c_d);
    _mm256_storeu_ps(&gamma(3, 16), c_D);
}

void gemm_14x8kernel(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    register __m256 c_0 = _mm256_loadu_ps(&gamma(0, 0));
    register __m256 c_1 = _mm256_loadu_ps(&gamma(1, 0));
    register __m256 c_2 = _mm256_loadu_ps(&gamma(2, 0));
    register __m256 c_3 = _mm256_loadu_ps(&gamma(3, 0));
    register __m256 c_4 = _mm256_loadu_ps(&gamma(4, 0));
    register __m256 c_5 = _mm256_loadu_ps(&gamma(5, 0));
    register __m256 c_6 = _mm256_loadu_ps(&gamma(6, 0));
    register __m256 c_7 = _mm256_loadu_ps(&gamma(7, 0));
    register __m256 c_8 = _mm256_loadu_ps(&gamma(8, 0));
    register __m256 c_9 = _mm256_loadu_ps(&gamma(9, 0));
    register __m256 c_10 = _mm256_loadu_ps(&gamma(10, 0));
    register __m256 c_11 = _mm256_loadu_ps(&gamma(11, 0));
    register __m256 c_12 = _mm256_loadu_ps(&gamma(12, 0));
    register __m256 c_13 = _mm256_loadu_ps(&gamma(13, 0));

    for(int p = 0; p < k; p++) {
        register __m256 b = _mm256_loadu_ps(&beta(p, 0));
        register __m256 a = _mm256_set1_ps(alpha(0, p));

        c_0 = _mm256_fmadd_ps(a, b, c_0);

        a = _mm256_set1_ps(alpha(1, p));
        c_1 = _mm256_fmadd_ps(a, b, c_1);

        a = _mm256_set1_ps(alpha(2, p));
        c_2 = _mm256_fmadd_ps(a, b, c_2);

        a = _mm256_set1_ps(alpha(3, p));
        c_3 = _mm256_fmadd_ps(a, b, c_3);

        a = _mm256_set1_ps(alpha(4, p));
        c_4 = _mm256_fmadd_ps(a, b, c_4);

        a = _mm256_set1_ps(alpha(5, p));
        c_5 = _mm256_fmadd_ps(a, b, c_5);

        a = _mm256_set1_ps(alpha(6, p));
        c_6 = _mm256_fmadd_ps(a, b, c_6);

        a = _mm256_set1_ps(alpha(7, p));
        c_7 = _mm256_fmadd_ps(a, b, c_7);

        a = _mm256_set1_ps(alpha(8, p));
        c_8 = _mm256_fmadd_ps(a, b, c_8);

        a = _mm256_set1_ps(alpha(9, p));
        c_9 = _mm256_fmadd_ps(a, b, c_9);

        a = _mm256_set1_ps(alpha(10, p));
        c_10 = _mm256_fmadd_ps(a, b, c_10);

        a = _mm256_set1_ps(alpha(11, p));
        c_11 = _mm256_fmadd_ps(a, b, c_11);

        a = _mm256_set1_ps(alpha(12, p));
        c_12 = _mm256_fmadd_ps(a, b, c_12);

        a = _mm256_set1_ps(alpha(13, p));
        c_13 = _mm256_fmadd_ps(a, b, c_13);
    }

    /* Store the results. */
    _mm256_storeu_ps(&gamma(0, 0), c_0);
    _mm256_storeu_ps(&gamma(1, 0), c_1);
    _mm256_storeu_ps(&gamma(2, 0), c_2);
    _mm256_storeu_ps(&gamma(3, 0), c_3);
    _mm256_storeu_ps(&gamma(4, 0), c_4);
    _mm256_storeu_ps(&gamma(5, 0), c_5);
    _mm256_storeu_ps(&gamma(6, 0), c_6);
    _mm256_storeu_ps(&gamma(7, 0), c_7);
    _mm256_storeu_ps(&gamma(8, 0), c_8);
    _mm256_storeu_ps(&gamma(9, 0), c_9);
    _mm256_storeu_ps(&gamma(10, 0), c_10);
    _mm256_storeu_ps(&gamma(11, 0), c_11);
    _mm256_storeu_ps(&gamma(12, 0), c_12);
    _mm256_storeu_ps(&gamma(13, 0), c_13);
}

void gemm_8x16kernel(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    register __m256 c_0 = _mm256_loadu_ps(&gamma(0, 0));
    register __m256 c_a = _mm256_loadu_ps(&gamma(0, 8));

    register __m256 c_1 = _mm256_loadu_ps(&gamma(1, 0));
    register __m256 c_b = _mm256_loadu_ps(&gamma(1, 8));

    register __m256 c_2 = _mm256_loadu_ps(&gamma(2, 0));
    register __m256 c_c = _mm256_loadu_ps(&gamma(2, 8));

    register __m256 c_3 = _mm256_loadu_ps(&gamma(3, 0));
    register __m256 c_d = _mm256_loadu_ps(&gamma(3, 8));

    register __m256 c_4 = _mm256_loadu_ps(&gamma(4, 0));
    register __m256 c_e = _mm256_loadu_ps(&gamma(4, 8));

    register __m256 c_5 = _mm256_loadu_ps(&gamma(5, 0));
    register __m256 c_f = _mm256_loadu_ps(&gamma(5, 8));

    register __m256 c_6 = _mm256_loadu_ps(&gamma(6, 0));
    register __m256 c_g = _mm256_loadu_ps(&gamma(6, 8));

    register __m256 c_7 = _mm256_loadu_ps(&gamma(7, 0));
    register __m256 c_h = _mm256_loadu_ps(&gamma(7, 8));

    for(int p = 0; p < k; p++) {
        register __m256 b = _mm256_loadu_ps(&beta(p, 0));
        register __m256 b_l = _mm256_loadu_ps(&beta(p, 8));

        register __m256 a = _mm256_set1_ps(alpha(0, p));
        c_0 = _mm256_fmadd_ps(a, b, c_0);
        c_a = _mm256_fmadd_ps(a, b_l, c_a);

        a = _mm256_set1_ps(alpha(1, p));
        c_1 = _mm256_fmadd_ps(a, b, c_1);
        c_b = _mm256_fmadd_ps(a, b_l, c_b);

        a = _mm256_set1_ps(alpha(2, p));
        c_2 = _mm256_fmadd_ps(a, b, c_2);
        c_c = _mm256_fmadd_ps(a, b_l, c_c);

        a = _mm256_set1_ps(alpha(3, p));
        c_3 = _mm256_fmadd_ps(a, b, c_3);
        c_d = _mm256_fmadd_ps(a, b_l, c_d);

        a = _mm256_set1_ps(alpha(4, p));
        c_4 = _mm256_fmadd_ps(a, b, c_4);
        c_e = _mm256_fmadd_ps(a, b_l, c_e);

        a = _mm256_set1_ps(alpha(5, p));
        c_5 = _mm256_fmadd_ps(a, b, c_5);
        c_f = _mm256_fmadd_ps(a, b_l, c_f);

        a = _mm256_set1_ps(alpha(6, p));
        c_6 = _mm256_fmadd_ps(a, b, c_6);
        c_g = _mm256_fmadd_ps(a, b_l, c_g);

        a = _mm256_set1_ps(alpha(7, p));
        c_7 = _mm256_fmadd_ps(a, b, c_7);
        c_h = _mm256_fmadd_ps(a, b_l, c_h);
    }

    /* Store the results. */
    _mm256_storeu_ps(&gamma(0, 0), c_0);
    _mm256_storeu_ps(&gamma(0, 8), c_a);

    _mm256_storeu_ps(&gamma(1, 0), c_1);
    _mm256_storeu_ps(&gamma(1, 8), c_b);

    _mm256_storeu_ps(&gamma(2, 0), c_2);
    _mm256_storeu_ps(&gamma(2, 8), c_c);

    _mm256_storeu_ps(&gamma(3, 0), c_3);
    _mm256_storeu_ps(&gamma(3, 8), c_d);

    _mm256_storeu_ps(&gamma(4, 0), c_4);
    _mm256_storeu_ps(&gamma(4, 8), c_e);

    _mm256_storeu_ps(&gamma(5, 0), c_5);
    _mm256_storeu_ps(&gamma(5, 8), c_f);

    _mm256_storeu_ps(&gamma(6, 0), c_6);
    _mm256_storeu_ps(&gamma(6, 8), c_g);

    _mm256_storeu_ps(&gamma(7, 0), c_7);
    _mm256_storeu_ps(&gamma(7, 8), c_h);
}

void gemm_12x16kernel(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    register __m256 c_0 = _mm256_loadu_ps(&gamma(0, 0));
    register __m256 c_a = _mm256_loadu_ps(&gamma(0, 8));

    register __m256 c_1 = _mm256_loadu_ps(&gamma(1, 0));
    register __m256 c_b = _mm256_loadu_ps(&gamma(1, 8));

    register __m256 c_2 = _mm256_loadu_ps(&gamma(2, 0));
    register __m256 c_c = _mm256_loadu_ps(&gamma(2, 8));

    register __m256 c_3 = _mm256_loadu_ps(&gamma(3, 0));
    register __m256 c_d = _mm256_loadu_ps(&gamma(3, 8));

    register __m256 c_4 = _mm256_loadu_ps(&gamma(4, 0));
    register __m256 c_e = _mm256_loadu_ps(&gamma(4, 8));

    register __m256 c_5 = _mm256_loadu_ps(&gamma(5, 0));
    register __m256 c_f = _mm256_loadu_ps(&gamma(5, 8));

    register __m256 c_6 = _mm256_loadu_ps(&gamma(6, 0));
    register __m256 c_g = _mm256_loadu_ps(&gamma(6, 8));

    register __m256 c_7 = _mm256_loadu_ps(&gamma(7, 0));
    register __m256 c_h = _mm256_loadu_ps(&gamma(7, 8));

    register __m256 c_8 = _mm256_loadu_ps(&gamma(8, 0));
    register __m256 c_i = _mm256_loadu_ps(&gamma(8, 8));

    register __m256 c_9 = _mm256_loadu_ps(&gamma(9, 0));
    register __m256 c_j = _mm256_loadu_ps(&gamma(9, 8));

    register __m256 c_10 = _mm256_loadu_ps(&gamma(10, 0));
    register __m256 c_k = _mm256_loadu_ps(&gamma(10, 8));

    register __m256 c_11 = _mm256_loadu_ps(&gamma(11, 0));
    register __m256 c_l = _mm256_loadu_ps(&gamma(11, 8));

    for(int p = 0; p < k; p++) {
        register __m256 b = _mm256_loadu_ps(&beta(p, 0));
        register __m256 b_l = _mm256_loadu_ps(&beta(p, 8));

        register __m256 a = _mm256_set1_ps(alpha(0, p));
        c_0 = _mm256_fmadd_ps(a, b, c_0);
        c_a = _mm256_fmadd_ps(a, b_l, c_a);

        a = _mm256_set1_ps(alpha(1, p));
        c_1 = _mm256_fmadd_ps(a, b, c_1);
        c_b = _mm256_fmadd_ps(a, b_l, c_b);

        a = _mm256_set1_ps(alpha(2, p));
        c_2 = _mm256_fmadd_ps(a, b, c_2);
        c_c = _mm256_fmadd_ps(a, b_l, c_c);

        a = _mm256_set1_ps(alpha(3, p));
        c_3 = _mm256_fmadd_ps(a, b, c_3);
        c_d = _mm256_fmadd_ps(a, b_l, c_d);

        a = _mm256_set1_ps(alpha(4, p));
        c_4 = _mm256_fmadd_ps(a, b, c_4);
        c_e = _mm256_fmadd_ps(a, b_l, c_e);

        a = _mm256_set1_ps(alpha(5, p));
        c_5 = _mm256_fmadd_ps(a, b, c_5);
        c_f = _mm256_fmadd_ps(a, b_l, c_f);

        a = _mm256_set1_ps(alpha(6, p));
        c_6 = _mm256_fmadd_ps(a, b, c_6);
        c_g = _mm256_fmadd_ps(a, b_l, c_g);

        a = _mm256_set1_ps(alpha(7, p));
        c_7 = _mm256_fmadd_ps(a, b, c_7);
        c_h = _mm256_fmadd_ps(a, b_l, c_h);

        a = _mm256_set1_ps(alpha(8, p));
        c_8 = _mm256_fmadd_ps(a, b, c_8);
        c_i = _mm256_fmadd_ps(a, b_l, c_i);

        a = _mm256_set1_ps(alpha(9, p));
        c_9 = _mm256_fmadd_ps(a, b, c_9);
        c_j = _mm256_fmadd_ps(a, b_l, c_j);

        a = _mm256_set1_ps(alpha(10, p));
        c_10 = _mm256_fmadd_ps(a, b, c_10);
        c_k = _mm256_fmadd_ps(a, b_l, c_k);

        a = _mm256_set1_ps(alpha(11, p));
        c_11 = _mm256_fmadd_ps(a, b, c_11);
        c_l = _mm256_fmadd_ps(a, b_l, c_l);
    }

    /* Store the results. */
    _mm256_storeu_ps(&gamma(0, 0), c_0);
    _mm256_storeu_ps(&gamma(0, 8), c_a);

    _mm256_storeu_ps(&gamma(1, 0), c_1);
    _mm256_storeu_ps(&gamma(1, 8), c_b);

    _mm256_storeu_ps(&gamma(2, 0), c_2);
    _mm256_storeu_ps(&gamma(2, 8), c_c);

    _mm256_storeu_ps(&gamma(3, 0), c_3);
    _mm256_storeu_ps(&gamma(3, 8), c_d);

    _mm256_storeu_ps(&gamma(4, 0), c_4);
    _mm256_storeu_ps(&gamma(4, 8), c_e);

    _mm256_storeu_ps(&gamma(5, 0), c_5);
    _mm256_storeu_ps(&gamma(5, 8), c_f);

    _mm256_storeu_ps(&gamma(6, 0), c_6);
    _mm256_storeu_ps(&gamma(6, 8), c_g);

    _mm256_storeu_ps(&gamma(7, 0), c_7);
    _mm256_storeu_ps(&gamma(7, 8), c_h);

    _mm256_storeu_ps(&gamma(8, 0), c_8);
    _mm256_storeu_ps(&gamma(8, 8), c_i);

    _mm256_storeu_ps(&gamma(9, 0), c_9);
    _mm256_storeu_ps(&gamma(9, 8), c_j);

    _mm256_storeu_ps(&gamma(10, 0), c_10);
    _mm256_storeu_ps(&gamma(10, 8), c_k);

    _mm256_storeu_ps(&gamma(11, 0), c_11);
    _mm256_storeu_ps(&gamma(11, 8), c_l);
}

/* ================ UNPACKED KERNELS ================ */


/* ================ PACKED KERNELS ================ */

void gemm_4x24kernel_packed(int k, float *packed_A, float *packed_B, float *C, int ldC) {
    register __m256 c_0 = _mm256_load_ps(&gamma(0, 0));
    register __m256 c_a = _mm256_load_ps(&gamma(0, 8));
    register __m256 c_A = _mm256_load_ps(&gamma(0, 16));

    register __m256 c_1 = _mm256_load_ps(&gamma(1, 0));
    register __m256 c_b = _mm256_load_ps(&gamma(1, 8));
    register __m256 c_B = _mm256_load_ps(&gamma(1, 16));

    register __m256 c_2 = _mm256_load_ps(&gamma(2, 0));
    register __m256 c_c = _mm256_load_ps(&gamma(2, 8));
    register __m256 c_C = _mm256_load_ps(&gamma(2, 16));

    register __m256 c_3 = _mm256_load_ps(&gamma(3, 0));
    register __m256 c_d = _mm256_load_ps(&gamma(3, 8));
    register __m256 c_D = _mm256_load_ps(&gamma(3, 16));

    for(int p = 0; p < k; p++) {
        register __m256 b = _mm256_load_ps(packed_B);
        register __m256 bl = _mm256_load_ps(packed_B + 8);
        register __m256 bL = _mm256_load_ps(packed_B + 16);

        register __m256 a = _mm256_set1_ps(packed_A[0]);
        c_0 = _mm256_fmadd_ps(a, b, c_0);
        c_a = _mm256_fmadd_ps(a, bl, c_a);
        c_A = _mm256_fmadd_ps(a, bL, c_A);

        a = _mm256_set1_ps(packed_A[1]);
        c_1 = _mm256_fmadd_ps(a, b, c_1);
        c_b = _mm256_fmadd_ps(a, bl, c_b);
        c_B = _mm256_fmadd_ps(a, bL, c_B);

        a = _mm256_set1_ps(packed_A[2]);
        c_2 = _mm256_fmadd_ps(a, b, c_2);
        c_c = _mm256_fmadd_ps(a, bl, c_c);
        c_C = _mm256_fmadd_ps(a, bL, c_C);

        a = _mm256_set1_ps(packed_A[3]);
        c_3 = _mm256_fmadd_ps(a, b, c_3);
        c_d = _mm256_fmadd_ps(a, bl, c_d);
        c_D = _mm256_fmadd_ps(a, bL, c_D);

        packed_A += 4;
        packed_B += 24;
    }

    /* Store the results. */
    _mm256_store_ps(&gamma(0, 0), c_0);
    _mm256_store_ps(&gamma(0, 8), c_a);
    _mm256_store_ps(&gamma(0, 16), c_A);

    _mm256_store_ps(&gamma(1, 0), c_1);
    _mm256_store_ps(&gamma(1, 8), c_b);
    _mm256_store_ps(&gamma(1, 16), c_B);

    _mm256_store_ps(&gamma(2, 0), c_2);
    _mm256_store_ps(&gamma(2, 8), c_c);
    _mm256_store_ps(&gamma(2, 16), c_C);

    _mm256_store_ps(&gamma(3, 0), c_3);
    _mm256_store_ps(&gamma(3, 8), c_d);
    _mm256_store_ps(&gamma(3, 16), c_D);
}

void gemm_14x8kernel_packed(int k, float *packed_A, float *packed_B, float *C, int ldC) {
    register __m256 c_0 = _mm256_load_ps(&gamma(0, 0));
    register __m256 c_1 = _mm256_load_ps(&gamma(1, 0));
    register __m256 c_2 = _mm256_load_ps(&gamma(2, 0));
    register __m256 c_3 = _mm256_load_ps(&gamma(3, 0));
    register __m256 c_4 = _mm256_load_ps(&gamma(4, 0));
    register __m256 c_5 = _mm256_load_ps(&gamma(5, 0));
    register __m256 c_6 = _mm256_load_ps(&gamma(6, 0));
    register __m256 c_7 = _mm256_load_ps(&gamma(7, 0));
    register __m256 c_8 = _mm256_load_ps(&gamma(8, 0));
    register __m256 c_9 = _mm256_load_ps(&gamma(9, 0));
    register __m256 c_10 = _mm256_load_ps(&gamma(10, 0));
    register __m256 c_11 = _mm256_load_ps(&gamma(11, 0));
    register __m256 c_12 = _mm256_load_ps(&gamma(12, 0));
    register __m256 c_13 = _mm256_load_ps(&gamma(13, 0));

    for(int p = 0; p < k; p++) {
        register __m256 b = _mm256_load_ps(packed_B);
        register __m256 a = _mm256_set1_ps(packed_A[0]);

        c_0 = _mm256_fmadd_ps(a, b, c_0);

        a = _mm256_set1_ps(packed_A[1]);
        c_1 = _mm256_fmadd_ps(a, b, c_1);

        a = _mm256_set1_ps(packed_A[2]);
        c_2 = _mm256_fmadd_ps(a, b, c_2);

        a = _mm256_set1_ps(packed_A[3]);
        c_3 = _mm256_fmadd_ps(a, b, c_3);

        a = _mm256_set1_ps(packed_A[4]);
        c_4 = _mm256_fmadd_ps(a, b, c_4);

        a = _mm256_set1_ps(packed_A[5]);
        c_5 = _mm256_fmadd_ps(a, b, c_5);

        a = _mm256_set1_ps(packed_A[6]);
        c_6 = _mm256_fmadd_ps(a, b, c_6);

        a = _mm256_set1_ps(packed_A[7]);
        c_7 = _mm256_fmadd_ps(a, b, c_7);

        a = _mm256_set1_ps(packed_A[8]);
        c_8 = _mm256_fmadd_ps(a, b, c_8);

        a = _mm256_set1_ps(packed_A[9]);
        c_9 = _mm256_fmadd_ps(a, b, c_9);

        a = _mm256_set1_ps(packed_A[10]);
        c_10 = _mm256_fmadd_ps(a, b, c_10);

        a = _mm256_set1_ps(packed_A[11]);
        c_11 = _mm256_fmadd_ps(a, b, c_11);

        a = _mm256_set1_ps(packed_A[12]);
        c_12 = _mm256_fmadd_ps(a, b, c_12);

        a = _mm256_set1_ps(packed_A[13]);
        c_13 = _mm256_fmadd_ps(a, b, c_13);

        packed_A += 14;
        packed_B += 8;
    }

    /* Store the results. */
    _mm256_store_ps(&gamma(0, 0), c_0);
    _mm256_store_ps(&gamma(1, 0), c_1);
    _mm256_store_ps(&gamma(2, 0), c_2);
    _mm256_store_ps(&gamma(3, 0), c_3);
    _mm256_store_ps(&gamma(4, 0), c_4);
    _mm256_store_ps(&gamma(5, 0), c_5);
    _mm256_store_ps(&gamma(6, 0), c_6);
    _mm256_store_ps(&gamma(7, 0), c_7);
    _mm256_store_ps(&gamma(8, 0), c_8);
    _mm256_store_ps(&gamma(9, 0), c_9);
    _mm256_store_ps(&gamma(10, 0), c_10);
    _mm256_store_ps(&gamma(11, 0), c_11);
    _mm256_store_ps(&gamma(12, 0), c_12);
    _mm256_store_ps(&gamma(13, 0), c_13);
}

void gemm_8x16kernel_packed(int k, float *packed_A, float *packed_B, float *C, int ldC) {
    register __m256 c_0 = _mm256_load_ps(&gamma(0, 0));
    register __m256 c_a = _mm256_load_ps(&gamma(0, 8));

    register __m256 c_1 = _mm256_load_ps(&gamma(1, 0));
    register __m256 c_b = _mm256_load_ps(&gamma(1, 8));

    register __m256 c_2 = _mm256_load_ps(&gamma(2, 0));
    register __m256 c_c = _mm256_load_ps(&gamma(2, 8));

    register __m256 c_3 = _mm256_load_ps(&gamma(3, 0));
    register __m256 c_d = _mm256_load_ps(&gamma(3, 8));

    register __m256 c_4 = _mm256_load_ps(&gamma(4, 0));
    register __m256 c_e = _mm256_load_ps(&gamma(4, 8));

    register __m256 c_5 = _mm256_load_ps(&gamma(5, 0));
    register __m256 c_f = _mm256_load_ps(&gamma(5, 8));

    register __m256 c_6 = _mm256_load_ps(&gamma(6, 0));
    register __m256 c_g = _mm256_load_ps(&gamma(6, 8));

    register __m256 c_7 = _mm256_load_ps(&gamma(7, 0));
    register __m256 c_h = _mm256_load_ps(&gamma(7, 8));

    for(int p = 0; p < k; p++) {
        register __m256 b = _mm256_load_ps(packed_B);
        register __m256 b_l = _mm256_load_ps(packed_B + 8);

        register __m256 a = _mm256_set1_ps(packed_A[0]);
        c_0 = _mm256_fmadd_ps(a, b, c_0);
        c_a = _mm256_fmadd_ps(a, b_l, c_a);

        a = _mm256_set1_ps(packed_A[1]);
        c_1 = _mm256_fmadd_ps(a, b, c_1);
        c_b = _mm256_fmadd_ps(a, b_l, c_b);

        a = _mm256_set1_ps(packed_A[2]);
        c_2 = _mm256_fmadd_ps(a, b, c_2);
        c_c = _mm256_fmadd_ps(a, b_l, c_c);

        a = _mm256_set1_ps(packed_A[3]);
        c_3 = _mm256_fmadd_ps(a, b, c_3);
        c_d = _mm256_fmadd_ps(a, b_l, c_d);

        a = _mm256_set1_ps(packed_A[4]);
        c_4 = _mm256_fmadd_ps(a, b, c_4);
        c_e = _mm256_fmadd_ps(a, b_l, c_e);

        a = _mm256_set1_ps(packed_A[5]);
        c_5 = _mm256_fmadd_ps(a, b, c_5);
        c_f = _mm256_fmadd_ps(a, b_l, c_f);

        a = _mm256_set1_ps(packed_A[6]);
        c_6 = _mm256_fmadd_ps(a, b, c_6);
        c_g = _mm256_fmadd_ps(a, b_l, c_g);

        a = _mm256_set1_ps(packed_A[7]);
        c_7 = _mm256_fmadd_ps(a, b, c_7);
        c_h = _mm256_fmadd_ps(a, b_l, c_h);

        packed_A += 8;
        packed_B += 16;
    }

    /* Store the results. */
    _mm256_store_ps(&gamma(0, 0), c_0);
    _mm256_store_ps(&gamma(0, 8), c_a);

    _mm256_store_ps(&gamma(1, 0), c_1);
    _mm256_store_ps(&gamma(1, 8), c_b);

    _mm256_store_ps(&gamma(2, 0), c_2);
    _mm256_store_ps(&gamma(2, 8), c_c);

    _mm256_store_ps(&gamma(3, 0), c_3);
    _mm256_store_ps(&gamma(3, 8), c_d);

    _mm256_store_ps(&gamma(4, 0), c_4);
    _mm256_store_ps(&gamma(4, 8), c_e);

    _mm256_store_ps(&gamma(5, 0), c_5);
    _mm256_store_ps(&gamma(5, 8), c_f);

    _mm256_store_ps(&gamma(6, 0), c_6);
    _mm256_store_ps(&gamma(6, 8), c_g);

    _mm256_store_ps(&gamma(7, 0), c_7);
    _mm256_store_ps(&gamma(7, 8), c_h);
}

void gemm_12x16kernel_packed(int k, float *packed_A, float *packed_B, float *C, int ldC) {
    register __m256 c_0 = _mm256_load_ps(&gamma(0, 0));
    register __m256 c_a = _mm256_load_ps(&gamma(0, 8));

    register __m256 c_1 = _mm256_load_ps(&gamma(1, 0));
    register __m256 c_b = _mm256_load_ps(&gamma(1, 8));

    register __m256 c_2 = _mm256_load_ps(&gamma(2, 0));
    register __m256 c_c = _mm256_load_ps(&gamma(2, 8));

    register __m256 c_3 = _mm256_load_ps(&gamma(3, 0));
    register __m256 c_d = _mm256_load_ps(&gamma(3, 8));

    register __m256 c_4 = _mm256_load_ps(&gamma(4, 0));
    register __m256 c_e = _mm256_load_ps(&gamma(4, 8));

    register __m256 c_5 = _mm256_load_ps(&gamma(5, 0));
    register __m256 c_f = _mm256_load_ps(&gamma(5, 8));

    register __m256 c_6 = _mm256_load_ps(&gamma(6, 0));
    register __m256 c_g = _mm256_load_ps(&gamma(6, 8));

    register __m256 c_7 = _mm256_load_ps(&gamma(7, 0));
    register __m256 c_h = _mm256_load_ps(&gamma(7, 8));

    register __m256 c_8 = _mm256_load_ps(&gamma(8, 0));
    register __m256 c_i = _mm256_load_ps(&gamma(8, 8));

    register __m256 c_9 = _mm256_load_ps(&gamma(9, 0));
    register __m256 c_j = _mm256_load_ps(&gamma(9, 8));

    register __m256 c_10 = _mm256_load_ps(&gamma(10, 0));
    register __m256 c_k = _mm256_load_ps(&gamma(10, 8));

    register __m256 c_11 = _mm256_load_ps(&gamma(11, 0));
    register __m256 c_l = _mm256_load_ps(&gamma(11, 8));

    #pragma GCC unroll 8
    for(int p = 0; p < k; p++) {
        register __m256 b = _mm256_load_ps(packed_B);
        register __m256 b_l = _mm256_load_ps(packed_B + 8);

        register __m256 a = _mm256_set1_ps(packed_A[0]);
        c_0 = _mm256_fmadd_ps(a, b, c_0);
        c_a = _mm256_fmadd_ps(a, b_l, c_a);

        a = _mm256_set1_ps(packed_A[1]);
        c_1 = _mm256_fmadd_ps(a, b, c_1);
        c_b = _mm256_fmadd_ps(a, b_l, c_b);

        a = _mm256_set1_ps(packed_A[2]);
        c_2 = _mm256_fmadd_ps(a, b, c_2);
        c_c = _mm256_fmadd_ps(a, b_l, c_c);

        a = _mm256_set1_ps(packed_A[3]);
        c_3 = _mm256_fmadd_ps(a, b, c_3);
        c_d = _mm256_fmadd_ps(a, b_l, c_d);

        a = _mm256_set1_ps(packed_A[4]);
        c_4 = _mm256_fmadd_ps(a, b, c_4);
        c_e = _mm256_fmadd_ps(a, b_l, c_e);

        a = _mm256_set1_ps(packed_A[5]);
        c_5 = _mm256_fmadd_ps(a, b, c_5);
        c_f = _mm256_fmadd_ps(a, b_l, c_f);

        a = _mm256_set1_ps(packed_A[6]);
        c_6 = _mm256_fmadd_ps(a, b, c_6);
        c_g = _mm256_fmadd_ps(a, b_l, c_g);

        a = _mm256_set1_ps(packed_A[7]);
        c_7 = _mm256_fmadd_ps(a, b, c_7);
        c_h = _mm256_fmadd_ps(a, b_l, c_h);

        a = _mm256_set1_ps(packed_A[8]);
        c_8 = _mm256_fmadd_ps(a, b, c_8);
        c_i = _mm256_fmadd_ps(a, b_l, c_i);

        a = _mm256_set1_ps(packed_A[9]);
        c_9 = _mm256_fmadd_ps(a, b, c_9);
        c_j = _mm256_fmadd_ps(a, b_l, c_j);

        a = _mm256_set1_ps(packed_A[10]);
        c_10 = _mm256_fmadd_ps(a, b, c_10);
        c_k = _mm256_fmadd_ps(a, b_l, c_k);

        a = _mm256_set1_ps(packed_A[11]);
        c_11 = _mm256_fmadd_ps(a, b, c_11);
        c_l = _mm256_fmadd_ps(a, b_l, c_l);

        packed_A += 12;
        packed_B += 16;
    }

    /* Store the results. */
    _mm256_store_ps(&gamma(0, 0), c_0);
    _mm256_store_ps(&gamma(0, 8), c_a);

    _mm256_store_ps(&gamma(1, 0), c_1);
    _mm256_store_ps(&gamma(1, 8), c_b);

    _mm256_store_ps(&gamma(2, 0), c_2);
    _mm256_store_ps(&gamma(2, 8), c_c);

    _mm256_store_ps(&gamma(3, 0), c_3);
    _mm256_store_ps(&gamma(3, 8), c_d);

    _mm256_store_ps(&gamma(4, 0), c_4);
    _mm256_store_ps(&gamma(4, 8), c_e);

    _mm256_store_ps(&gamma(5, 0), c_5);
    _mm256_store_ps(&gamma(5, 8), c_f);

    _mm256_store_ps(&gamma(6, 0), c_6);
    _mm256_store_ps(&gamma(6, 8), c_g);

    _mm256_store_ps(&gamma(7, 0), c_7);
    _mm256_store_ps(&gamma(7, 8), c_h);

    _mm256_store_ps(&gamma(8, 0), c_8);
    _mm256_store_ps(&gamma(8, 8), c_i);

    _mm256_store_ps(&gamma(9, 0), c_9);
    _mm256_store_ps(&gamma(9, 8), c_j);

    _mm256_store_ps(&gamma(10, 0), c_10);
    _mm256_store_ps(&gamma(10, 8), c_k);

    _mm256_store_ps(&gamma(11, 0), c_11);
    _mm256_store_ps(&gamma(11, 8), c_l);
}

/* ================ PACKED KERNELS END ================ */
