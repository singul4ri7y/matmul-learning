#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define alpha(i, j)    A[i * ldA + j]
#define beta(i, j)     B[i * ldB + j]
#define gamma(i, j)    C[i * ldC + j]

#define min(x, y)    ((x) < (y) ? (x) : (y))

int MC = 2048, KC = 224, NC = 240;

/* Register micro-tiles */
#define MR    4
#define NR    24

void gemm_8x8_kernel(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    register __m256 c_0 = _mm256_loadu_ps(&gamma(0, 0));
    register __m256 c_1 = _mm256_loadu_ps(&gamma(1, 0));
    register __m256 c_2 = _mm256_loadu_ps(&gamma(2, 0));
    register __m256 c_3 = _mm256_loadu_ps(&gamma(3, 0));
    register __m256 c_4 = _mm256_loadu_ps(&gamma(4, 0));
    register __m256 c_5 = _mm256_loadu_ps(&gamma(5, 0));
    register __m256 c_6 = _mm256_loadu_ps(&gamma(6, 0));
    register __m256 c_7 = _mm256_loadu_ps(&gamma(7, 0));

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

    }

    _mm256_storeu_ps(&gamma(0, 0), c_0);
    _mm256_storeu_ps(&gamma(1, 0), c_1);
    _mm256_storeu_ps(&gamma(2, 0), c_2);
    _mm256_storeu_ps(&gamma(3, 0), c_3);
    _mm256_storeu_ps(&gamma(4, 0), c_4);
    _mm256_storeu_ps(&gamma(5, 0), c_5);
    _mm256_storeu_ps(&gamma(6, 0), c_6);
    _mm256_storeu_ps(&gamma(7, 0), c_7);
}

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

        packed_A += MR;
        packed_B += NR;
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

        packed_A += MR;
        packed_B += NR;
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

        packed_A += MR;
        packed_B += NR;
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

/* MyGEMM in row-major order. */
void gemm_ij_8x8_kernel(int m, int n, int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    for(int i = 0; i < m; i += MR) {
        // int ib = min(MB, m - i);
        for(int j = 0; j < n; j += NR) {
            // int jb = min(NB, n - j);
            gemm_8x8_kernel(k, &alpha(i, 0), ldA, &beta(0, j), ldB, &gamma(i, j), ldC);
        }
    }
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

        packed_A += MR;
        packed_B += NR;
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


__attribute__((flatten)) void loop_one(int n, int k, float *packed_A, float *packed_B, float *C, int ldC) {
    for(int j = 0; j < n; j += NR) {
        gemm_4x24kernel_packed(k, packed_A, packed_B, &gamma(0, j), ldC);
    }
}

__attribute__((flatten)) void loop_two(int m, int n, int k, float *packed_A, float *packed_B, float *C, int ldC) {
    // Macro kernel
    for(int i = 0; i < m; i += MR) {
        loop_one(n, k, packed_A, packed_B, &gamma(i, 0), ldC);
    }
}

// TODO: USE AVX2
__attribute__((flatten)) void pack_block_B_KCxNR(int k, int n, float *B, int ldB, float *Btilde) {
    for(int p = 0; p < k; p++) {
        int j = 0;
        for(; j < n; j++) 
            *Btilde++ = beta(p, j);
        for(; j < NR; j++) 
            *Btilde++ = 0.0f;
    }
}

__attribute__((flatten)) void pack_block_B_KCxNC(int k, int n, float *B, int ldB, float *Btilde) {
    for(int j = 0; j < n; j += NR) {
        int jb = min(NR, n - j);
        pack_block_B_KCxNR(k, jb, &beta(0, j), ldB, Btilde);
        Btilde += k * jb;
    }
}

__attribute__((flatten)) void loop_three(int m, int n, int k, float *packed_A, float *B, int ldB, float *C, int ldC, float *Btilde) {
    for(int j = 0; j < n; j += NC) {
        int jb = min(NC, n - j);
        pack_block_B_KCxNC(k, jb, &beta(0, j), ldB, Btilde);
        loop_two(m, jb, k, packed_A, Btilde, &gamma(0, j), ldC);
    }
}

// TODO: USE AVX2
__attribute__((flatten)) void pack_upanel_A_MRxKC(int m, int k, float *A, int ldA, float *Atilde) {
    for(int p = 0; p < k; p++) {
        int i = 0;
        for(; i < m; i++) 
            *Atilde++ = alpha(i, p);

        for(; i < MR; i++) 
            *Atilde++ = 0.0f;
    }
}

__attribute__((flatten)) void pack_panel_A_MCxKC(int m, int k, float *A, int ldA, float *Atilde) {
    for(int i = 0; i < m; i += MR) {
        int ib = min(MR, m - i);
        pack_upanel_A_MRxKC(ib, k, &alpha(i, 0), ldA, Atilde);
        Atilde += ib * k;
    }
}

__attribute__((flatten)) void loop_four(int m, int n, int k, float *A, int ldA, float *B, int ldB, float *C, int ldC, float *Atilde, float *Btilde) {
    for(int p = 0; p < k; p += KC) {
        int pb = min(KC, k - p);
        pack_panel_A_MCxKC(m, pb, &alpha(0, p), ldA, Atilde);
        loop_three(m, n, pb, Atilde, &beta(p, 0), ldB, C, ldC, Btilde);
    }
}

// This is loop-5
__attribute__((flatten)) void mygemm(int m, int n, int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    memset(C, 0, m * n * sizeof(float));

    float *Atilde = (float *) _mm_malloc(MC * KC * sizeof(float), 64);
    float *Btilde = (float *) _mm_malloc(KC * NC * sizeof(float), 64);

    for(int i = 0; i < m; i += MC) {
        int ib = min(MC, m - i);
        loop_four(ib, n, k, &alpha(i, 0), ldA, B, ldB, &gamma(i, 0), ldC, Atilde, Btilde);
    }

    _mm_free(Atilde);
    _mm_free(Btilde);
}

// void mygemm(int m, int n, int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
//     memset(C, 0, m * n * sizeof(float));
// 
//     for(int p = 0; p < k; p += KC) {
//         int pb = min(KC, k - p);
//         for(int j = 0; j < n; j += NC) {
//             int jb = min(NC, n - j);
//             gemm_ij_8x8_kernel(m, jb, pb, &alpha(0, p), ldA, &beta(p, j), ldB, &gamma(0, j), ldC);
//         }
//     }
// }

void mygemm2(int m, int n, int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    memset(C, 0, m * n * sizeof(float));

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int p = 0; p < k; p++) {
                C[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }
}

