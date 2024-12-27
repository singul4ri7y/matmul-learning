#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "kernel.h"

#define alpha(i, j)    A[i * ldA + j]
#define beta(i, j)     B[i * ldB + j]
#define gamma(i, j)    C[i * ldC + j]

#define min(x, y)    ((x) < (y) ? (x) : (y))

int MC = 4032, KC = 328, NC = 480;

/* Register micro-tiles */
int MR = 12;
int NR = 16;

__attribute__((flatten)) void loop_one(int n, int k, float *packed_A, float *packed_B, float *C, int ldC) {
    // Slice block of B in columns which is in L2 cache in terms of register blocking size
    for(int j = 0; j < n; j += NR) {
        gemm_12x16kernel_packed(k, packed_A, &packed_B[k * j], &gamma(0, j), ldC);
    }
}

__attribute__((flatten)) void loop_two(int m, int n, int k, float *packed_A, float *packed_B, float *C, int ldC) {
    // Slice panels of A in rows which is in L3 cache in terms of register blocking size
#pragma omp parallel for
    for(int i = 0; i < m; i += MR) {
        loop_one(n, k, &packed_A[i * k], packed_B, &gamma(i, 0), ldC);
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
    // Slice in terms of `NC`, so that blocks of B stay in L2 cache.
    for(int j = 0; j < n; j += NC) {
        int jb = min(NC, n - j);
        pack_block_B_KCxNC(k, jb, &beta(0, j), ldB, Btilde);
        _mm_prefetch((char *) Btilde, _MM_HINT_T1);
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
    // Slice in terms of `KC`.
    for(int p = 0; p < k; p += KC) {
        int pb = min(KC, k - p);
        // Pack whole panel of A and keep in L3 cache
        pack_panel_A_MCxKC(m, pb, &alpha(0, p), ldA, Atilde);
        _mm_prefetch((char *) Atilde, _MM_HINT_T2);
        loop_three(m, n, pb, Atilde, &beta(p, 0), ldB, C, ldC, Btilde);
    }
}

__attribute__((flatten)) void mygemm(int m, int n, int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    float *Atilde = (float *) _mm_malloc(MC * KC * sizeof(float), 64);
    float *Btilde = (float *) _mm_malloc(KC * NC * sizeof(float), 64);

    memset(C, 0, m * n * sizeof(float));
    
    // Slice in terms of rows so that MCxKC panels of A fits in L3 cache.
    for(int i = 0; i < m; i += MC) {
        int ib = min(MC, m - i);
        loop_four(ib, n, k, &alpha(i, 0), ldA, B, ldB, &gamma(i, 0), ldC, Atilde, Btilde);
    }

    _mm_free(Atilde);
    _mm_free(Btilde);
}

void mygemm2(int m, int n, int k, float *A, int ldA, float *B, int ldB, float *C, int ldC) {
    memset(C, 0, m * n * sizeof(float));

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int p = 0; p < k; p++) {
                C[i * ldC + j] += A[i * ldA + p] * B[p * ldB + j];
            }
        }
    }
}

