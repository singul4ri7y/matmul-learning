#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "util.h"
#include "kernel.h"

#define alpha(i, j)    A[i * ldA + j]
#define beta(i, j)     B[i * ldB + j]
#define gamma(i, j)    C[i * ldC + j]

typedef void (*gemm_kernel)(int k, float *A, int ldA, float *B, int ldB, float *C, int ldC);

gemm_kernel kernels[] = {
    gemm_4x24kernel,
    gemm_8x16kernel,
    gemm_12x16kernel,
    gemm_14x8kernel
};

int mr_r[] = { 4, 8, 12, 14 };
int nr_r[] = { 24, 16, 16, 8 };

int KERNSIZ = (sizeof(kernels) / sizeof(kernels[0]));

int C_MR, C_NR;
void custom_mygemm_ij(int m, int n, int k, float *A, int ldA, float *B, int ldB, float *C, int ldC, gemm_kernel kernel) {
    memset(C, 0, m * n * sizeof(float));

    for(int i = 0; i < m; i += C_MR) {
        for(int j = 0; j < n; j += C_NR) {
            kernel(k, &alpha(i, 0), ldA, &beta(0, j), ldB, &gamma(i, j), ldC);
        }
    }
}

int main(int argc, char **argv) {
    if(argc < 5) {
        printf("Invalid argument count. <repeat_operations> <start_dim> <end_dim> <dim_step>\n");
        return -1;
    }

    int repeat = atoi(argv[1]);
    int start = atoi(argv[2]);
    int end = atoi(argv[3]);
    int step = atoi(argv[4]);

    int _BUFSIZ = 128;
    char buffer[_BUFSIZ];

    for(int i = 0; i < KERNSIZ; i++) {
        C_MR = mr_r[i];
        C_NR = nr_r[i];

        snprintf(buffer, _BUFSIZ, "data/kernel/mygemm_%dx%dkernel.dat", C_MR, C_NR);

        FILE *mygemm_kernel = fopen(buffer, "w");
        if(mygemm_kernel == NULL)
            return -2;

        for(int dim = start; dim <= end; dim += step) {
            double gflops = 2 * (double) dim * dim * dim * 1e-9;
            printf("Total GFLOPS = %lf\n", gflops);

            /* Create matrices. */
            float *A = new_mat(dim, dim);
            float *B = new_mat(dim, dim);
            float *C = new_mat(dim, dim);

            double ttook_best = INFINITY;

            /* MyGEMM performance graph for row-major ordering (dimension vs GFLOPS). */
            for(int rep = 0; rep < repeat; rep++) {
                double ttook = time_snap();
                custom_mygemm_ij(dim, dim, dim, A, dim, B, dim, C, dim, kernels[i]);
                ttook = time_snap() - ttook;

                if(rep == 0)   // First iteration, the observed time is the best time
                    ttook_best = ttook;
                else ttook_best = ttook < ttook_best ? ttook : ttook_best;
                printf("%d. %dx%d Kernel, nxn=%d, time took: %lf\n", rep, C_MR, C_NR, dim, ttook);
            }
            double achieved_gflops = gflops / ttook_best;
            fprintf(mygemm_kernel, "%d        %lf\n", dim, achieved_gflops);
            printf("Asif's GEMM repetation done. Took (best) = %lf, GFLOPS = %lf\n", ttook_best, achieved_gflops);
            fflush(mygemm_kernel);

            free(A);
            free(B);
            free(C);
        }

        fclose(mygemm_kernel);
    }
}
