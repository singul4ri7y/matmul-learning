#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mkl.h>

#include "util.h"
#include "mygemm.h"

int main(int argc, char **argv) {
    if(argc < 5) {
        printf("Invalid argument count. <repeat_operations> <start_dim> <end_dim> <dim_step>\n");
        return -1;
    }

    int repeat = atoi(argv[1]);
    int start = atoi(argv[2]);
    int end = atoi(argv[3]);
    int step = atoi(argv[4]);

    FILE *mklf = fopen("data/perf/mkl.dat", "w");
    if(mklf == NULL) 
        return -2;
    FILE *mgemmf = fopen("data/perf/mygemm.dat", "w");
    if(mgemmf == NULL) 
        return -2;

    for(; start <= end; start += step) {
        double gflops = 2 * (double) start * start * start * 1e-9;
        printf("Total GFLOPS = %lf\n", gflops);

        /* Create matrices. */
        float *A = new_mat(start, start);
        float *B = new_mat(start, start);
        float *C = new_mat(start, start);
        float *Cold = alloc_mat(start, start);

        double ttook_best = INFINITY;

        /* Intel MKL performance graph (dimension vs GFLOPS). */
        for(int i = 0; i < repeat; i++) {
            memset(Cold, 0, start * start * sizeof(float));
            double ttook = time_snap();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, start, start, start, 1.0, A, start, B, start, 1.0, Cold, start);
            ttook = time_snap() - ttook;

            if(i == 0)   // First iteration, the observed time is the best time
                ttook_best = ttook;
            else ttook_best = ttook < ttook_best ? ttook : ttook_best;
            printf("Intel MKL nxn=%d rep=%d, time took: %lf\n", start, i, ttook);
        }
        double achieved_gflops = gflops / ttook_best;
        fprintf(mklf, "%d        %lf\n", start, achieved_gflops);
        printf("Intel MKL repetition done. Took (best) = %lf, GFLOPS = %lf\n", ttook_best, achieved_gflops);
        fflush(mklf);

        /* MyGEMM performance graph for row-major ordering (dimension vs GFLOPS). */
        for(int i = 0; i < repeat; i++) {
            double ttook = time_snap();
            mygemm(start, start, start, A, start, B, start, C, start);
            ttook = time_snap() - ttook;

            if(i == 0)   // First iteration, the observed time is the best time
                ttook_best = ttook;
            else ttook_best = ttook < ttook_best ? ttook : ttook_best;
            printf("Asif's GEMM nxn=%d rep=%d, time took: %lf\n", start, i, ttook);
        }
        achieved_gflops = gflops / ttook_best;
        fprintf(mgemmf, "%d        %lf\n", start, achieved_gflops);
        printf("Asif's GEMM repetition done. Took (best) = %lf, GFLOPS = %lf\n", ttook_best, achieved_gflops);
        fflush(mgemmf);

        /* Check for correctness. If the error threshold is not acceptable, terminate. */
        // for(int i = 0; i < start; i++) {
        //     for(int j = 0; j < start; j++) {
        //         float diff = Cold[i * start + j] - C[i * start + j];
        //         if(fabsf(diff) > 1e-3) {
        //             fprintf(stderr, "There might be some error with the implementation, the difference is %f - %f = %f at (%d, %d)\n", 
        //                     Cold[i * start + j], C[i * start + j], diff, i, j);

        //             free(A);
        //             free(B);
        //             free(C);
        //             free(Cold);
        //             fclose(mklf);
        //             fclose(mgemmf);

        //             return -3;
        //         }
        //     }
        // }

        free(A);
        free(B);
        free(C);
        free(Cold);
    }

    fclose(mklf);
    fclose(mgemmf);
}
