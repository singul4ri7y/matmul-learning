#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "util.h"
#include "mygemm.h"

extern int KC, NC, MR, NR;

int main(int argc, char **argv) {
    if(argc < 5) {
        printf("Invalid argument count. <repeat_operations> <start_dim> <end_dim> <dim_step>\n");
        return -1;
    }

    int repeat = atoi(argv[1]);
    int start = atoi(argv[2]);
    int end = atoi(argv[3]);
    int step = atoi(argv[4]);

    FILE *mgemmf = fopen("data/empirical/mygemm_KCxNC.dat", "w");
    if(mgemmf == NULL) 
        return -2;

    for(KC = 4; KC <= 512; KC += 4) {
        for(NC = NR; NC <= 512; NC += NR) {
            double total_gflops_count = 0;
            int count = 0;
            for(int dim = start; dim <= end; dim += step, count++) {
                double gflops = 2 * (double) dim * dim * dim * 1e-9;
                // printf("Total GFLOPS achievable: %lf\n", gflops);

                /* Create matrices. */
                float *A = new_mat(dim, dim);
                float *B = new_mat(dim, dim);
                float *C = new_mat(dim, dim);

                double ttook_best = INFINITY;

                /* MyGEMM performance graph for row-major ordering (dimension vs GFLOPS). */
                for(int i = 0; i < repeat; i++) {
                    double ttook = time_snap();
                    mygemm(dim, dim, dim, A, dim, B, dim, C, dim);
                    ttook = time_snap() - ttook;

                    if(i == 0)   // First iteration, the observed time is the best time
                        ttook_best = ttook;
                    else ttook_best = ttook < ttook_best ? ttook : ttook_best;
                    // printf("Asif's GEMM nxn=%d rep=%d, time took: %lf\n", dim, repeat, ttook);
                }
                total_gflops_count += (gflops / ttook_best);

                free(A);
                free(B);
                free(C);
            }
            total_gflops_count /= count;
            // printf("All reps done, KC=%d, NC=%d, gflops=%lf\n", KC, NC, total_gflops_count);
            fprintf(mgemmf, "%d        %d        %lf\n", KC, NC, total_gflops_count);
            fflush(mgemmf);
        }
    }

    fclose(mgemmf);
}
