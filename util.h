#ifndef MATMUL_UTIL
#define MATMUL_UTIL

void rand_mat(float *data, int m, int n, int ldA);
void *alloc_mat(int m, int n);
void *new_mat(int m, int n);
double time_snap();

#endif  // MATMUL_UTIL
