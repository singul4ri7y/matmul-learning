#include <time.h>
#include <stdlib.h>

double time_snap() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void rand_mat(float *data, int m, int n, int ldA) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            data[i * ldA + j] = (float) drand48();
        }
    }
}

void *alloc_mat(int m, int n) {
    void *data;
    if(posix_memalign(&data, 64, m * n * sizeof(float))) 
        return NULL;
    return data;
}

void *new_mat(int m, int n) {
    void *data = alloc_mat(m, n);
    if(data == NULL) 
        return data;
    rand_mat(data, m, n, m);
    return data;
}

