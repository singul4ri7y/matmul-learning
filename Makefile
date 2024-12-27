CC = gcc
CFLAGS = -std=gnu99 -Wall -Wextra -pedantic -march=native -O3 -mavx2 -fopenmp -lpthread -ffast-math
LFLAGS = -L/opt/intel/mkl/lib/intel64/ -lmkl_rt -fopenmp
INCLUDES = -I/opt/intel/mkl/include

OBJ_FILES = obj/util.c.obj obj/mygemm.c.obj obj/kernel.c.obj

all: $(OBJ_FILES) obj/driver.c.obj
	$(CC) $(LFLAGS) $^ -o output/matmul

KCxNC: $(OBJ_FILES) obj/driver_KCxNC.c.obj
	$(CC) $(LFLAGS) $^ -o output/matmul_KCxNC

MC: $(OBJ_FILES) obj/driver_MC.c.obj
	$(CC) $(LFLAGS) $^ -o output/matmul_MC

kernel: $(OBJ_FILES) obj/driver_kernel.c.obj
	$(CC) $(LFLAGS) $^ -o output/matmul_kernel

mt: $(OBJ_FILES) obj/driver_mt.c.obj
	$(CC) $(LFLAGS) $^ -o output/matmul_mt

clean:
	rm -rf obj/* output/*

obj/%.c.obj: %.c
	$(CC) $(CFLAGS) $(INCLUDES) $(MYFLAGS) -c $< -o $@
