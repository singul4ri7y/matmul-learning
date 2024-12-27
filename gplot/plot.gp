set terminal qt font "Rubik,14"

set xlabel "Matrix shape (NxN)"
set ylabel "GFLOPS"
plot "data/perf/mkl.dat" title "Intel MKL" lt 7 lc 6 lw 3 w lp, \
"data/perf/mygemm.dat" title "Asif's GEMM" lt 7 lc 15 lw 3 w lp
# "data/perf/mygemm_old.dat" title "Asif's GEMM (old)" lt 7 lc 3 w lp
