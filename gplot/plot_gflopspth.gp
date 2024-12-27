set terminal qt font "Rubik,14"

threads = 12

set title "GFLOPS per Thread"
set xlabel "Matrix shape (NxN)"
set ylabel "GFLOPS/thread"
plot '< paste data/mt/mkl_aggr.dat' u ($2/threads) lt 7 lc 6 lw 2 w lp title "Intel MKL", \
'< paste data/mt/mygemm_aggr.dat' u ($2/threads) lt 7 lc 15 lw 2 w lp title "Asif's GEMM"
