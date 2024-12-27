set terminal qt font "Rubik,14"

set title "Relative Speedup w.r.t. Single Core"
set xlabel "Matrix shape (NxN)"
set ylabel "Speedup factor"
plot '< paste data/mt/mkl_aggr.dat data/mt/mkl.dat' u ($2/$4) lt 7 lc 6 lw 2 w lp title "Intel MKL", \
'< paste data/mt/mygemm_aggr.dat data/mt/mygemm.dat' u ($2/$4) lt 7 lc 15 lw 2 w lp title "Asif's GEMM"

