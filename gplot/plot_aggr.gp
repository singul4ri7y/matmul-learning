set terminal qt font "Rubik,14"

set title "Aggregated GFLOPS"
set xlabel "Matrix shape (NxN)"
set ylabel "GFLOPS"
plot "data/mt/mkl_aggr.dat" title "Intel MKL" lt 7 lc 6 lw 2 w lp, \
"data/mt/mygemm_aggr.dat" title "MyGEMM" lt 7 lc 15 lw 2 w lp
