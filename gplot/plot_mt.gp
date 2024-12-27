set terminal qt font "Rubik,14"

threads = 12

set title "Aggregated GFLOPS"
set xlabel "Matrix shape (NxN)"
set ylabel "GFLOPS"
plot "data/mt/mkl_aggr.dat" title "Intel MKL" lt 7 lc 6 lw 2 w lp, \
"data/mt/mygemm_aggr.dat" title "MyGEMM" lt 7 lc 15 lw 2 w lp

set title "Relative Speedup w.r.t. Single Core"
set ylabel "Speedup factor"
plot '< paste data/mt/mkl_aggr.dat data/mt/mkl.dat' u ($2/$4) lt 7 lc 6 lw 2 w lp, \
'< paste data/mt/mygemm_aggr.dat data/mt/mygemm.dat' u ($2/$4) lt 7 lc 15 lw 2 w lp

set title "GFLOPS per Thread"
set ylabel "GFLOPS/thread"
plot '< paste data/mt/mkl_aggr.dat' u ($2/threads) lt 7 lc 6 lw 2 w lp, \
'< paste data/mt/mygemm_aggr.dat' u ($2/threads) lt 7 lc 15 lw 2 w lp
