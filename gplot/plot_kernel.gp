set terminal qt font "Rubik,14"
set xlabel "Matrix shape (NxN)"
set ylabel "GFLOPS"
set title "Performance of different μKernels"

plot "data/kernel/mygemm_4x24kernel.dat" title "4x24" lt 7 lw 2 lc 6 w lp, \
"data/kernel/mygemm_8x16kernel.dat" title "8x16" lt 7 lc 12 lw 2 w lp, \
"data/kernel/mygemm_12x16kernel.dat" title "12x16" lt 7 lc 3 lw 2 w lp, \
"data/kernel/mygemm_14x8kernel.dat" title "14x8" lt 7 lc 5 lw 2 w lp
