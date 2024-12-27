set terminal wxt
set xrange[0:512]
set yrange[0:512]
plot "../data/mygemm_KCxNC.dat" with image, 80 * 256 / x lt 7 lc 6 dt 2 title "L1 Cache (80KB)", 512 * 256 / x lt 7 lc 0 dt 3 title "L2 Cache (512KB)"
