#! /usr/bin/env zsh

gnuplot -persist gplot/plot_aggr.gp&
gnuplot -persist gplot/plot_speedup.gp&
gnuplot -persist gplot/plot_gflopspth.gp&
