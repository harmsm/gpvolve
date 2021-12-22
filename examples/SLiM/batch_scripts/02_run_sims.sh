#!/bin/bash
# for each map size/roughness combination,
# permute mutation rate [1e-1, 1e-2, 1e-3, 1e-4]
# permute population size [50, 100, 500, 1000, 5000]
# submit run_sim.batch (runs simulations for all 10 maps)

mkdir out working_dir slim_dir  # output directories

for L in {3..8}; do
    for rf in 0.{1..5}; do
        for u in 1e-{1..4}; do
            for N in 50 100 500 1000 5000; do
                sbatch run_slim.batch $L $N $u $rf
            done
        done
    done
done
