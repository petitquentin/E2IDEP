#!/bin/bash

mkdir -p data
mkdir -p data/tmp
mkdir -p data/output
mkdir -p data/output/radar

mkdir -p log

#Find eigenvalues and eigenvectors for mnist covaraince matrix
echo "Start find eigenvectors";
for k in $(seq 1 50);
do
    { srun -p cpu_med --ntasks=10 ./code/lib_MIRAMns/build/test/eigenvectors_save -i tmp/radar_cov.mtx -o output/radar/eigen_${k}.mtx -k ${k} -l 200 ; } > log/radar_${k}.log &
done

wait