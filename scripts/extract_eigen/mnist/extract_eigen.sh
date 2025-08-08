#!/bin/bash

mkdir -p data
mkdir -p data/tmp
mkdir -p data/output
mkdir -p data/output/mnist

mkdir -p log

#Find eigenvalues and eigenvectors for mnist covaraince matrix
echo "Start find eigenvectors";
for k in $(seq 1 50);
do
    { srun -p cpu_short --ntasks=10 ./code/lib_MIRAMns/build/test/eigenvectors_save -i data/tmp/mnist_cov.mtx -o data/output/mnist/eigen_${k}.mtx -k ${k} -l 100 ; } > log/mnist_${k}.log &
done

wait