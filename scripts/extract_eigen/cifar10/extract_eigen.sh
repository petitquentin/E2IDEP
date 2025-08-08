#!/bin/bash

mkdir -p data
mkdir -p data/tmp
mkdir -p data/output
mkdir -p data/output/cifar10

mkdir -p log

#Find eigenvalues and eigenvectors for mnist covaraince matrix
echo "Start find eigenvectors";
for k in $(seq 1 50);
do
    { srun -p cpu_med --ntasks=100 ./code/lib_MIRAMns/build/test/eigenvectors_save -i data/tmp/cifar10_cov.mtx -o data/output/cifar10/eigen_${k}.mtx -k ${k} -l 200 ; } > log/cifar10_${k}.log &
done

wait