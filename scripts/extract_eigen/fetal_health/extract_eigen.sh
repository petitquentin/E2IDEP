#!/bin/bash

mkdir -p data
mkdir -p data/tmp
mkdir -p data/output
mkdir -p data/output/fetal_health

mkdir -p log


#Find eigenvalues and eigenvectors for mnist covaraince matrix
echo "Start find eigenvectors";
for k in $(seq 1 10);
do
    { srun -p cpu_med --ntasks=10 ./code/lib_MIRAMns/build/test/eigenvectors_save -i data/tmp/fetal_health_med_cov.mtx -o data/output/fetal_health/eigen_${k}.mtx -k ${k} -l 200 ; } > log/fetal_health_${k}.log &
done

wait