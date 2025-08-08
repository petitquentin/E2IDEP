#!/bin/bash

mkdir -p data
mkdir -p data/tmp
mkdir -p data/output
mkdir -p data/output/date_fruit

mkdir -p log


#Find eigenvalues and eigenvectors for mnist covaraince matrix
echo "Start find eigenvectors";
for k in $(seq 1 17);
do
    { mpirun -n 4 ./code/lib_MIRAMns/build/test/eigenvectors_save -i data/tmp/date_fruits_cov.mtx -o data/output/date_fruit/eigen_${k}.mtx -k ${k} -l 200 ; } > log/date_fruit_${k}.log 
done

wait