#!/bin/bash

mkdir -p log
mkdir -p log/scalability
mkdir -p log/scalability/dense

echo "Variable definition";

restart=0

size=(1 2 4 8 16 32 64 128 256 512)

nb_it=$((5 * ${#size[@]}))
cit=0

        for s in $(seq 0 $((${#size[@]} - 1)))
        do
            #echo "" > ./log/scalability/dense/mnist_${size[${s}]}_mi.log
            #echo "" > ./log/scalability/dense/mnist_${size[${s}]}_i.log
            for it in $(seq 0 10)
            do
                cit=$(($cit + 1))
                echo "${cit} on ${nb_it} : "
                { srun -p cpu_short --time=00:15:00 --ntasks=${size[${s}]} ./code/lib_MIRAMns/build/test/scalability_eigenvectors -i data/tmp/mnist_cov.mtx -k 5 -l 0 ; } >> ./log/scalability/dense/mnist_${size[${s}]}_mi.log
                { srun -p cpu_short --time=00:15:00 --ntasks=${size[${s}]} ./code/lib_MIRAMns/build/test/scalability_eigenvectors -i data/tmp/mnist_cov.mtx -k 5 -l 10 ; } >> ./log/scalability/dense/mnist_${size[${s}]}_i.log 
           
            done
            python ./code/lib_MIRAMns/build/tools/python/means.py --input ./log/scalability/dense/mnist_${size[${s}]}_mi
            python ./code/lib_MIRAMns/build/tools/python/means.py --input ./log/scalability/dense/mnist_${size[${s}]}_i
        done

wait