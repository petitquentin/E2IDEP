#!/bin/bash

mkdir -p log
mkdir -p log/scalability
mkdir -p log/scalability/sparse

echo "Variable definition";

size=(1 2 4 8 16 32 64 128 256 512 1000)
ds=(af23560 bfw782a mnist cifar10)

nb_it=$((5 * ${#size[@]}))
cit=0

        for s in $(seq 0 $((${#size[@]} - 1)))
        do
            #echo "" > ./log/scalability/sparse/cifar10_${size[${s}]}_mi.log
            #echo "" > ./log/scalability/sparse/cifar10_${size[${s}]}_i.log
            for d in $(seq 0 $((${#ds[@]} - 1)))
            do

                python ./code/lib_MIRAMns/build/tools/python/means.py --input ./log/scalability/sparse/${ds[${d}]}_${size[${s}]}_mi
                python ./code/lib_MIRAMns/build/tools/python/means.py --input ./log/scalability/sparse/${ds[${d}]}_${size[${s}]}_i
                python ./code/lib_MIRAMns/build/tools/python/means.py --input ./log/scalability/dense/${ds[${d}]}_${size[${s}]}_mi
                python ./code/lib_MIRAMns/build/tools/python/means.py --input ./log/scalability/dense/${ds[${d}]}_${size[${s}]}_i
            done
        done

wait