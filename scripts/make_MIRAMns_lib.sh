#!/bin/bash

mkdir -p code/lib_MIRAMns/build

cmake -S ./code/lib_MIRAMns/ -B ./code/lib_MIRAMns/build/

make -C ./code/lib_MIRAMns/build/ -j$(nproc)