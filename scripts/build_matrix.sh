#!/bin/bash

mkdir -p data/training


python code/build_matrix/date_fruit.py
python code/build_matrix/fetal_health.py
python code/build_matrix/radar.py
python code/build_matrix/cifar10.py
python code/build_matrix/mnist.py