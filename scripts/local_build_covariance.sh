#!/bin/bash

mkdir -p data
mkdir -p data/tmp
mkdir -p log



echo "Build covariance matrix CIFAR10";
mpirun -n 4   ./code/lib_MIRAMns/build/test/buildCovariance -i data/training/train_cifar10.mtx -o data/tmp/cifar10_cov.mtx > log/local_build_covariance_cifar_10.log ;

echo "Build covariance matrix DATE_FRUITS";
mpirun -n 4   ./code/lib_MIRAMns/build/test/buildCovariance -i data/training/train_date_fruits.mtx -o data/tmp/date_fruits_cov.mtx -d > log/local_build_covariance_date_fruits.log ;

echo "Build covariance matrix FETAL_HEALTH";
mpirun -n 4   ./code/lib_MIRAMns/build/test/buildCovariance -i data/training/train_fetal_health.mtx -o data/tmp/fetal_health_cov.mtx -d > log/local_build_covariance_fetal_health.log ;

echo "Build covariance matrix MNIST";
mpirun -n 4   ./code/lib_MIRAMns/build/test/buildCovariance -i data/training/train_mnist.mtx -o data/tmp/mnist_cov.mtx > log/local_build_covariance_mnist.log ;

echo "Build covariance matrix RADAR";
mpirun -n 4 ./code/lib_MIRAMns/build/test/buildCovariance -i data/training/train_radar_ds.mtx -o data/tmp/radar_cov.mtx -d > log/local_build_covariance_radar.log ;