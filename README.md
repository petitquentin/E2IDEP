[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16781371.svg)](https://doi.org/10.5281/zenodo.16781371)

# Efficient Embedding Initialization via Dominant Eigenvector Projections

This repository contains the source code, data, and experimental scripts used in the paper for the submission to the ScalAH Workshop 2025. It supports reproducibility of the results and figures presented in the manuscript.

## Overview

We propose an efficient initialization method for embedding layers in deep learning models using dominant eigenvector projections. This approach leverages spectral properties of co-occurrence matrices to produce informative starting points for training. Our implementation integrates:
 - Python preprocessing to extract dataset information and prepare data
 -  Parallel C++ (MPI) for scalable eigenvector extraction (via MIRAMns)
 - Deep Learning training pipeline in Tensorflow and MindSpore

## Project structure

```
code/
  ├── build_matrix/        # Python scripts for preprocessing and building matrices
  ├── lib_MIRAMns/           # CMake project (C++11, MPI) implementing MIRAMns
data/
  ├── *.csv                # Small datasets
  └── README.md            # Instructions to download large datasets
models/
  ├── mindspore/           # Mindspore models and generated figures
  └── tensorflow/          # Tensorlfow models and figures output
scripts/
  ├── build_matrix.sh      # Run preprocessing pipeline
  ├── make_MIRAMns_lib.sh  # Compile the MIRAMns C++ library
  └── extract_eigen/
      ├── <dataset>/extract_eigen.sh         # Run eigenvector extraction on SLURM (Ruche cluster)
      └── <dataset>/local_extract_eigen.sh   # Alternative script (no SLURM required)
```


Details of the C++ library lib_MIRAMns:
```
lib_MIRAMns/
  ├── algorithms/          # Core matrix factorization algorithms (Arnoldi, IRAM, MIRAM)
  ├── data/                # Matrix files for testing
  ├── include/             # Include files
  ├── iofiles/
  ├── proto/               # Protobuf definition
  ├── test/                # Test and example files to use the library
  ├── tools/               # Matrix manipulation functions
  │   ├── python/          # Python compute means of execution time
  ├── utils/               # Utility functions (printing, matrix transformations, etc.)
  └── CMakeLists.txt       # CMake configuration file to build the project
```

## Requirements

The following software and libraries are required:

| Library / Framework | Usage |
| --- | --- |
| [GCC](https://gcc.gnu.org/gcc-11/) | Compiler |
| [CMake](https://cmake.org/) | Build system |
| [Eigen](https://eigen.tuxfamily.org/) | linear algebra |
| [MPI](https://www.open-mpi.org/) | distributed computing |
| [Protobuf](https://protobuf.dev/) | encoding / decoding data |
| [Tensorflow](https://www.tensorflow.org/) | Deep model training and evaluation |
| [MindSpore](https://www.mindspore.cn/) | Deep model training and evaluation |

### Installation (Ubuntu/Debian)

```bash
# Eigen
sudo apt install libeigen3-dev

# MPI
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev

# Protobuf
sudo apt install protobuf-compiler libprotobuf-dev
```

For MindSpore installation, please refer to [MindSpore installation guide](https://www.mindspore.cn/install/en) 

## Quick Start


1. Prepare Data
    1. Go to ```data/README.md``` to download and organize large datasets.
    2. Deploy a python environment with the packages lsited in ```data/requirements.txt```
    3. Preprocess data and build matrices:
    ```bash
    bash scripts/build_matrix.sh
    ```
2. Compile the MIRAMns Library
    ```bash
    bash scripts/make_MIRAMns_lib.sh
    ```
    This builds the lib_MIRAMns library (C++11, MPI).
3. Extract eigenvectors
    1. Build covariance matrix representation of datasets:
      - SLURM (Ruche cluster)
        ```bash
        bash scripts/build_covariance.sh
        ```
      - MPIRUN (no SLURM required)
        ```bash
        bash scripts/local_build_covariance.sh
        ```
    2. For each datasets, two scripts are available:
      - SLURM (Ruche cluster)
        ```bash
        bash scripts/extract_eigen/<dataset>/extract_eigen.sh
        ```
      - MPIRUN (no SLURM required)
        ```bash
        bash scripts/extract_eigen/<dataset>/local_extract_eigen.sh
        ```
> **_NOTE:_**  Precomputed eigenvectors are available in ```data/output/``` folder. Steps 1–3 can be skipped if only training is desired.

4. Deep learning models are trained using data extracted during the previous steps. 
    The folder contains notebooks with the Mindspore and TensorFlow frameworks.
    
    Please visit the Mindspore installation page to deploy the framework: [https://www.mindspore.cn/install/en](https://www.mindspore.cn/install/en).

## Reproduction of paper results

All figures in the paper are generated using PGF files in ```models/tensorflow/figures```. We adopt TensorFlow's deterministic mode to ensure reproducibility. To reproduce results:
  1. Run steps 1–4 above, or
  2. Use precomputed embeddings and directly run the training notebooks.
After the generation, we use scripts in ```models/tensorflow/postprocessfigure``` to add information to make the figure easier to understand. 