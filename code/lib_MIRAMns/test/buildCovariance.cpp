#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>
#include <iofiles/iofiles.hpp>
#include <utils/utils.hpp>
#include <mpi.h>

// sudo yum install protobuf-compiler
// sudo yum install protobuf-devel

int main(int argc, char *argv[])
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    auto begin = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time;

    // Params
    bool debug = false;
    std::string pathInput = "../data/dense/testcov2.mtx";
    std::string pathOutput = "../../output/outputcov.mtx";
    for (int i = 0; i < argc; i++)
    {
        if (strcasecmp(argv[i], "-input") == 0 || strcasecmp(argv[i], "-i") == 0)
        {
            pathInput = argv[i + 1];
        }
        if (strcasecmp(argv[i], "-output") == 0 || strcasecmp(argv[i], "-o") == 0)
        {
            pathOutput = argv[i + 1];
        }
        if (strcasecmp(argv[i], "-d") == 0 || strcasecmp(argv[i], "-debug") == 0)
        {
            debug = true;
        }
        if (strcasecmp(argv[i], "-help") == 0 || strcasecmp(argv[i], "-h") == 0)
        {
            std::cout << "Parameters :" << std::endl;
            std::cout << " -i or -input : the path to the input protobuf file (extracted from Mindspore)" << std::endl;
            std::cout << " -o or -output : the path to the output protobuf file (the matrix save)" << std::endl;
            std::cout << " -d or -debug : Activate debug mode" << std::endl;

            return 0;
        }
    }
    if (debug && rank == 0)
    {
        std::cout << "-----------" << std::endl;
        std::cout << "Parameters:" << std::endl;

        std::cout << " - Input path : " << pathInput << std::endl;
        std::cout << " - Output path : " << pathOutput << std::endl;
        std::cout << " - Debug mode : " << debug << std::endl;
        std::cout << "-----------" << std::endl;
    }

    unsigned long int nb_rows, nb_cols;

    double *v = NULL;
    double *cov = NULL;
    if (debug && rank == 0)
    {
        std::cout << "Nb procs : " << p << std::endl;
        std::cout << "Read the message OK" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    begin = std::chrono::high_resolution_clock::now();

    read_mtx_dense(pathInput, &v, nb_rows, nb_cols, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    if (debug && rank == 0)
    {
        std::cout << " - OK (time : " << cpu_time / std::pow(10, 9) << ")" << std::endl;
    }

    if (debug && rank == 0)
    {
        std::cout << "Build Covariance matrix" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    begin = std::chrono::high_resolution_clock::now();
    build_covariance(v, &cov, nb_rows, nb_cols, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    if (debug && rank == 0)
    {
        std::cout << " - OK (time : " << cpu_time / std::pow(10, 9) << ")" << std::endl;
    }

    int *v_size = (int *)malloc(p * sizeof(int));
    unsigned long int startRow = 0;

    for (unsigned long int i = 0; i < (unsigned long)p; i++)
    {
        v_size[i] = std::floor(nb_cols * 1.0 / p);
        if (i < nb_cols % p)
        {
            v_size[i] = v_size[i] + 1;
        }
        if (i < (unsigned long)rank)
        {
            startRow += v_size[i];
        }
    }
    if (debug)
    {
        for (int i = 0; i < p; i++)
        {
            if (i == rank)
            {
                std::cout << std::endl
                          << "Rank = " << rank << std::endl;
                for (int r = 0; r < v_size[rank]; r++)
                {
                    for (int c = 0; c < (int)nb_cols; c++)
                    {
                        std::cout << cov[r * nb_cols + c] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    std::cout << "Save in mtx file " << std::endl;

    save_mtx_dense(cov, nb_cols, nb_cols, pathOutput, MPI_COMM_WORLD);
    // save_mtx_dense(cov, 0, 0, pathOutput, MPI_COMM_WORLD);

    free(v);
    free(v_size);
    free(cov);

    MPI_Finalize();

    return 0;
}