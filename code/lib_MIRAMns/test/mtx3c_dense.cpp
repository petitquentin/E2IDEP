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

    // Params
    bool debug = false;
    std::string pathInput = "../data/dense/test.mtx";
    for (int i = 0; i < argc; i++)
    {
        if (strcasecmp(argv[i], "-input") == 0 || strcasecmp(argv[i], "-i") == 0)
        {
            pathInput = argv[i + 1];
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
    if (debug)
    {
        std::cout << "-----------" << std::endl;
        std::cout << "Parameters:" << std::endl;

        std::cout << " - Input path : " << pathInput << std::endl;
        std::cout << " - Debug mode : " << debug << std::endl;
        std::cout << "-----------" << std::endl;
    }

    unsigned long int nb_rows, nb_cols;

    double *v = NULL;

    read_mtx_dense(pathInput, &v, nb_rows, nb_cols, MPI_COMM_WORLD);
    if (debug)
    {
        std::cout << "Read the message OK" << std::endl;
    }

    std::cout << std::endl
              << "Number of rows : " << nb_rows << std::endl;
    std::cout << std::endl
              << "Number of cols : " << nb_cols << std::endl;

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
                for (int r = 0; r < (int)nb_rows; r++)
                {
                    for (int c = 0; c < v_size[rank]; c++)
                    {
                        std::cout << v[r * v_size[rank] + c] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    free(v);
    free(v_size);

    MPI_Finalize();

    return 0;
}