#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>
#include <iofiles/iofiles.hpp>
#include <utils/utils.hpp>
#include <tools/tools.hpp>
#include <algorithms/algorithms.hpp>
#include <mpi.h>

// sudo yum install protobuf-compiler
// sudo yum install protobuf-devel

int main(int argc, char *argv[])
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    MPI_Init(NULL, NULL);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    srand(time(NULL) + rank);

    auto begin = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    // double cpu_time;

    // Params
    bool debug = false;
    std::string pathInput = "../data/square/bfw62a.mtx";

    double tolerance = 1e-8;
    int max_it = 500;
    int lmax = 0;
    int k_size;
    for (int i = 0; i < argc; i++)
    {
        if (strcasecmp(argv[i], "-input") == 0 || strcasecmp(argv[i], "-i") == 0)
        {
            pathInput = argv[i + 1];
        }
        if (strcasecmp(argv[i], "-kdesired") == 0 || strcasecmp(argv[i], "-k") == 0)
        {
            k_size = atoi(argv[i + 1]);
        }
        if (strcasecmp(argv[i], "-d") == 0 || strcasecmp(argv[i], "-debug") == 0)
        {
            debug = true;
        }
        if (strcasecmp(argv[i], "-m") == 0 || strcasecmp(argv[i], "-max") == 0)
        {
            max_it = atoi(argv[i + 1]);
        }
        if (strcasecmp(argv[i], "-l") == 0 || strcasecmp(argv[i], "-lmax") == 0)
        {
            lmax = atoi(argv[i + 1]);
        }
        if (strcasecmp(argv[i], "-t") == 0 || strcasecmp(argv[i], "-tolerance") == 0)
        {
            tolerance = atof(argv[i + 1]);
        }
        if (strcasecmp(argv[i], "-help") == 0 || strcasecmp(argv[i], "-h") == 0)
        {
            std::cout << "Parameters :" << std::endl;
            std::cout << " -i or -input : the path to the input protobuf file (extracted from Mindspore)" << std::endl;
            std::cout << " -k or -kdesired : The number of desired dominant eigenvalues and eigenvectors" << std::endl;
            std::cout << " -m or -max : The maximum number of iteration (default value = 500)" << std::endl;
            std::cout << " -l or -lmax : The size of subspaces research, should be greater than 2* k (default value = 2 * k)" << std::endl;
            std::cout << " -t or -tolerance : Tolerance (max error) (default value = 1e-8)" << std::endl;
            std::cout << " -d or -debug : Activate debug mode" << std::endl;

            return 0;
        }
    }
    assert(k_size > 0);
    if (lmax < 2 * k_size)
    {
        if (rank == 0)
        {
            std::cout << "Invalid lmax (l parameter): change by 2 * k (previous value : " << lmax << ", new : " << 2 * k_size << ")" << std::endl;
        }
        lmax = 2 * k_size;
    }
    if (debug && rank == 0)
    {
        std::cout << "-----------" << std::endl;
        std::cout << "Parameters:" << std::endl;

        std::cout << " - Input path : " << pathInput << std::endl;
        std::cout << " - k : " << k_size << std::endl;
        std::cout << " - Max iteration : " << max_it << std::endl;
        std::cout << " - Subspace interval max : " << lmax << std::endl;
        std::cout << " - Tolerance : " << tolerance << std::endl;
        std::cout << " - Debug mode : " << debug << std::endl;
        std::cout << "-----------" << std::endl;
    }

    unsigned long int nb_rows, nb_cols;
    get_mtx_dim(pathInput, nb_rows, nb_cols);

    if (debug && rank == 0)
    {
        std::cout << "Matrix size : " << nb_rows << " " << nb_cols << std::endl;
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
    if (debug && rank == 0)
    {
        std::cout << "v_size = [" << v_size[0];
        for (unsigned long int i = 1; i < (unsigned long)p; i++)
        {
            std::cout << ", " << v_size[i];
        }
        std::cout << "]" << std::endl;
    }

    Eigen::MatrixXd disA = Eigen::MatrixXd::Zero(v_size[rank], nb_cols);
    disA = (read_mtx(pathInput)).block(startRow, 0, v_size[rank], nb_cols);

    long double norm_A = 0;
    for (int i = 0; i < v_size[rank]; i++)
    {
        for (unsigned long int j = 0; j < nb_cols; j++)
        {
            norm_A += disA(i, j) * disA(i, j);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &norm_A, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    norm_A = std::sqrt(norm_A);

    // some miram parameters
    Eigen::VectorXcd desired_eigenvalues_Hmm(k_size);
    Eigen::VectorXd residuals(k_size);
    Eigen::MatrixXcd U_desired(disA.cols(), k_size);

    MPI_Barrier(MPI_COMM_WORLD);
    begin = std::chrono::high_resolution_clock::now();
    distributed_iram(disA, norm_A, lmax, k_size, tolerance, max_it, desired_eigenvalues_Hmm, U_desired, v_size, MPI_COMM_WORLD);
    // distributed_multi_iram(disA, numVectors, k_size, tolerance, max_it, desired_eigenvalues_Hmm, U_desired, subspaces_size, norm_A, v_size, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();
    // output
    if (rank == 0)
    {
        std::cout << "Desired eigen values: " << desired_eigenvalues_Hmm << "\n";
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / std::pow(10, 9) << " sec" << std::endl;
    }
    free(v_size);

    MPI_Finalize();
    return 0;
}