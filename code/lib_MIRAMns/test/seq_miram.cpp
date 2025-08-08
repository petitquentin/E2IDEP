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

// sudo yum install protobuf-compiler
// sudo yum install protobuf-devel

int main(int argc, char *argv[])
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

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
            std::cout << " -l or -lmax : The maximum size of subspaces interval research, should be greater than 2* k (default value = 2 * k)" << std::endl;
            std::cout << " -t or -tolerance : Tolerance (max error) (default value = 1e-8)" << std::endl;
            std::cout << " -d or -debug : Activate debug mode" << std::endl;

            return 0;
        }
    }
    assert(k_size > 0);
    if (lmax < 2 * k_size)
    {
        std::cout << "Invalid lmax (l parameter): change by 2 * k (previous value : " << lmax << ", new : " << 2 * k_size << ")" << std::endl;
        lmax = 2 * k_size;
    }
    if (debug)
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

    Eigen::MatrixXd A = read_mtx(pathInput);
    /*MatrixXd B = A.transpose();
    MatrixXd C = A + B;
    A = C - MatrixXd(A.diagonal().asDiagonal())
double nrmfr = A.norm();*/

    // some miram parameters
    Eigen::VectorXcd desired_eigenvalues_Hmm(k_size);
    Eigen::VectorXd residuals(k_size);
    Eigen::MatrixXcd U_desired(A.cols(), k_size);

    // Subspace autom
    int subspace_size = (lmax - (2 * k_size)) + 1;
    Eigen::VectorXd subspaces_size(subspace_size);
    for (int i = 0; i < subspace_size; i++)
    {
        subspaces_size(i) = 2 * k_size + i;
    }
    int numVectors = subspaces_size(subspace_size - 1);

    begin = std::chrono::high_resolution_clock::now();
    sequential_multi_iram(A, numVectors, k_size, tolerance, max_it, desired_eigenvalues_Hmm, U_desired, subspaces_size, A.norm());

    end = std::chrono::high_resolution_clock::now();
    // output
    std::cout << "Desired eigen values: " << desired_eigenvalues_Hmm << "\n";
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " [ns]" << std::endl;

    return 0;
}