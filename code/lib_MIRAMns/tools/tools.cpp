// #include "../include/tools.hpp"
// #include "../include/algorithms.hpp"

#include <algorithms/algorithms.hpp>
#include <tools/tools.hpp>
#include <bits/stdc++.h>
#include <cassert>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <fstream>

/*##################################################################################################################
        Description of each of these input functions are available in include/tools.hpp file.
##################################################################################################################*/

Eigen::VectorXd generate_random_vector(int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Eigen::VectorXd randomVector(size);
    for (int i = 0; i < size; ++i)
    {
        randomVector(i) = dis(gen);
    }

    return randomVector;
}

void QR(Eigen::MatrixXcd &H_mm, Eigen::MatrixXcd &V_m, Eigen::VectorXcd &eigenvalues_Hmm, int numVectors, int desired)
{
    int j = 0;
    int k = 1;
    Eigen::MatrixXcd Q;
    while (j < numVectors - desired)
    {
        if (abs(eigenvalues_Hmm(j).imag()) == 0) // eigenvalues_Hmm(j) is real : single shift
        {
            Eigen::HouseholderQR<Eigen::MatrixXcd> qr(H_mm - eigenvalues_Hmm(j) * Eigen::MatrixXcd::Identity(numVectors, numVectors));
            Q = qr.householderQ();
        }
        else //  eigenvalues_Hmm(j) is imaginary : double shift
        {
            Eigen::HouseholderQR<Eigen::MatrixXcd> qr((H_mm - eigenvalues_Hmm(j) * Eigen::MatrixXcd::Identity(numVectors, numVectors)) *
                                                          (H_mm - eigenvalues_Hmm(j) * Eigen::MatrixXcd::Identity(numVectors, numVectors)) +
                                                      pow(eigenvalues_Hmm(j).imag(), 2) * Eigen::MatrixXcd::Identity(numVectors, numVectors));
            Q = qr.householderQ();
        }

        // Hmm = Q.H * Hmm * Q
        H_mm = Q.adjoint() * H_mm * Q;
        // Vm ← Vm Qj
        V_m = V_m * Q;

        // Clean up rounding error noise below the first subdiagonal
        for (k = 1; k < numVectors - 1; k++)
        {
            H_mm.block(k + 1, k - 1, numVectors - k - 1, 1) *= 0;
        }

        if (abs(eigenvalues_Hmm(j).imag()) > 0)
        {
            j = j + 2;
        }
        else
        {
            j = j + 1;
        }
    }
}

void eigs(Eigen::MatrixXcd &H_mm, Eigen::MatrixXcd &eigenvectors_Hmm, Eigen::VectorXcd &eigenvalues_Hmm)
{
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigensolver;
    eigensolver.compute(H_mm);
    eigenvalues_Hmm = eigensolver.eigenvalues();
    eigenvectors_Hmm = eigensolver.eigenvectors();
}

Eigen::MatrixXd read_mtx(std::string dataset_path)
{
    Eigen::MatrixXd A;
    std::ifstream fin(dataset_path);
    if (fin)
    {

        int M, N, L;
        while (fin.peek() == '%')
            fin.ignore(2048, '\n');
        fin >> M >> N >> L;
        A = Eigen::MatrixXd::Zero(M, N);
        for (int l = 0; l < L; l++)
        {
            int m, n;
            double data;
            fin >> m >> n >> data;
            A(m - 1, n - 1) = data;
        }
        fin.close();
    }
    else
    {
        std::cerr << "Impossible to open the file " << dataset_path << std::endl;
    }
    return A;
}

Eigen::MatrixXd read_mtx(std::string dataset_path, unsigned long int startRow, unsigned long int startCol, unsigned long int nb_rows, unsigned long int nb_cols)
{
    Eigen::MatrixXd A;
    std::ifstream fin(dataset_path);
    if (fin)
    {

        unsigned long int M, N, L;
        while (fin.peek() == '%')
            fin.ignore(2048, '\n');
        fin >> M >> N >> L;
        A = Eigen::MatrixXd::Zero(nb_rows, nb_cols);
        for (unsigned long int l = 0; l < L; l++)
        {
            unsigned long int m, n;
            double data;
            fin >> m >> n >> data;
            m--;
            n--;
            if(m >= startRow && m < startRow + nb_rows && n >= startCol && n < startCol + nb_cols){
                A(m-startRow, n -startCol) = data;
            }
        }
        fin.close();
    }
    else
    {
        std::cerr << "Impossible to open the file " << dataset_path << std::endl;
    }
    return A;
}


Eigen::SparseMatrix<double> read_mtx_sparse(std::string dataset_path, unsigned long int startRow, unsigned long int startCol, unsigned long int nb_rows, unsigned long int nb_cols)
{
    Eigen::SparseMatrix<double> A;
    std::ifstream fin(dataset_path);
    if (fin)
    {

        unsigned long int M, N, L;
        while (fin.peek() == '%')
            fin.ignore(2048, '\n');
        fin >> M >> N >> L;
        A = Eigen::SparseMatrix<double>(nb_rows, nb_cols);
        for (unsigned long int l = 0; l < L; l++)
        {
            unsigned long int m, n;
            double data;
            fin >> m >> n >> data;
            m--;
            n--;
            if(m >= startRow && m < startRow + nb_rows && n >= startCol && n < startCol + nb_cols){
                A.coeffRef(m-startRow, n -startCol) = data;
            }
        }
        fin.close();
    }
    else
    {
        std::cerr << "Impossible to open the file " << dataset_path << std::endl;
    }
    A.makeCompressed();
    return A;
}
