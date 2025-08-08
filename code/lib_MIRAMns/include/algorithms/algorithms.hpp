#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP

#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <eigen3/Eigen/Dense>
#include <Eigen/SparseCore>
#include <complex>
#include <math.h>

// using namespace std;
// using namespace Eigen;

/*##################################################################################################################
        This file contains the following algorithms : Arnoldi reduction, Ritz peers, IRAM and MIRAM.
##################################################################################################################*/

/*
 Description :
     This function realize the computation of a (m1-k),(m2-k) and (m3-k)-step Arnoldi factorization ***
 k=1 corresponds to a complete factorization
 k>1 corresponds to a partial factorization. This means the factorization begins with a k-step factorization.
*/

/*
  @param A : square matrix of large size n
  @param startStep : the starting position of the factorization
  @param numVector : the subspace size
  @param v & w & h : local variables of this function
  @return f : the residual vector issued from a k-step factorization
  @return V : the matrix of orthogonal basis issued from a k-step factorization
  @return H : te projected matrix issued from (m3-k)-step Arnoldi factorization
*/
void arnoldiReductionGSM(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::MatrixXd &A, Eigen::MatrixXcd &V,
                         Eigen::MatrixXcd &H, Eigen::VectorXcd &f, int startStep, int numVectors, Eigen::MatrixXd &submatrix, int blockEnd, int numRowsPerBlock, Eigen::VectorXcd &tmp);
void arnoldiReductionGSM_miram(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::MatrixXd &A, Eigen::MatrixXcd &V,
                               Eigen::MatrixXcd &H, Eigen::VectorXcd &fm, int startStep, int numVectors, Eigen::MatrixXd &submatrix, int blockEnd, int numRowsPerBlock, Eigen::VectorXcd &tmp, Eigen::VectorXcd &subspaces_size, Eigen::MatrixXcd &f);
void arnoldiReductionGSM_distrib(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::MatrixXd &submatrix_A, Eigen::MatrixXcd &V,
                                 Eigen::MatrixXcd &H, Eigen::VectorXcd &f, int startStep, int numVectors, Eigen::VectorXcd &tmp, int *v_size, MPI_Comm comm);
void arnoldiReductionGSM_miram_distrib(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::MatrixXd &A, Eigen::MatrixXcd &V,
                                       Eigen::MatrixXcd &H, Eigen::VectorXcd &fm, int startStep, int numVectors, Eigen::VectorXcd &tmp, Eigen::VectorXd &subspaces_size, Eigen::MatrixXcd &f, int *v_size, MPI_Comm comm);
void arnoldiReductionGSM_miram_distrib(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::SparseMatrix<double> &A, Eigen::MatrixXcd &V,
                                       Eigen::MatrixXcd &H, Eigen::VectorXcd &fm, int startStep, int numVectors, Eigen::VectorXcd &tmp, Eigen::VectorXd &subspaces_size, Eigen::MatrixXcd &f, int *v_size, MPI_Comm comm);

/*
  Description :
      This function realize the computation of the Ritz elements
  of matrix A and the min/max of their residual norms.
  The input of the function H is issued fom an Arnoldi
  factorization on matrix A.
*/

/*
  Ritz Algorithm
  @param V : the matrix that contains the m vectors of V matrix
  @param eigenvalues_Hmm : the vector that contains the eigenvalues ​​of Hmm
  @param eigenvectors_Hmm : the matrix that contains the eigenvectors of Hmm
  @param desired_eigenvalues_Hmm : the vector that contains the desired eigenvalues of H
  @nrmfr: norm of matrix A
  @pram k : number of desired eigen elements
  @return  U_desired : the matrix that contains the ritz eigen vectors of the A matrix
  @return desired_eigenvalues_Hmm : the vector that contains the desired eigen values of Hmm
*/
double ritz(Eigen::MatrixXcd &eigenvectors_Hmm, Eigen::VectorXcd &eigenvalues_Hmm, Eigen::MatrixXcd &V, Eigen::MatrixXcd &H, Eigen::VectorXcd &f, int k, double nrmfr,
            Eigen::MatrixXcd &U_desired, Eigen::VectorXcd &residuals, Eigen::VectorXcd &desired_eigenvalues_Hmm, Eigen::MatrixXcd &desired_eigenvectors_Hmm);

/*
  IRAM Algorithm
  @param A : the matrix for which we want to calculate the eigen elements
  @param desired : number of wanted eigen elements
  @param numVectors : the subspace that we want to obtain
  @param nbcycle_max : maximum IRAM iterations
  @param tolerance : error tolerance that must not be exceeded
  @return desired_eigenvalues_Hmm : the vector that contains the eigen values of our A matrix
*/
void sequential_iram(Eigen::MatrixXd &A, double nrmfr, int numVectors, int desired, double tolerance, int nbcycle_max,
                     Eigen::VectorXcd &desired_eigenvalues_Hmm, Eigen::MatrixXcd &U_desired);
void distributed_iram(Eigen::MatrixXd &A, double nrmfr, int numVectors, int desired, double tolerance, int nbcycle_max,
                      Eigen::VectorXcd &desired_eigenvalues_Hmm, Eigen::MatrixXcd &U_desired, int *v_size, MPI_Comm comm);

/*
  Multiple IRAM Nested Subspaces (MIRAM) Algorithm
  @param A : the matrix for which we want to calculate the eigen elements
  @param desired : number of wanted eigen elements
  @param subsspaces_size : the vector that contains the subspaces
  @param nbcycle_max : maximum IRAM iterations
  @param tolerance : error tolerance that must not be exceeded
*/
void sequential_multi_iram(Eigen::MatrixXd &A, int numVectors, int desired, double tolerance,
                           int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr);
void distributed_multi_iram(Eigen::MatrixXd &A, int numVectors, int desired, double tolerance,
                            int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr, int *v_size, MPI_Comm comm);
void distributed_multi_iram_nolog(Eigen::MatrixXd &A, int numVectors, int desired, double tolerance,
                            int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr, int *v_size, MPI_Comm comm);
void distributed_multi_iram(Eigen::SparseMatrix<double> &A, int numVectors, int desired, double tolerance,
                            int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr, int *v_size, MPI_Comm comm);
void distributed_multi_iram_nolog(Eigen::SparseMatrix<double> &A, int numVectors, int desired, double tolerance,
                            int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr, int *v_size, MPI_Comm comm);

// For Modularity function and algorithms
void get_volume_distrib(Eigen::MatrixXd &A, Eigen::VectorXd &volume, MPI_Comm comm);
double get_modularity_distrib(Eigen::MatrixXd &A, Eigen::VectorXd &volume, Eigen::VectorXd &clusters, MPI_Comm comm);
void get_modularity_matrix_distrib(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::VectorXd &volume, int *v_size, MPI_Comm comm);

void build_ATA_distrib(Eigen::MatrixXd &A, Eigen::MatrixXd resA, int *v_size, MPI_Comm comm);
void build_AAT_distrib(Eigen::MatrixXd &A, Eigen::MatrixXd resA, int *v_size, MPI_Comm comm);

#endif
