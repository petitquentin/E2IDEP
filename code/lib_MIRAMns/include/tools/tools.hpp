#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <eigen3/Eigen/Dense>
#include <Eigen/SparseCore>
#include <complex>
#include <math.h>
#include <algorithms/algorithms.hpp>

/*##################################################################################################################
	This file contains a toolbox allowing to code the algorithms of the algorithms.cpp file.
##################################################################################################################*/

/*
 Function to generate a random vector
 @param size : size of the vector to generate
 @return :  random vector for the given size
*/
Eigen::VectorXd generate_random_vector(int size);

/*
 Function for shifted QR
 @param H_mm : m x m of upper hessenberg matrix
 @param V_m : m first columns of V matrix
 @param eigenvalues_Hmm :eigen values of Hmm matrix
 @param numVectors : nbr of vectors of krylov basis
 @param desired : number of desired eigen elements
 @return R : R matrix from shifted QR
 @return Q : Q matrix from shifted QR
*/
void QR(Eigen::MatrixXcd &H_mm, Eigen::MatrixXcd &V_m, Eigen::VectorXcd &eigenvalues_Hmm, int numVectors, int desired);

/*
 Function for eigen elements computation
 @param H_mm : m x m of upper hessenberg matrix
 @return eigenvalues_Hmm: eigen values of H_mm matrix
 @return eigenvectors_Hmm : eigen vectors of H_mm matrix
*/
void eigs(Eigen::MatrixXcd &H_mm, Eigen::MatrixXcd &eigenvectors_Hmm, Eigen::VectorXcd &eigenvalues_Hmm);

/*
 Function for reading mtx file which containing a matrix
*/
Eigen::MatrixXd read_mtx(std::string dataset_path);
Eigen::MatrixXd read_mtx(std::string dataset_path, unsigned long int startRow, unsigned long int startCol, unsigned long int nb_rows, unsigned long int nb_cols);
Eigen::SparseMatrix<double> read_mtx_sparse(std::string dataset_path, unsigned long int startRow, unsigned long int startCol, unsigned long int nb_rows, unsigned long int nb_cols);

#endif
