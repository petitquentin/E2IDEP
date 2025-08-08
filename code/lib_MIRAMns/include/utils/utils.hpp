#ifndef UTILS_HPP
#define UTILS_HPP

#include <proto/protoData.pb.h>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <string>

void print_listVectors(const extractData::listVectorBool listVectors);

void print_matrix_from_master_double(std::string path);
void print_matrix_from_master_int(std::string path);

void getGridDim(const int nbNodes, int *xGrid, int *yGrid);
void print2Ddouble(double *mat, int nb_row, int nb_col);
int createELLPACK(extractData::listVectorBool dataset, int **matrix, MPI_Comm comm);
int createELLPACKfromCOO(std::vector<unsigned long int> row, std::vector<unsigned long int> col, const unsigned long int nb_rows, const unsigned long int nb_cols, int **matrix, MPI_Comm comm);
void createCSRfromCOO(std::vector<unsigned long int> row, std::vector<unsigned long int> col, const unsigned long int nb_rows, std::vector<unsigned long int> *idx, MPI_Comm comm);
double sparsity_double(std::string path);
double sparsity_bool(std::string path);

void mtx_to_dense(bool **v, std::vector<unsigned long int> row, std::vector<unsigned long int> col, const unsigned long int nb_rows, const unsigned long int nb_cols, MPI_Comm comm);
void mtx_to_dense(double **v, std::vector<unsigned long int> row, std::vector<unsigned long int> col, std::vector<double> val, const unsigned long int nb_rows, const unsigned long int nb_cols, MPI_Comm comm);

void build_covariance(double *data, double **output, const unsigned long int nb_rows, const unsigned long int nb_cols, MPI_Comm comm);
void build_covariance_nosecurity(double *data, double **output, const unsigned long int nb_rows, const unsigned long int nb_cols, MPI_Comm comm);

#endif