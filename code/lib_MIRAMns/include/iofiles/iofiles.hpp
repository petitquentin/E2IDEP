#ifndef IOFILES_HPP
#define IOFILES_HPP

#include "proto/protoData.pb.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <mpi.h>

// read
extractData::listVectorBool read_protobuf_message(const std::string path);
void read_mtx(const std::string path, std::vector<unsigned long int> &row, std::vector<unsigned long int> &col, unsigned long int &nb_rows, unsigned long int &nb_cols, unsigned long int &nnz);
void read_mtx(const std::string path, std::vector<unsigned long int> &row, std::vector<unsigned long int> &col, std::vector<double> &val, unsigned long int &nb_rows, unsigned long int &nb_cols, unsigned long int &nnz);
void read_mtx_dense(const std::string path, double **matrix, unsigned long int &nb_rows, unsigned long int &nb_cols);
void read_mtx_dense(const std::string path, double **matrix, unsigned long int &nb_rows, unsigned long int &nb_cols, MPI_Comm comm);
extractData::listVectorDouble read_listDouble(const std::string path);
extractData::listVectorInt read_listInt(const std::string path);
extractData::masterListVectorDouble read_masterListDouble(std::string path);
extractData::masterListVectorInt read_masterListInt(std::string path);
extractData::masterListVectorBool read_masterListBool(std::string path);
void get_mtx_dim(const std::string path, unsigned long int &nb_rows, unsigned long int &nb_cols);

// write
void save_vector_double(double *matrix, const int nbRow, const int nbCol, const std::string path);
void save_vector_int(double *matrix, const int nbRow, const int nbCol, const std::string path);
void save_master_vector_double(double *matrix, const int nbRow, const int nbCol, const std::string path, const int delim = 1000);
void save_master_vector_int(double *matrix, const int nbRow, const int nbCol, const std::string path, const int delim = 1000);
void save_vector_bool(bool *matrix, const int nbRow, const int nbCol, const std::string path);
void save_mtx_dense(Eigen::MatrixXcd A, const std::string path);
void save_mtx_dense(Eigen::MatrixXd A, const std::string path);

void save_mtx_dense(double *data, const unsigned long int nb_rows, const unsigned long int nb_cols, const std::string path);
void save_mtx_dense(double *data, const unsigned long int nb_rows, const unsigned long int nb_cols, const std::string path, MPI_Comm comm);

#endif