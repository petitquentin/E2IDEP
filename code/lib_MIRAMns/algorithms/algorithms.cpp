// #include "../include/algorithms.hpp"
// #include "../include/tools.hpp"

#include <algorithms/algorithms.hpp>
#include <tools/tools.hpp>
#include <bits/stdc++.h>
#include <cassert>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <fstream>
#include <chrono>

/*##################################################################################################################
		Description of each of these function inputs is available in include/algorithms.hpp file.
##################################################################################################################*/

void arnoldiReductionGSM(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::MatrixXd &A, Eigen::MatrixXcd &V,
						 Eigen::MatrixXcd &H, Eigen::VectorXcd &f, int startStep, int numVectors, Eigen::MatrixXd &submatrix, int blockEnd, int numRowsPerBlock, Eigen::VectorXcd &tmp)
{
	int i;

	if (startStep == 1)
	{
		beta = f.norm();
		v = f / beta;
		for (i = 0; i < A.rows(); i = i + numRowsPerBlock) // w = A * v
		{
			blockEnd = std::min(i + numRowsPerBlock - 1, static_cast<int>(A.rows()) - 1);
			submatrix = A.block(i, 0, blockEnd - i + 1, A.cols());
			tmp = submatrix * v;
			w.segment(i, tmp.size()) = tmp;
		}
		alpha = v.adjoint() * w;
		f = w - v * alpha;
		V.col(0) = v;
		H(0, 0) = alpha;
	}

	for (int j = startStep + 1; j <= numVectors; j++)
	{
		beta = f.norm();
		v = f / beta;
		H(j - 1, j - 2) = beta;
		V.col(j - 1) = v;

		for (i = 0; i < A.rows(); i = i + numRowsPerBlock) // w = A * v
		{
			blockEnd = std::min(i + numRowsPerBlock - 1, static_cast<int>(A.rows()) - 1);
			submatrix = A.block(i, 0, blockEnd - i + 1, A.cols());
			tmp = submatrix * v;
			w.segment(i, tmp.size()) = tmp;
		}

		h = V.block(0, 0, V.rows(), j).adjoint() * w;
		f = w - V.block(0, 0, V.rows(), j) * h;
		H.block(0, j - 1, j, 1) = h;
	}
}

void arnoldiReductionGSM_distrib(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::MatrixXd &submatrix_A, Eigen::MatrixXcd &V,
								 Eigen::MatrixXcd &H, Eigen::VectorXcd &f, int startStep, int numVectors, Eigen::VectorXcd &tmp, int *v_size, MPI_Comm comm)
{
	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);
	// int i;

	int *displs = (int *)malloc(p * sizeof(int));
	displs[0] = 0;
	for (int i = 1; i < p; i++)
	{
		displs[i] = displs[i - 1] + v_size[i - 1];
	}

	if (startStep == 1)
	{
		beta = f.norm();
		v = f / beta;

		tmp = submatrix_A * v;
		MPI_Allgatherv(tmp.data(), tmp.size(), MPI_DOUBLE_COMPLEX, w.data(), v_size, displs, MPI_DOUBLE_COMPLEX, comm);

		alpha = v.adjoint() * w;
		f = w - v * alpha;
		V.col(0) = v;
		H(0, 0) = alpha;
	}

	for (int j = startStep + 1; j <= numVectors; j++)
	{
		beta = f.norm();
		v = f / beta;
		H(j - 1, j - 2) = beta;
		V.col(j - 1) = v;

		tmp = submatrix_A * v;
		MPI_Allgatherv(tmp.data(), tmp.size(), MPI_DOUBLE_COMPLEX, w.data(), v_size, displs, MPI_DOUBLE_COMPLEX, comm);

		h = V.block(0, 0, V.rows(), j).adjoint() * w;
		f = w - V.block(0, 0, V.rows(), j) * h;
		H.block(0, j - 1, j, 1) = h;
	}
	free(displs);
}

void arnoldiReductionGSM_miram(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::MatrixXd &A, Eigen::MatrixXcd &V,
							   Eigen::MatrixXcd &H, Eigen::VectorXcd &fm, int startStep, int numVectors, Eigen::MatrixXd &submatrix, int blockEnd, int numRowsPerBlock, Eigen::VectorXcd &tmp, Eigen::VectorXd &subspaces_size,
							   Eigen::MatrixXcd &f)
{
	int i;

	if (startStep == 1)
	{
		beta = fm.norm();
		v = fm / beta;
		for (i = 0; i < A.rows(); i = i + numRowsPerBlock) // w = A * v
		{
			blockEnd = std::min(i + numRowsPerBlock - 1, static_cast<int>(A.rows()) - 1);
			submatrix = A.block(i, 0, blockEnd - i + 1, A.cols());
			tmp = submatrix * v;
			w.segment(i, tmp.size()) = tmp;
		}
		alpha = v.adjoint() * w;
		fm = w - v * alpha;
		V.col(0) = v;
		H(0, 0) = alpha;
	}

	for (int j = startStep + 1; j <= numVectors; j++)
	{
		beta = fm.norm();
		v = fm / beta;
		H(j - 1, j - 2) = beta;
		V.col(j - 1) = v;

		for (i = 0; i < A.rows(); i = i + numRowsPerBlock) // w = A * v
		{
			blockEnd = std::min(i + numRowsPerBlock - 1, static_cast<int>(A.rows()) - 1);
			submatrix = A.block(i, 0, blockEnd - i + 1, A.cols());
			tmp = submatrix * v;
			w.segment(i, tmp.size()) = tmp;
		}

		h = V.block(0, 0, V.rows(), j).adjoint() * w;
		fm = w - V.block(0, 0, V.rows(), j) * h;
		H.block(0, j - 1, j, 1) = h;

		for (i = 0; i < subspaces_size.size(); i++)
		{
			if (j == subspaces_size(i))
			{
				f.block(0, i, fm.rows(), 1) = fm;
			}
		}
	}
}

void arnoldiReductionGSM_miram_distrib(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::MatrixXd &submatrix_A, Eigen::MatrixXcd &V,
									   Eigen::MatrixXcd &H, Eigen::VectorXcd &fm, int startStep, int numVectors, Eigen::VectorXcd &tmp, Eigen::VectorXd &subspaces_size, Eigen::MatrixXcd &f, int *v_size, MPI_Comm comm)
{

	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);

	int *displs = (int *)malloc(p * sizeof(int));
	displs[0] = 0;
	for (int i = 1; i < p; i++)
	{
		displs[i] = displs[i - 1] + v_size[i - 1];
	}

	if (startStep == 1)
	{
		beta = fm.norm();
		v = fm / beta;

		tmp = submatrix_A * v;
		MPI_Allgatherv(tmp.data(), tmp.size(), MPI_DOUBLE_COMPLEX, w.data(), v_size, displs, MPI_DOUBLE_COMPLEX, comm);

		alpha = v.adjoint() * w;
		fm = w - v * alpha;
		V.col(0) = v;
		H(0, 0) = alpha;
	}

	for (int j = startStep + 1; j <= numVectors; j++)
	{
		beta = fm.norm();
		v = fm / beta;
		H(j - 1, j - 2) = beta;
		V.col(j - 1) = v;

		tmp = submatrix_A * v;

		MPI_Allgatherv(tmp.data(), tmp.size(), MPI_DOUBLE_COMPLEX, w.data(), v_size, displs, MPI_DOUBLE_COMPLEX, comm);

		h = V.block(0, 0, V.rows(), j).adjoint() * w;
		fm = w - V.block(0, 0, V.rows(), j) * h;
		H.block(0, j - 1, j, 1) = h;

		for (int i = 0; i < subspaces_size.size(); i++)
		{
			if (j == subspaces_size(i))
			{
				f.block(0, i, fm.rows(), 1) = fm;
			}
		}
	}
	free(displs);
}

void arnoldiReductionGSM_miram_distrib(std::complex<double> beta, std::complex<double> alpha, Eigen::VectorXcd &v, Eigen::VectorXcd &w, Eigen::VectorXcd &h, Eigen::SparseMatrix<double> &submatrix_A, Eigen::MatrixXcd &V,
									   Eigen::MatrixXcd &H, Eigen::VectorXcd &fm, int startStep, int numVectors, Eigen::VectorXcd &tmp, Eigen::VectorXd &subspaces_size, Eigen::MatrixXcd &f, int *v_size, MPI_Comm comm)
{

	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);

	int *displs = (int *)malloc(p * sizeof(int));
	displs[0] = 0;
	for (int i = 1; i < p; i++)
	{
		displs[i] = displs[i - 1] + v_size[i - 1];
	}

	if (startStep == 1)
	{
		beta = fm.norm();
		v = fm / beta;

		tmp = submatrix_A * v;
		MPI_Allgatherv(tmp.data(), tmp.size(), MPI_DOUBLE_COMPLEX, w.data(), v_size, displs, MPI_DOUBLE_COMPLEX, comm);

		alpha = v.adjoint() * w;
		fm = w - v * alpha;
		V.col(0) = v;
		H(0, 0) = alpha;
	}

	for (int j = startStep + 1; j <= numVectors; j++)
	{
		beta = fm.norm();
		v = fm / beta;
		H(j - 1, j - 2) = beta;
		V.col(j - 1) = v;

		tmp = submatrix_A * v;

		MPI_Allgatherv(tmp.data(), tmp.size(), MPI_DOUBLE_COMPLEX, w.data(), v_size, displs, MPI_DOUBLE_COMPLEX, comm);

		h = V.block(0, 0, V.rows(), j).adjoint() * w;
		fm = w - V.block(0, 0, V.rows(), j) * h;
		H.block(0, j - 1, j, 1) = h;

		for (int i = 0; i < subspaces_size.size(); i++)
		{
			if (j == subspaces_size(i))
			{
				f.block(0, i, fm.rows(), 1) = fm;
			}
		}
	}
	free(displs);
}

double ritz(Eigen::MatrixXcd &eigenvectors_Hmm, Eigen::VectorXcd &eigenvalues_Hmm, Eigen::MatrixXcd &V, Eigen::MatrixXcd &H, Eigen::VectorXcd &f, int k, double nrmfr,
			Eigen::MatrixXcd &U_desired, Eigen::VectorXcd &residuals, Eigen::VectorXcd &desired_eigenvalues_Hmm, Eigen::MatrixXcd &desired_eigenvectors_Hmm)
{
	int i, j;

	// we select first biggest eigen values and their eigen vectors ("LM").
	// Note: Eigen library sorts eigen values (and their eigen vectors) in ascending order according to their modulus.
	j = 0;

	for (i = eigenvalues_Hmm.size() - 1; i > eigenvalues_Hmm.size() - 1 - k; i--)
	{
		desired_eigenvalues_Hmm(j) = eigenvalues_Hmm(i);
		desired_eigenvectors_Hmm.col(j) = eigenvectors_Hmm.col(i);
		j++;
	}

	// desired eigen vectors
	U_desired = V * desired_eigenvectors_Hmm;

	// we compute ritz estimate
	for (i = 0; i < k; i++)
	{
		residuals(i) = f.norm() * abs(desired_eigenvectors_Hmm(desired_eigenvectors_Hmm.rows() - 1, i));
	}

	return residuals.norm() / nrmfr;
}

void sequential_iram(Eigen::MatrixXd &A, double nrmfr, int numVectors, int desired, double tolerance,
					 int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired)
{
	double residuals_norm = 0.0;

	// VARIABLES FOR ARNOLDI FUNCTION:
	Eigen::MatrixXcd H_mm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd V_m = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	Eigen::MatrixXd submatrix;
	int blockEnd = 0;
	int numRowsPerBlock = 2;
	Eigen::VectorXcd tmp(A.cols());
	std::complex<double> beta, alpha;
	Eigen::VectorXcd v = Eigen::VectorXcd::Zero(A.cols());
	Eigen::VectorXcd w = Eigen::VectorXcd::Zero(v.size());
	Eigen::VectorXcd h = Eigen::VectorXcd::Zero(v.size());

	// VARIABLES FOR RITZ FUNCTION:
	Eigen::VectorXcd f = Eigen::VectorXcd::Zero(A.cols());
	Eigen::VectorXcd residuals(desired);
	Eigen::MatrixXcd desired_eigenvectors_Hmm(H_mm.rows(), desired);
	Eigen::MatrixXcd eigenvectors_Hmm(numVectors, numVectors);
	Eigen::VectorXcd eigenvalues_Hmm(numVectors);

	// VARIABLE : used to compute Vm
	Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(numVectors, numVectors);

	int it;

	for (it = 0; it < A.cols(); it++)
	{
		f(it) = {1 / sqrt(A.cols()), 0};
	}

	// k step Arnoldi factorization
	arnoldiReductionGSM(beta, alpha, v, w, h, A, V_m, H_mm, f, 1, numVectors, submatrix, blockEnd, numRowsPerBlock, tmp);
	// arnoldiReductionGSM_distrib(beta, alpha, v, w, h, A, V_m, H_mm, f, 1, numVectors,  tmp);

	for (it = 0; it < nbcycle_max; it++)
	{
		// we compute eigen elements of H_m,m (eigen values and eigen vectors)
		eigs(H_mm, eigenvectors_Hmm, eigenvalues_Hmm);

		// we compute Ritz peers
		residuals_norm = ritz(eigenvectors_Hmm, eigenvalues_Hmm, V_m, H_mm, f, desired, nrmfr, U_desired,
							  residuals, desired_eigenvalues, desired_eigenvectors_Hmm);

		std::cout << "Iteration : " << it + 1 << " , Residuals : " << residuals_norm << std::endl;

		// tolerance
		if (residuals_norm < tolerance)
		{
			break;
		}

		Q = Eigen::MatrixXcd::Identity(numVectors, numVectors);

		// shifted QR with the unwanted eigenvalues (eigen values from Eigen library are sorted in ascending order by their magnitude)
		QR(H_mm, Q, eigenvalues_Hmm, numVectors, desired);

		// residual vector
		f = V_m.block(0, 0, V_m.rows(), numVectors) * Q.block(0, desired, Q.rows(), 1) * H_mm(desired, desired - 1) + f * Q(numVectors - 1, desired - 1);

		// we take only first desired cols
		V_m.block(0, 0, V_m.rows(), desired) = V_m.block(0, 0, V_m.rows(), numVectors) * Q.block(0, 0, Q.rows(), desired);

		// arnoldi extension
		arnoldiReductionGSM(beta, alpha, v, w, h, A, V_m, H_mm, f, desired, numVectors, submatrix, blockEnd, numRowsPerBlock, tmp);
		// arnoldiReductionGSM_distrib(beta, alpha, v, w, h, A, V_m, H_mm, f, desired, numVectors, tmp);
	}
}

void distributed_iram(Eigen::MatrixXd &A, double nrmfr, int numVectors, int desired, double tolerance,
					  int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, int *v_size, MPI_Comm comm)
{
	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);
	double residuals_norm = 0.0;

	// VARIABLES FOR ARNOLDI FUNCTION:
	Eigen::MatrixXcd H_mm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd V_m = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	Eigen::MatrixXd submatrix;
	// int blockEnd;
	// int numRowsPerBlock = 2;
	Eigen::VectorXcd tmp(A.cols());
	std::complex<double> beta, alpha;
	Eigen::VectorXcd v = Eigen::VectorXcd::Zero(A.cols());
	Eigen::VectorXcd w = Eigen::VectorXcd::Zero(v.size());
	Eigen::VectorXcd h = Eigen::VectorXcd::Zero(v.size());

	// VARIABLES FOR RITZ FUNCTION:
	Eigen::VectorXcd f = Eigen::VectorXcd::Zero(A.cols());
	Eigen::VectorXcd residuals(desired);
	Eigen::MatrixXcd desired_eigenvectors_Hmm(H_mm.rows(), desired);
	Eigen::MatrixXcd eigenvectors_Hmm(numVectors, numVectors);
	Eigen::VectorXcd eigenvalues_Hmm(numVectors);

	// VARIABLE : used to compute Vm
	Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(numVectors, numVectors);

	int it;

	for (it = 0; it < A.cols(); it++)
	{
		f(it) = {1 / sqrt(A.cols()), 0};
	}

	// k step Arnoldi factorization
	arnoldiReductionGSM_distrib(beta, alpha, v, w, h, A, V_m, H_mm, f, 1, numVectors, tmp, v_size, comm);

	for (it = 0; it < nbcycle_max; it++)
	{
		// we compute eigen elements of H_m,m (eigen values and eigen vectors)
		eigs(H_mm, eigenvectors_Hmm, eigenvalues_Hmm);

		// we compute Ritz peers
		residuals_norm = ritz(eigenvectors_Hmm, eigenvalues_Hmm, V_m, H_mm, f, desired, nrmfr, U_desired,
							  residuals, desired_eigenvalues, desired_eigenvectors_Hmm);

		if (rank == 0)
			std::cout << "Iteration : " << it + 1 << " , Residuals : " << residuals_norm << std::endl;

		// tolerance
		if (residuals_norm < tolerance)
		{
			break;
		}

		Q = Eigen::MatrixXcd::Identity(numVectors, numVectors);

		// shifted QR with the unwanted eigenvalues (eigen values from Eigen library are sorted in ascending order by their magnitude)
		QR(H_mm, Q, eigenvalues_Hmm, numVectors, desired);

		// residual vector
		f = V_m.block(0, 0, V_m.rows(), numVectors) * Q.block(0, desired, Q.rows(), 1) * H_mm(desired, desired - 1) + f * Q(numVectors - 1, desired - 1);

		// we take only first desired cols
		V_m.block(0, 0, V_m.rows(), desired) = V_m.block(0, 0, V_m.rows(), numVectors) * Q.block(0, 0, Q.rows(), desired);

		// arnoldi extension
		arnoldiReductionGSM_distrib(beta, alpha, v, w, h, A, V_m, H_mm, f, desired, numVectors, tmp, v_size, comm);
	}
}

void sequential_multi_iram(Eigen::MatrixXd &A, int numVectors, int desired, double tolerance,
						   int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr)
{
	double residuals_norm = 0.0;

	// VARIABLES FOR ARNOLDI FUNCTION:
	Eigen::MatrixXcd H_mm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd V_m = Eigen::MatrixXcd::Zero(A.rows(), numVectors);
	Eigen::MatrixXd submatrix;
	int blockEnd = 0;
	int numRowsPerBlock = 2;
	Eigen::VectorXcd tmp(A.cols());
	std::complex<double> beta, alpha;
	Eigen::VectorXcd v = Eigen::VectorXcd::Zero(A.cols());
	Eigen::VectorXcd w = Eigen::VectorXcd::Zero(v.size());
	Eigen::VectorXcd h = Eigen::VectorXcd::Zero(v.size());

	// VARIABLES FOR RITZ FUNCTION:
	Eigen::VectorXcd fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd f = Eigen::MatrixXcd::Zero(fm.rows(), subspaces_size.size());
	Eigen::VectorXcd residuals(desired);
	Eigen::MatrixXcd desired_eigenvectors_Hmm(H_mm.rows(), desired);
	Eigen::MatrixXcd eigenvectors_Hmm(numVectors, numVectors);
	Eigen::VectorXcd eigenvalues_Hmm(numVectors);

	// VARIABLE : used to compute Vm
	Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(numVectors, numVectors);

	// VARIABLES FOR MIRAM
	int it;
	// int m = 0;
	int i;
	double selected_residuals_norm = 0.0;
	Eigen::VectorXcd selected_fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd selected_Hmm;
	Eigen::MatrixXcd selected_Vm;
	Eigen::MatrixXcd current_Hmm;
	Eigen::MatrixXcd current_Vm;

	for (it = 0; it < A.cols(); it++)
	{
		fm(it) = {1 / nrmfr, 0};
	}

	// k step Arnoldi factorization
	arnoldiReductionGSM_miram(beta, alpha, v, w, h, A, V_m, H_mm, fm, 1, numVectors, submatrix, blockEnd, numRowsPerBlock, tmp, subspaces_size, f);

	for (it = 0; it < nbcycle_max; it++)
	{
		selected_residuals_norm = 0.0;
		for (i = subspaces_size.size() - 1; i >= 0; i--)
		{
			current_Hmm = H_mm.block(0, 0, subspaces_size(i), subspaces_size(i));
			current_Vm = V_m.block(0, 0, V_m.rows(), subspaces_size(i));
			eigenvectors_Hmm.resize(subspaces_size(i), subspaces_size(i));
			eigenvalues_Hmm.resize(subspaces_size(i));
			desired_eigenvectors_Hmm.resize(subspaces_size(i), desired);
			eigs(current_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);
			fm = f.block(0, i, f.rows(), 1);
			residuals_norm = ritz(eigenvectors_Hmm, eigenvalues_Hmm, current_Vm, current_Hmm, fm, desired, nrmfr, U_desired,
								  residuals, desired_eigenvalues, desired_eigenvectors_Hmm);

			if (selected_residuals_norm == 0.0 || selected_residuals_norm > residuals_norm)
			{
				numVectors = subspaces_size(i);
				selected_residuals_norm = residuals_norm;
				selected_fm = fm;
				selected_Hmm = current_Hmm;
				selected_Vm = current_Vm;
			}
		}

		eigenvectors_Hmm.resize(numVectors, numVectors);
		eigenvalues_Hmm.resize(numVectors);

		eigs(selected_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);

		std::cout << "Iteration : " << it + 1 << " , Residuals : " << selected_residuals_norm << " , Numvectors : " << numVectors << std::endl;

		if (selected_residuals_norm < tolerance)
		{
			break;
		}

		Q.resize(numVectors, numVectors);
		Q = Eigen::MatrixXcd::Identity(numVectors, numVectors);

		QR(selected_Hmm, Q, eigenvalues_Hmm, numVectors, desired);

		selected_fm = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, desired, Q.rows(), 1) * selected_Hmm(desired, desired - 1) + selected_fm * Q(numVectors - 1, desired - 1);

		selected_Vm.block(0, 0, V_m.rows(), desired) = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, 0, Q.rows(), desired);

		H_mm.block(0, 0, selected_Hmm.rows(), selected_Hmm.cols()) = selected_Hmm;
		V_m.block(0, 0, selected_Vm.rows(), selected_Vm.cols()) = selected_Vm;

		// arnoldi extension
		arnoldiReductionGSM_miram(beta, alpha, v, w, h, A, V_m, H_mm, selected_fm, desired, subspaces_size(subspaces_size.size() - 1), submatrix, blockEnd, numRowsPerBlock, tmp, subspaces_size, f);
	}
}

void distributed_multi_iram(Eigen::MatrixXd &A, int numVectors, int desired, double tolerance,
							int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr, int *v_size, MPI_Comm comm)
{
	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);

	double residuals_norm = 0.0;
	bool is_selected = false;

	// VARIABLES FOR ARNOLDI FUNCTION:
	Eigen::MatrixXcd H_mm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd V_m = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	std::complex<double> beta, alpha;
	Eigen::VectorXcd v = Eigen::VectorXcd::Zero(A.cols());
	Eigen::VectorXcd w = Eigen::VectorXcd::Zero(v.size());
	Eigen::VectorXcd h = Eigen::VectorXcd::Zero(v.size());

	// VARIABLES FOR RITZ FUNCTION:
	Eigen::VectorXcd fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd f = Eigen::MatrixXcd::Zero(fm.rows(), subspaces_size.size());
	Eigen::VectorXcd residuals(desired);
	Eigen::MatrixXcd desired_eigenvectors_Hmm(H_mm.rows(), desired);
	Eigen::MatrixXcd eigenvectors_Hmm(numVectors, numVectors);
	Eigen::VectorXcd eigenvalues_Hmm(numVectors);
	Eigen::VectorXcd tmp(A.cols());

	// VARIABLE : used to compute Vm
	Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(numVectors, numVectors);

	// VARIABLES FOR MIRAM
	int it;
	// int m = 0;
	int i;
	double selected_residuals_norm = 0.0;
	Eigen::VectorXcd selected_fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd selected_Hmm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd selected_Vm = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	Eigen::MatrixXcd current_Hmm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd current_Vm = Eigen::MatrixXcd::Zero(A.cols(), numVectors);

	for (it = 0; it < A.cols(); it++)
	{
		fm(it) = {1 / sqrt(A.cols()), 0};
	}

	// k step Arnoldi factorization

	arnoldiReductionGSM_miram_distrib(beta, alpha, v, w, h, A, V_m, H_mm, fm, 1, numVectors, tmp, subspaces_size, f, v_size, comm);

	for (it = 0; it < nbcycle_max; it++)
	{
		selected_residuals_norm = 0.0;
		is_selected = false;
		for (i = subspaces_size.size() - 1; i >= 0; i--)
		{
			current_Hmm.resize(subspaces_size(i), subspaces_size(i));
			current_Vm.resize(V_m.rows(), subspaces_size(i));
			current_Hmm = H_mm.block(0, 0, subspaces_size(i), subspaces_size(i));
			current_Vm = V_m.block(0, 0, V_m.rows(), subspaces_size(i));
			eigenvectors_Hmm.resize(subspaces_size(i), subspaces_size(i));
			eigenvalues_Hmm.resize(subspaces_size(i));
			desired_eigenvectors_Hmm.resize(subspaces_size(i), desired);
			eigs(current_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);
			fm = f.block(0, i, f.rows(), 1);

			residuals_norm = ritz(eigenvectors_Hmm, eigenvalues_Hmm, current_Vm, current_Hmm, fm, desired, nrmfr, U_desired,
								  residuals, desired_eigenvalues, desired_eigenvectors_Hmm);

			if ((selected_residuals_norm == 0.0 && is_selected == false) || selected_residuals_norm >= residuals_norm)
			{
				is_selected = true;
				numVectors = subspaces_size(i);
				selected_Hmm.resize(numVectors, numVectors);
				selected_Vm.resize(V_m.rows(), numVectors);
				selected_residuals_norm = residuals_norm;
				selected_fm = fm;
				selected_Hmm = current_Hmm;
				selected_Vm = current_Vm;
			}
		}

		eigenvectors_Hmm.resize(numVectors, numVectors);
		eigenvalues_Hmm.resize(numVectors);

		eigs(selected_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);

		if (rank == 0)
			std::cout << "Iteration : " << it + 1 << " , Residuals : " << selected_residuals_norm << " , Numvectors : " << numVectors << std::endl;

		if (selected_residuals_norm < tolerance)
		{
			break;
		}

		Q.resize(numVectors, numVectors);
		Q = Eigen::MatrixXcd::Identity(numVectors, numVectors);

		QR(selected_Hmm, Q, eigenvalues_Hmm, numVectors, desired);

		selected_fm = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, desired, Q.rows(), 1) * selected_Hmm(desired, desired - 1) + selected_fm * Q(numVectors - 1, desired - 1);

		selected_Vm.block(0, 0, V_m.rows(), desired) = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, 0, Q.rows(), desired);

		H_mm.block(0, 0, selected_Hmm.rows(), selected_Hmm.cols()) = selected_Hmm;
		V_m.block(0, 0, selected_Vm.rows(), selected_Vm.cols()) = selected_Vm;

		// arnoldi extension
		arnoldiReductionGSM_miram_distrib(beta, alpha, v, w, h, A, V_m, H_mm, selected_fm, desired, subspaces_size(subspaces_size.size() - 1), tmp, subspaces_size, f, v_size, comm);
	}
}

void distributed_multi_iram_nolog(Eigen::MatrixXd &A, int numVectors, int desired, double tolerance,
							int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr, int *v_size, MPI_Comm comm)
{
	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);

	double residuals_norm = 0.0;
	bool is_selected = false;

	// VARIABLES FOR ARNOLDI FUNCTION:
	Eigen::MatrixXcd H_mm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd V_m = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	std::complex<double> beta, alpha;
	Eigen::VectorXcd v = Eigen::VectorXcd::Zero(A.cols());
	Eigen::VectorXcd w = Eigen::VectorXcd::Zero(v.size());
	Eigen::VectorXcd h = Eigen::VectorXcd::Zero(v.size());

	// VARIABLES FOR RITZ FUNCTION:
	Eigen::VectorXcd fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd f = Eigen::MatrixXcd::Zero(fm.rows(), subspaces_size.size());
	Eigen::VectorXcd residuals(desired);
	Eigen::MatrixXcd desired_eigenvectors_Hmm(H_mm.rows(), desired);
	Eigen::MatrixXcd eigenvectors_Hmm(numVectors, numVectors);
	Eigen::VectorXcd eigenvalues_Hmm(numVectors);
	Eigen::VectorXcd tmp(A.cols());

	// VARIABLE : used to compute Vm
	Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(numVectors, numVectors);

	// VARIABLES FOR MIRAM
	int it;
	// int m = 0;
	int i;
	double selected_residuals_norm = 0.0;
	Eigen::VectorXcd selected_fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd selected_Hmm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd selected_Vm = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	Eigen::MatrixXcd current_Hmm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd current_Vm = Eigen::MatrixXcd::Zero(A.cols(), numVectors);

	for (it = 0; it < A.cols(); it++)
	{
		fm(it) = {1 / sqrt(A.cols()), 0};
	}

	// k step Arnoldi factorization

	arnoldiReductionGSM_miram_distrib(beta, alpha, v, w, h, A, V_m, H_mm, fm, 1, numVectors, tmp, subspaces_size, f, v_size, comm);

	for (it = 0; it < nbcycle_max; it++)
	{
		selected_residuals_norm = 0.0;
		is_selected = false;
		for (i = subspaces_size.size() - 1; i >= 0; i--)
		{
			current_Hmm.resize(subspaces_size(i), subspaces_size(i));
			current_Vm.resize(V_m.rows(), subspaces_size(i));
			current_Hmm = H_mm.block(0, 0, subspaces_size(i), subspaces_size(i));
			current_Vm = V_m.block(0, 0, V_m.rows(), subspaces_size(i));
			eigenvectors_Hmm.resize(subspaces_size(i), subspaces_size(i));
			eigenvalues_Hmm.resize(subspaces_size(i));
			desired_eigenvectors_Hmm.resize(subspaces_size(i), desired);
			eigs(current_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);
			fm = f.block(0, i, f.rows(), 1);

			residuals_norm = ritz(eigenvectors_Hmm, eigenvalues_Hmm, current_Vm, current_Hmm, fm, desired, nrmfr, U_desired,
								  residuals, desired_eigenvalues, desired_eigenvectors_Hmm);

			if ((selected_residuals_norm == 0.0 && is_selected == false) || selected_residuals_norm >= residuals_norm)
			{
				is_selected = true;
				numVectors = subspaces_size(i);
				selected_Hmm.resize(numVectors, numVectors);
				selected_Vm.resize(V_m.rows(), numVectors);
				selected_residuals_norm = residuals_norm;
				selected_fm = fm;
				selected_Hmm = current_Hmm;
				selected_Vm = current_Vm;
			}
		}

		eigenvectors_Hmm.resize(numVectors, numVectors);
		eigenvalues_Hmm.resize(numVectors);

		eigs(selected_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);

		if (selected_residuals_norm < tolerance)
		{
			break;
		}

		Q.resize(numVectors, numVectors);
		Q = Eigen::MatrixXcd::Identity(numVectors, numVectors);

		QR(selected_Hmm, Q, eigenvalues_Hmm, numVectors, desired);

		selected_fm = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, desired, Q.rows(), 1) * selected_Hmm(desired, desired - 1) + selected_fm * Q(numVectors - 1, desired - 1);

		selected_Vm.block(0, 0, V_m.rows(), desired) = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, 0, Q.rows(), desired);

		H_mm.block(0, 0, selected_Hmm.rows(), selected_Hmm.cols()) = selected_Hmm;
		V_m.block(0, 0, selected_Vm.rows(), selected_Vm.cols()) = selected_Vm;

		// arnoldi extension
		arnoldiReductionGSM_miram_distrib(beta, alpha, v, w, h, A, V_m, H_mm, selected_fm, desired, subspaces_size(subspaces_size.size() - 1), tmp, subspaces_size, f, v_size, comm);
	}
}


void distributed_multi_iram(Eigen::SparseMatrix<double> &A, int numVectors, int desired, double tolerance,
							int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr, int *v_size, MPI_Comm comm)
{
	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);

	double residuals_norm = 0.0;
	bool is_selected = false;

	// VARIABLES FOR ARNOLDI FUNCTION:
	Eigen::MatrixXcd H_mm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd V_m = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	std::complex<double> beta, alpha;
	Eigen::VectorXcd v = Eigen::VectorXcd::Zero(A.cols());
	Eigen::VectorXcd w = Eigen::VectorXcd::Zero(v.size());
	Eigen::VectorXcd h = Eigen::VectorXcd::Zero(v.size());

	// VARIABLES FOR RITZ FUNCTION:
	Eigen::VectorXcd fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd f = Eigen::MatrixXcd::Zero(fm.rows(), subspaces_size.size());
	Eigen::VectorXcd residuals(desired);
	Eigen::MatrixXcd desired_eigenvectors_Hmm(H_mm.rows(), desired);
	Eigen::MatrixXcd eigenvectors_Hmm(numVectors, numVectors);
	Eigen::VectorXcd eigenvalues_Hmm(numVectors);
	Eigen::VectorXcd tmp(A.cols());

	// VARIABLE : used to compute Vm
	Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(numVectors, numVectors);

	// VARIABLES FOR MIRAM
	int it;
	// int m = 0;
	int i;
	double selected_residuals_norm = 0.0;
	Eigen::VectorXcd selected_fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd selected_Hmm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd selected_Vm = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	Eigen::MatrixXcd current_Hmm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd current_Vm = Eigen::MatrixXcd::Zero(A.cols(), numVectors);

	for (it = 0; it < A.cols(); it++)
	{
		fm(it) = {1 / sqrt(A.cols()), 0};
	}

	// k step Arnoldi factorization

	arnoldiReductionGSM_miram_distrib(beta, alpha, v, w, h, A, V_m, H_mm, fm, 1, numVectors, tmp, subspaces_size, f, v_size, comm);

	for (it = 0; it < nbcycle_max; it++)
	{
		selected_residuals_norm = 0.0;
		is_selected = false;
		for (i = subspaces_size.size() - 1; i >= 0; i--)
		{
			current_Hmm.resize(subspaces_size(i), subspaces_size(i));
			current_Vm.resize(V_m.rows(), subspaces_size(i));
			current_Hmm = H_mm.block(0, 0, subspaces_size(i), subspaces_size(i));
			current_Vm = V_m.block(0, 0, V_m.rows(), subspaces_size(i));
			eigenvectors_Hmm.resize(subspaces_size(i), subspaces_size(i));
			eigenvalues_Hmm.resize(subspaces_size(i));
			desired_eigenvectors_Hmm.resize(subspaces_size(i), desired);
			eigs(current_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);
			fm = f.block(0, i, f.rows(), 1);

			residuals_norm = ritz(eigenvectors_Hmm, eigenvalues_Hmm, current_Vm, current_Hmm, fm, desired, nrmfr, U_desired,
								  residuals, desired_eigenvalues, desired_eigenvectors_Hmm);

			if ((selected_residuals_norm == 0.0 && is_selected == false) || selected_residuals_norm >= residuals_norm)
			{
				is_selected = true;
				numVectors = subspaces_size(i);
				selected_Hmm.resize(numVectors, numVectors);
				selected_Vm.resize(V_m.rows(), numVectors);
				selected_residuals_norm = residuals_norm;
				selected_fm = fm;
				selected_Hmm = current_Hmm;
				selected_Vm = current_Vm;
			}
		}

		eigenvectors_Hmm.resize(numVectors, numVectors);
		eigenvalues_Hmm.resize(numVectors);

		eigs(selected_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);

		if (rank == 0)
			std::cout << "Iteration : " << it + 1 << " , Residuals : " << selected_residuals_norm << " , Numvectors : " << numVectors << std::endl;

		if (selected_residuals_norm < tolerance)
		{
			break;
		}

		Q.resize(numVectors, numVectors);
		Q = Eigen::MatrixXcd::Identity(numVectors, numVectors);

		QR(selected_Hmm, Q, eigenvalues_Hmm, numVectors, desired);

		selected_fm = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, desired, Q.rows(), 1) * selected_Hmm(desired, desired - 1) + selected_fm * Q(numVectors - 1, desired - 1);

		selected_Vm.block(0, 0, V_m.rows(), desired) = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, 0, Q.rows(), desired);

		H_mm.block(0, 0, selected_Hmm.rows(), selected_Hmm.cols()) = selected_Hmm;
		V_m.block(0, 0, selected_Vm.rows(), selected_Vm.cols()) = selected_Vm;

		// arnoldi extension
		arnoldiReductionGSM_miram_distrib(beta, alpha, v, w, h, A, V_m, H_mm, selected_fm, desired, subspaces_size(subspaces_size.size() - 1), tmp, subspaces_size, f, v_size, comm);
	}
}

void distributed_multi_iram_nolog(Eigen::SparseMatrix<double> &A, int numVectors, int desired, double tolerance,
							int nbcycle_max, Eigen::VectorXcd &desired_eigenvalues, Eigen::MatrixXcd &U_desired, Eigen::VectorXd &subspaces_size, double nrmfr, int *v_size, MPI_Comm comm)
{
	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);

	double residuals_norm = 0.0;
	bool is_selected = false;

	// VARIABLES FOR ARNOLDI FUNCTION:
	Eigen::MatrixXcd H_mm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd V_m = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	std::complex<double> beta, alpha;
	Eigen::VectorXcd v = Eigen::VectorXcd::Zero(A.cols());
	Eigen::VectorXcd w = Eigen::VectorXcd::Zero(v.size());
	Eigen::VectorXcd h = Eigen::VectorXcd::Zero(v.size());

	// VARIABLES FOR RITZ FUNCTION:
	Eigen::VectorXcd fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd f = Eigen::MatrixXcd::Zero(fm.rows(), subspaces_size.size());
	Eigen::VectorXcd residuals(desired);
	Eigen::MatrixXcd desired_eigenvectors_Hmm(H_mm.rows(), desired);
	Eigen::MatrixXcd eigenvectors_Hmm(numVectors, numVectors);
	Eigen::VectorXcd eigenvalues_Hmm(numVectors);
	Eigen::VectorXcd tmp(A.cols());

	// VARIABLE : used to compute Vm
	Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(numVectors, numVectors);

	// VARIABLES FOR MIRAM
	int it;
	// int m = 0;
	int i;
	double selected_residuals_norm = 0.0;
	Eigen::VectorXcd selected_fm = Eigen::VectorXcd::Zero(A.cols());
	Eigen::MatrixXcd selected_Hmm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd selected_Vm = Eigen::MatrixXcd::Zero(A.cols(), numVectors);
	Eigen::MatrixXcd current_Hmm = Eigen::MatrixXcd::Zero(numVectors, numVectors);
	Eigen::MatrixXcd current_Vm = Eigen::MatrixXcd::Zero(A.cols(), numVectors);

	for (it = 0; it < A.cols(); it++)
	{
		fm(it) = {1 / sqrt(A.cols()), 0};
	}

	// k step Arnoldi factorization

	arnoldiReductionGSM_miram_distrib(beta, alpha, v, w, h, A, V_m, H_mm, fm, 1, numVectors, tmp, subspaces_size, f, v_size, comm);

	for (it = 0; it < nbcycle_max; it++)
	{
		selected_residuals_norm = 0.0;
		is_selected = false;
		for (i = subspaces_size.size() - 1; i >= 0; i--)
		{
			current_Hmm.resize(subspaces_size(i), subspaces_size(i));
			current_Vm.resize(V_m.rows(), subspaces_size(i));
			current_Hmm = H_mm.block(0, 0, subspaces_size(i), subspaces_size(i));
			current_Vm = V_m.block(0, 0, V_m.rows(), subspaces_size(i));
			eigenvectors_Hmm.resize(subspaces_size(i), subspaces_size(i));
			eigenvalues_Hmm.resize(subspaces_size(i));
			desired_eigenvectors_Hmm.resize(subspaces_size(i), desired);
			eigs(current_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);
			fm = f.block(0, i, f.rows(), 1);

			residuals_norm = ritz(eigenvectors_Hmm, eigenvalues_Hmm, current_Vm, current_Hmm, fm, desired, nrmfr, U_desired,
								  residuals, desired_eigenvalues, desired_eigenvectors_Hmm);

			if ((selected_residuals_norm == 0.0 && is_selected == false) || selected_residuals_norm >= residuals_norm)
			{
				is_selected = true;
				numVectors = subspaces_size(i);
				selected_Hmm.resize(numVectors, numVectors);
				selected_Vm.resize(V_m.rows(), numVectors);
				selected_residuals_norm = residuals_norm;
				selected_fm = fm;
				selected_Hmm = current_Hmm;
				selected_Vm = current_Vm;
			}
		}

		eigenvectors_Hmm.resize(numVectors, numVectors);
		eigenvalues_Hmm.resize(numVectors);

		eigs(selected_Hmm, eigenvectors_Hmm, eigenvalues_Hmm);

		if (selected_residuals_norm < tolerance)
		{
			break;
		}

		Q.resize(numVectors, numVectors);
		Q = Eigen::MatrixXcd::Identity(numVectors, numVectors);

		QR(selected_Hmm, Q, eigenvalues_Hmm, numVectors, desired);

		selected_fm = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, desired, Q.rows(), 1) * selected_Hmm(desired, desired - 1) + selected_fm * Q(numVectors - 1, desired - 1);

		selected_Vm.block(0, 0, V_m.rows(), desired) = selected_Vm.block(0, 0, V_m.rows(), numVectors) * Q.block(0, 0, Q.rows(), desired);

		H_mm.block(0, 0, selected_Hmm.rows(), selected_Hmm.cols()) = selected_Hmm;
		V_m.block(0, 0, selected_Vm.rows(), selected_Vm.cols()) = selected_Vm;

		// arnoldi extension
		arnoldiReductionGSM_miram_distrib(beta, alpha, v, w, h, A, V_m, H_mm, selected_fm, desired, subspaces_size(subspaces_size.size() - 1), tmp, subspaces_size, f, v_size, comm);
	}
}

void get_volume_distrib(Eigen::MatrixXd &A, Eigen::VectorXd &volume, MPI_Comm comm)
{
	volume = Eigen::VectorXd::Zero(A.cols());
	Eigen::VectorXd tmp = Eigen::VectorXd::Zero(A.cols());

	for (int i = 0; i < A.cols(); i++)
	{
		for (int j = 0; j < A.rows(); j++)
		{
			tmp[i] += A(j, i);
		}
	}
	MPI_Allreduce(tmp.data(), volume.data(), tmp.size(), MPI_DOUBLE, MPI_SUM, comm);
}

double get_modularity_distrib(Eigen::MatrixXd &A, Eigen::VectorXd &volume, Eigen::VectorXd &clusters, MPI_Comm comm)
{

	double mod = 0;
	double sum = 0;
	for (int i = 0; i < A.rows(); i++)
	{
		sum += volume[i];
	}
	for (int i = 0; i < A.rows(); i++)
	{
		for (int j = 0; j < A.cols(); j++)
		{
			if (clusters[i] == clusters[j])
			{
				mod += A(i, j) - (volume[i] * volume[j] / sum);
			}
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &mod, 1, MPI_DOUBLE, MPI_SUM, comm);
	mod = mod / sum;
	return mod;
}

void get_modularity_matrix_distrib(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::VectorXd &volume, int *v_size, MPI_Comm comm)
{
	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);

	B = Eigen::MatrixXd::Zero(A.rows(), A.cols());

	int *displs = (int *)malloc(p * sizeof(int));
	displs[0] = 0;
	for (int i = 1; i < p; i++)
	{
		displs[i] = displs[i - 1] + v_size[i - 1];
	}
	double sum = 0;
	for (int i = 0; i < A.rows(); i++)
	{
		sum += volume[i];
	}

	for (int i = 0; i < A.rows(); i++)
	{
		for (int j = 0; j < A.cols(); j++)
		{
			B(i, j) = A(i, j) - (volume[j] * volume[i + displs[rank]] / sum);
		}
	}

	free(displs);
}

void build_AAT_distrib(Eigen::MatrixXd &A, Eigen::MatrixXd resA, int *v_size, MPI_Comm comm)
{
	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);

	int *displs = (int *)malloc(p * sizeof(int));
	displs[0] = 0;
	for (int i = 1; i < p; i++)
	{
		displs[i] = displs[i - 1] + v_size[i - 1];
	}
	int nb_rows = displs[p - 1] + v_size[p - 1];

	resA = Eigen::MatrixXd::Zero(A.rows(), nb_rows);
	// Build local
	Eigen::MatrixXd tmp = A * A.transpose();

	for (int x = 0; x < v_size[rank]; x++)
	{
		for (int y = 0; y < v_size[rank]; y++)
		{
			resA(x, y + displs[rank]) = tmp(x, y);
		}
	}

	// Build from other blocks
	for (int e = 1; e < p; e++)
	{
		// std::cout << " e = " << e << std::endl;
		int id_send = (rank + e) % p;
		int id_recv = (rank - e + p) % p;

		Eigen::MatrixXd recvBlock = Eigen::MatrixXd(v_size[id_recv], A.cols());
		// Step 7: Communication
		if (rank < id_recv)
		{
			// Send/recv data
			MPI_Send(A.data(), A.cols() * v_size[rank], MPI_DOUBLE, id_send, e, comm);
			MPI_Recv(recvBlock.data(), A.cols() * v_size[id_recv], MPI_DOUBLE, id_recv, e, comm, MPI_STATUS_IGNORE);
		}
		else
		{
			// Send/recv data
			MPI_Recv(recvBlock.data(), A.cols() * v_size[id_recv], MPI_DOUBLE, id_recv, e, comm, MPI_STATUS_IGNORE);
			MPI_Send(A.data(), A.cols() * v_size[rank], MPI_DOUBLE, id_send, e, comm);
		}
		int first_var_recv = displs[id_recv];
		tmp = A * recvBlock.transpose();

		for (int x = 0; x < v_size[rank]; x++)
		{
			for (int y = 0; y < v_size[id_recv]; y++)
			{
				resA(x, y + first_var_recv) = tmp(x, y);
			}
		}
	}

	free(displs);
}

// Need a lot of memory, no security
void build_ATA_distrib(Eigen::MatrixXd &A, Eigen::MatrixXd resA, int *v_size, MPI_Comm comm)
{
	int rank, p;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &p);

	int *displs = (int *)malloc(p * sizeof(int));
	displs[0] = 0;
	for (int i = 1; i < p; i++)
	{
		displs[i] = displs[i - 1] + v_size[i - 1];
	}
	// Build local
	resA = A.transpose() * A;

	// Build from other blocks
	for (int e = 1; e < p; e++)
	{
		// std::cout << " e = " << e << std::endl;
		int id_send = (rank + e) % p;
		int id_recv = (rank - e + p) % p;

		Eigen::MatrixXd recvBlock = Eigen::MatrixXd(v_size[id_recv], A.cols());
		// Step 7: Communication
		if (rank < id_recv)
		{
			// Send/recv data
			MPI_Send(A.data(), A.cols() * v_size[rank], MPI_DOUBLE, id_send, e, comm);
			MPI_Recv(recvBlock.data(), A.cols() * v_size[id_recv], MPI_DOUBLE, id_recv, e, comm, MPI_STATUS_IGNORE);
		}
		else
		{
			// Send/recv data
			MPI_Recv(recvBlock.data(), A.cols() * v_size[id_recv], MPI_DOUBLE, id_recv, e, comm, MPI_STATUS_IGNORE);
			MPI_Send(A.data(), A.cols() * v_size[rank], MPI_DOUBLE, id_send, e, comm);
		}
		resA += A.transpose() * recvBlock;
	}

	free(displs);
}
