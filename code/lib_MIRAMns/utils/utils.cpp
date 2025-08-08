#include <proto/protoData.pb.h>
#include <utils/utils.hpp>
#include <iofiles/iofiles.hpp>
#include <iomanip>
#include <vector>
#include <mpi.h>
#include <cmath>

#define TAG_MEAN 0
#define TAG_DATA 1

void print_listVectors(const extractData::listVectorBool listVectors)
{
    int nb_vector = 0;
    for (int i = 0; i < listVectors.vector_size(); i++)
    {
        int nb_element = 0;
        for (int j = 0; j < listVectors.vector(i).val_size(); j++)
        {
            nb_element++;
            std::cout << j << " " << listVectors.vector(i).val(j) << std::endl;
        }
        std::cout << "Nb elem in vector " << i << ": " << nb_element << std::endl;
        nb_vector++;
    }
    std::cout << "Number of vectors : " << nb_vector << std::endl;
}

void print_matrix_from_master_double(std::string path)
{
    extractData::masterListVectorDouble masterFile = read_masterListDouble(path);
    for (int i = 0; i < masterFile.listvector_size(); i++)
    {
        extractData::listVectorDouble listVectorsDouble = read_listDouble(masterFile.listvector(i));
        for (int j = 0; j < listVectorsDouble.vector_size(); j++)
        {
            std::cout << listVectorsDouble.vector(j).val(0);
            for (int k = 1; k < listVectorsDouble.vector(j).val_size(); k++)
            {
                std::cout << " " << listVectorsDouble.vector(j).val(k);
            }
            std::cout << std::endl;
        }
    }
}

void getGridDim(const int nbNodes, int *xGrid, int *yGrid)
{
    int numberTwoPower = 0;
    int r = nbNodes;
    while (r % 2 == 0 && r != 1)
    {
        numberTwoPower++;
        r = r / 2;
    }

    if (r != 1)
    {
        if ((int)std::sqrt(r * 1.0) != std::sqrt(r * 1.0))
        {
            int a = std::round(std::sqrt(r));
            int b = a * a - r;
            while ((int)std::sqrt(b * 1.0) != std::sqrt(b * 1.0))
            {
                a++;
                b = a * a - r;
            }
            *xGrid = std::min(a - std::sqrt(b), a + std::sqrt(b));
            *yGrid = std::max(a - std::sqrt(b), a + std::sqrt(b));
        }
        else
        {
            *xGrid = std::sqrt(r);
            *yGrid = std::sqrt(r);
        }
    }
    else
    {
        *xGrid = 1;
        *yGrid = 1;
    }
    for (int i = 0; i < numberTwoPower; i++)
    {
        if (i % 2 == 0)
        {
            *xGrid *= 2;
        }
        else
        {
            *yGrid *= 2;
        }
    }
}

int createELLPACK(extractData::listVectorBool dataset, int **matrix, MPI_Comm comm)
{

    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    const int n = dataset.vector(0).val_size();
    const int k = dataset.vector_size();

    int *v_size = (int *)malloc(p * sizeof(int));
    int startRow = 0;

    for (int i = 0; i < p; i++)
    {
        v_size[i] = std::floor(n * 1.0 / p);
        if (i < n % p)
        {
            v_size[i] = v_size[i] + 1;
        }
        if (i < rank)
        {
            startRow += v_size[i];
        }
    }
    int *nnz = (int *)calloc(k, sizeof(int));

    for (int i = 0; i < k; i++)
    {
        for (int j = startRow; j < startRow + v_size[rank]; j++)
        {
            if (dataset.vector(i).val(j) != 0)
            {
                nnz[i]++;
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, nnz, k, MPI_INT, MPI_SUM, comm);

    int max = nnz[0];
    nnz[0] = 0;
    for (int i = 1; i < k; i++)
    {
        if (nnz[i] > max)
        {
            max = nnz[i];
        }
        nnz[i] = 0;
    }
    // Build ELLPACK matrix
    if ((*matrix) != NULL)
    {
        free(*matrix);
    }
    (*matrix) = (int *)malloc(k * max * sizeof(int));
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (dataset.vector(i).val(j) != 0)
            {
                (*matrix)[i * max + nnz[i]] = j;
                nnz[i]++;
            }
        }
        for (int j = nnz[i]; j < max; j++)
        {
            (*matrix)[i * max + j] = -1;
        }
    }

    free(v_size);
    free(nnz);

    return max;
}

int createELLPACKfromCOO(std::vector<unsigned long int> row, std::vector<unsigned long int> col, const unsigned long int nb_rows, const unsigned long int nb_cols, int **matrix, MPI_Comm comm)
{

    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    int *idx = (int *)malloc(sizeof(int) * (nb_rows + 1));
    unsigned long int index = 0;
    idx[0] = 0;

    for (unsigned long int i = 0; i < row.size(); i++)
    {
        while (row[i] > index)
        {
            idx[index + 1] = i;
            index++;
        }
    }

    for (unsigned long int i = index + 1; i < nb_rows + 1; i++)
    {
        idx[i] = row.size();
    }

    // Find max
    int max = 0;
    for (unsigned long int i = 0; i < nb_rows; i++)
    {
        if (idx[i + 1] - idx[i] > max)
        {
            max = idx[i + 1] - idx[i];
        }
    }

    int *nnz = (int *)calloc(nb_rows, sizeof(int));

    // Build ELLPACK matrix
    if ((*matrix) != NULL)
    {
        free(*matrix);
    }
    (*matrix) = (int *)malloc(nb_rows * max * sizeof(int));

    if (rank == 0)
    {
        if ((*matrix) == NULL)
        {
            std::cout << "not initalized " << (int)(nb_rows * max * sizeof(int)) << std::endl;
        }
    }

    for (unsigned long int i = 0; i < row.size(); i++)
    {
        if (rank == 0)
        {
            std::cout << (*matrix)[(int)(row[i] * max + nnz[row[i]])] << std::endl;
            std::cout << nnz[row[i]] << "/" << max << "(" << row[i] * max + nnz[row[i]] << ")" << std::endl;
        }
        (*matrix)[row[i] * max + nnz[row[i]]] = col[i];
        nnz[row[i]]++;
    }

    for (unsigned long int i = 0; i < nb_rows; i++)
    {
        for (int j = nnz[i]; j < max; j++)
        {
            (*matrix)[i * max + j] = -1;
        }
    }

    free(nnz);

    return max;
}

void createCSRfromCOO(std::vector<unsigned long int> row, std::vector<unsigned long int> col, const unsigned long int nb_rows, std::vector<unsigned long int> *idx, MPI_Comm comm)
{

    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    (*idx).clear();
    for (unsigned long int i = 0; i < nb_rows + 1; i++)
    {
        (*idx).push_back(0);
    }
    unsigned long int index = 0;
    (*idx)[0] = 0;

    for (unsigned long int i = 0; i < row.size(); i++)
    {
        while (row[i] > index)
        {
            (*idx)[index + 1] = i;
            index++;
        }
    }

    for (unsigned long int i = index + 1; i < nb_rows + 1; i++)
    {
        (*idx)[i] = row.size();
    }
}

void print2Ddouble(double *mat, int nb_row, int nb_col)
{
    for (int i = 0; i < nb_row; i++)
    {
        std::cout << mat[i * nb_col];
        for (int j = 1; j < nb_col; j++)
        {
            std::cout << " " << mat[i * nb_col + j];
        }
        std::cout << std::endl;
    }
}

double sparsity_double(std::string path)
{
    extractData::masterListVectorDouble masterFile = read_masterListDouble(path);
    double sparsity = 0;
    long long nnz_total = 0;

    for (int i = 0; i < masterFile.listvector_size(); i++)
    {

        int n = 0;
        int nnz = 0;
        extractData::listVectorDouble listVectorsDouble = read_listDouble(masterFile.listvector(i));
        for (int j = 0; j < listVectorsDouble.vector_size(); j++)
        {
            for (int k = 0; k < listVectorsDouble.vector(j).val_size(); k++)
            {
                n++;
                if (listVectorsDouble.vector(j).val(k) != 0)
                {
                    nnz++;
                }
            }
        }
        std::cout << "Step " << i << ": n : " << n << " nnz : " << nnz << " local_sparsity : " << nnz * 1.0 / n << " sparsity : " << sparsity << std::endl;
        sparsity += (n - nnz) * 1.0 / n;
        nnz_total += nnz;
    }
    std::cout << "nnz total : " << nnz_total << std::endl;
    return sparsity / (masterFile.listvector_size() * 1.0);
}

double sparsity_bool(std::string path)
{
    extractData::listVectorBool file = read_protobuf_message(path);
    ;
    double sparsity = 0;

    long n = 0;
    long nnz = 0;
    std::cout << "i = " << file.vector(0).val_size() << std::endl;
    std::cout << "j = " << file.vector_size() << std::endl;
    for (int j = 0; j < file.vector_size(); j++)
    {
        for (int k = 0; k < file.vector(j).val_size(); k++)
        {
            n++;
            if (file.vector(j).val(k) != 0)
            {
                nnz++;
            }
        }
    }
    std::cout << "n : " << n << " nnz : " << nnz << " local_sparsity : " << nnz * 1.0 / n << " sparsity : " << sparsity << std::endl;
    sparsity += (n - nnz) * 1.0 / n;
    return sparsity;
}

void mtx_to_dense(bool **v, std::vector<unsigned long int> row, std::vector<unsigned long int> col, const unsigned long int nb_rows, const unsigned long int nb_cols, MPI_Comm comm)
{

    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

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
    *v = (bool *)calloc(v_size[rank] * nb_rows, sizeof(bool));

    for (unsigned long int i = 0; i < row.size(); i++)
    {
        if (col[i] >= startRow && col[i] < startRow + v_size[rank])
        {

            (*v)[row[i] * v_size[rank] + col[i] - startRow] = 1;
        }
    }

    // Good before

    // GOOGLE_PROTOBUF_VERIFY_VERSION;
    // extractData::listVectorBool listVectors;

    // bool *vectorBool = (bool *)malloc(sizeof(bool) * nb_cols);

    // for (unsigned long int i = 0; i < nb_rows; i++)
    // {
    //     std::cout << i << "/" << nb_rows << std::endl;
    //     extractData::VectorBool *currentVector = listVectors.add_vector();
    //     for (unsigned long int j = 0; j < nb_cols; j++)
    //     {
    //         vectorBool[j] = 0;
    //     }
    //     for (int j = idx[i]; j < idx[i + 1]; j++)
    //     {
    //         vectorBool[col[j]] = 1;
    //     }
    //     for (unsigned long int j = 0; j < nb_cols; j++)
    //     {
    //         currentVector->add_val(vectorBool[j]);
    //     }
    // }
    // free(vectorBool);

    // extractData::VectorBool vector;
    // std::fstream input(path, std::fstream::in | std::fstream::binary);

    // extractData::listVectorBool listVectors;

    // for (int i = 0; i < nbRow; i++)
    // {
    //     extractData::VectorBool *vectorInt = listVectors.add_vector();
    //     ;

    //     for (int j = 0; j < nbCol; j++)
    //     {
    //         vectorInt->add_val(matrix[i * nbCol + j]);
    //     };
    // }
    // std::cout << "NbRows (in message) : " << listVectors.vector_size() << std::endl;
    // std::cout << "NbCols (in message) : " << listVectors.vector(0).val_size() << std::endl;

    // std::fstream output(path, std::fstream::out | std::fstream::binary);
    // if (!listVectors.SerializeToOstream(&output))
    // {
    //     std::cerr << "Failed to write the vector" << std::endl;
    // }
    free(v_size);
}

void mtx_to_dense(double **v, std::vector<unsigned long int> row, std::vector<unsigned long int> col, std::vector<double> val, const unsigned long int nb_rows, const unsigned long int nb_cols, MPI_Comm comm)
{

    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

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
    *v = (double *)calloc(v_size[rank] * nb_rows, sizeof(double));

    for (unsigned long int i = 0; i < row.size(); i++)
    {
        if (col[i] >= startRow && col[i] < startRow + v_size[rank])
        {

            (*v)[row[i] * v_size[rank] + col[i] - startRow] = val[i];
        }
    }

    free(v_size);
}

void build_covariance(double *data, double **output, const unsigned long int nb_rows, const unsigned long int nb_cols, MPI_Comm comm)
{

    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

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

    if ((*output) != NULL)
    {
        free(*output);
    }
    (*output) = (double *)calloc(v_size[rank] * nb_cols, sizeof(double));
    // Step 1 Find the mean of variables (X).
    double *mean = (double *)calloc(v_size[rank], sizeof(double));
    // double *variance = (double *)calloc(v_size[rank], sizeof(double));
    for (int i = 0; i < v_size[rank]; i++)
    {
        bool overflow = false;
        for (unsigned long int j = 0; j < nb_rows; j++)
        {
            if (std::abs(data[j * v_size[rank] + i]) > std::numeric_limits<double>::max() - std::abs(mean[i]) || overflow)
            {
                if (!overflow)
                {
                    std::cout << "S1: Overflow avoided (j = " << j << "/" << nb_rows << ")" << std::endl;
                    mean[i] = mean[i] * 1.0 / nb_rows;
                    overflow = true;
                }
                mean[i] += data[j * v_size[rank] + i] * 1.0 / nb_rows;
            }
            else
            {
                mean[i] += data[j * v_size[rank] + i];
            }
        }
        if (!overflow)
        {

            mean[i] = mean[i] * 1.0 / nb_rows;
        }
        // Step 2: Subtract the mean (not need here, we will redo computation to avoid lost of precission)
        // Step 3 : Take the sum of the squares of the differences obtained in the previous step.
        // Step 4 (merged): Divide this value by 1 less than the total to get the sample variance of the first variable
        overflow = false;
        for (unsigned long int j = 0; j < nb_rows; j++)
        {
            if (std::abs((data[j * v_size[rank] + i] - mean[i]) * (data[j * v_size[rank] + i] - mean[i])) > std::numeric_limits<double>::max() - std::abs((*output)[i * nb_cols + startRow + i]) || overflow)
            {
                if (!overflow)
                {
                    std::cout << "S4: Overflow avoided (j = " << j << "/" << nb_rows << ")" << std::endl;
                    (*output)[i * nb_cols + startRow + i] = (*output)[i * nb_cols + startRow + i] * 1.0 / (nb_rows - 1);
                    overflow = true;
                }
                (*output)[i * nb_cols + startRow + i] += (data[j * v_size[rank] + i] - mean[i]) * (data[j * v_size[rank] + i] - mean[i]) / (nb_rows - 1);
            }
            else
            {
                (*output)[i * nb_cols + startRow + i] += (data[j * v_size[rank] + i] - mean[i]) * (data[j * v_size[rank] + i] - mean[i]);
            }

            //(*output)[i * nb_cols + startRow + i] += (data[j * v_size[rank] + i] - mean[i]) * (data[j * v_size[rank] + i] - mean[i]) / (nb_rows - 1);
        }
        if (!overflow)
        {
            (*output)[i * nb_cols + startRow + i] = (*output)[i * nb_cols + startRow + i] * 1.0 / (nb_rows - 1);
        }
    }
    // Step 5: Compute local co-variance
    for (int x = 0; x < v_size[rank]; x++)
    {
        for (int y = x + 1; y < v_size[rank]; y++)
        {
            bool overflow = false;
            for (unsigned long int i = 0; i < nb_rows; i++)
            {
                if (std::abs((data[i * v_size[rank] + x] - mean[x]) * (data[i * v_size[rank] + y] - mean[y])) > std::numeric_limits<double>::max() - std::abs((*output)[x * nb_cols + startRow + y]) || overflow)
                {
                    if (!overflow)
                    {
                        std::cout << "S5: Overflow avoided (i = " << i << "/" << nb_rows << ")" << std::endl;
                        (*output)[x * nb_cols + startRow + y] = (*output)[x * nb_cols + startRow + y] * 1.0 / (nb_rows - 1);
                        overflow = true;
                    }
                    (*output)[x * nb_cols + startRow + y] += (data[i * v_size[rank] + x] - mean[x]) * (data[i * v_size[rank] + y] - mean[y]) / (nb_rows - 1);
                }
                else
                {
                    (*output)[x * nb_cols + startRow + y] += (data[i * v_size[rank] + x] - mean[x]) * (data[i * v_size[rank] + y] - mean[y]);
                }
                //(*output)[x * nb_cols + startRow + y] += (data[i * v_size[rank] + x] - mean[x]) * (data[i * v_size[rank] + y] - mean[y]) / (nb_rows - 1);
            }

            if (!overflow)
            {
                (*output)[x * nb_cols + startRow + y] = (*output)[x * nb_cols + startRow + y] * 1.0 / (nb_rows - 1);
            }
            (*output)[y * nb_cols + startRow + x] = (*output)[x * nb_cols + startRow + y];
        }
    }
    // Step 6 :Prepare rotation of blocks
    double *rot_mean = (double *)malloc(v_size[(p + rank - 1) % p] * sizeof(double));
    double *rot_data = (double *)malloc(v_size[(p + rank - 1) % p] * nb_rows * sizeof(double));

    for (int e = 1; e < p; e++)
    {
        // std::cout << " e = " << e << std::endl;
        int id_send = (rank + e) % p;
        int id_recv = (rank - e + p) % p;

        if (e != 1 && v_size[(rank + p - (e - 1)) % p] != v_size[id_recv])
        {
            rot_mean = (double *)realloc(rot_mean, v_size[id_recv] * sizeof(double));
            rot_data = (double *)realloc(rot_data, nb_rows * v_size[id_recv] * sizeof(double));
        }
        // Step 7: Communication
        if (rank < id_recv)
        {
            // std::cout << "Rank " << rank << ": I send first (id_recv " << id_recv << ", id_send " << id_send << ")" << std::endl;
            //  Send/recv mean
            MPI_Send(mean, v_size[rank], MPI_DOUBLE, id_send, TAG_MEAN, comm);
            MPI_Recv(rot_mean, v_size[id_recv], MPI_DOUBLE, id_recv, TAG_MEAN, comm, MPI_STATUS_IGNORE);
            // Send/recv data
            MPI_Send(data, nb_rows * v_size[rank], MPI_DOUBLE, id_send, TAG_MEAN, comm);
            MPI_Recv(rot_data, nb_rows * v_size[id_recv], MPI_DOUBLE, id_recv, TAG_MEAN, comm, MPI_STATUS_IGNORE);
        }
        else
        {
            // std::cout << "Rank " << rank << ": I recv first (id_recv " << id_recv << ", id_send " << id_send << ")" << std::endl;
            //  Send/recv mean
            MPI_Recv(rot_mean, v_size[id_recv], MPI_DOUBLE, id_recv, TAG_MEAN, comm, MPI_STATUS_IGNORE);
            MPI_Send(mean, v_size[rank], MPI_DOUBLE, id_send, TAG_MEAN, comm);
            // Send/recv data
            MPI_Recv(rot_data, nb_rows * v_size[id_recv], MPI_DOUBLE, id_recv, TAG_MEAN, comm, MPI_STATUS_IGNORE);
            MPI_Send(data, nb_rows * v_size[rank], MPI_DOUBLE, id_send, TAG_MEAN, comm);
        }
        int first_var_recv = 0;
        for (int i = 0; i < id_recv; i++)
        {
            first_var_recv += v_size[i];
        }
        // step 8: Compute Covariance with others blocks [TODO]

        for (int x = 0; x < v_size[id_recv]; x++)
        {
            for (int y = 0; y < v_size[rank]; y++)
            {
                bool overflow = false;
                for (unsigned long int i = 0; i < nb_rows; i++)
                {
                    if (std::abs((rot_data[i * v_size[id_recv] + x] - rot_mean[x]) * (data[i * v_size[rank] + y] - mean[y])) > std::numeric_limits<double>::max() - std::abs((*output)[y * nb_cols + first_var_recv + x]) || overflow)
                    {
                        if (!overflow)
                        {
                            std::cout << "S8: Overflow avoided (i = " << i << "/" << nb_rows << ")" << std::endl;
                            (*output)[y * nb_cols + first_var_recv + x] = (*output)[y * nb_cols + first_var_recv + x] * 1.0 / (nb_rows - 1);
                            overflow = true;
                        }
                        (*output)[y * nb_cols + first_var_recv + x] += (rot_data[i * v_size[id_recv] + x] - rot_mean[x]) * (data[i * v_size[rank] + y] - mean[y]) / (nb_rows - 1);
                    }
                    else
                    {
                        (*output)[y * nb_cols + first_var_recv + x] += (rot_data[i * v_size[id_recv] + x] - rot_mean[x]) * (data[i * v_size[rank] + y] - mean[y]);
                    }
                    // (*output)[y * nb_cols + first_var_recv + x] += (rot_data[i * v_size[id_recv] + x] - rot_mean[x]) * (data[i * v_size[rank] + y] - mean[y]) / (nb_rows - 1);
                }
                if (!overflow)
                {
                    (*output)[y * nb_cols + first_var_recv + x] = (*output)[y * nb_cols + first_var_recv + x] * 1.0 / (nb_rows - 1);
                }
            }
        }
    }
    free(rot_mean);
    free(rot_data);
    free(mean);
    // free(variance);
    free(v_size);
}

void build_covariance_nosecurity(double *data, double **output, const unsigned long int nb_rows, const unsigned long int nb_cols, MPI_Comm comm)
{

    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

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

    if ((*output) != NULL)
    {
        free(*output);
    }
    (*output) = (double *)calloc(v_size[rank] * nb_cols, sizeof(double));
    // Step 1 Find the mean of variables (X).
    double *mean = (double *)calloc(v_size[rank], sizeof(double));
    // double *variance = (double *)calloc(v_size[rank], sizeof(double));
    for (int i = 0; i < v_size[rank]; i++)
    {
        for (unsigned long int j = 0; j < nb_rows; j++)
        {
            mean[i] += data[j * v_size[rank] + i];
        }
        mean[i] = mean[i] * 1.0 / nb_rows;
        // Step 2: Subtract the mean (not need here, we will redo computation to avoid lost of precission)
        // Step 3 : Take the sum of the squares of the differences obtained in the previous step.
        // Step 4 (merged): Divide this value by 1 less than the total to get the sample variance of the first variable
        for (unsigned long int j = 0; j < nb_rows; j++)
        {
            (*output)[i * nb_cols + startRow + i] += (data[j * v_size[rank] + i] - mean[i]) * (data[j * v_size[rank] + i] - mean[i]) / (nb_rows - 1);
        }
    }
    // Step 5: Compute local co-variance
    for (int x = 0; x < v_size[rank]; x++)
    {
        for (int y = x + 1; y < v_size[rank]; y++)
        {
            for (unsigned long int i = 0; i < nb_rows; i++)
            {
                (*output)[x * nb_cols + startRow + y] += (data[i * v_size[rank] + x] - mean[x]) * (data[i * v_size[rank] + y] - mean[y]) / (nb_rows - 1);
            }
            (*output)[y * nb_cols + startRow + x] = (*output)[x * nb_cols + startRow + y];
        }
    }
    // Step 6 :Prepare rotation of blocks
    double *rot_mean = (double *)malloc(v_size[(p + rank - 1) % p] * sizeof(double));
    double *rot_data = (double *)malloc(v_size[(p + rank - 1) % p] * nb_rows * sizeof(double));

    for (int e = 1; e < p; e++)
    {
        // std::cout << " e = " << e << std::endl;
        int id_send = (rank + e) % p;
        int id_recv = (rank - e + p) % p;

        if (e != 1 && v_size[(rank + p - (e - 1)) % p] != v_size[id_recv])
        {
            rot_mean = (double *)realloc(rot_mean, v_size[id_recv] * sizeof(double));
            rot_data = (double *)realloc(rot_data, nb_rows * v_size[id_recv] * sizeof(double));
        }
        // Step 7: Communication
        if (rank < id_recv)
        {
            // std::cout << "Rank " << rank << ": I send first (id_recv " << id_recv << ", id_send " << id_send << ")" << std::endl;
            //  Send/recv mean
            MPI_Send(mean, v_size[rank], MPI_DOUBLE, id_send, TAG_MEAN, comm);
            MPI_Recv(rot_mean, v_size[id_recv], MPI_DOUBLE, id_recv, TAG_MEAN, comm, MPI_STATUS_IGNORE);
            // Send/recv data
            MPI_Send(data, nb_rows * v_size[rank], MPI_DOUBLE, id_send, TAG_MEAN, comm);
            MPI_Recv(rot_data, nb_rows * v_size[id_recv], MPI_DOUBLE, id_recv, TAG_MEAN, comm, MPI_STATUS_IGNORE);
        }
        else
        {
            // std::cout << "Rank " << rank << ": I recv first (id_recv " << id_recv << ", id_send " << id_send << ")" << std::endl;
            //  Send/recv mean
            MPI_Recv(rot_mean, v_size[id_recv], MPI_DOUBLE, id_recv, TAG_MEAN, comm, MPI_STATUS_IGNORE);
            MPI_Send(mean, v_size[rank], MPI_DOUBLE, id_send, TAG_MEAN, comm);
            // Send/recv data
            MPI_Recv(rot_data, nb_rows * v_size[id_recv], MPI_DOUBLE, id_recv, TAG_MEAN, comm, MPI_STATUS_IGNORE);
            MPI_Send(data, nb_rows * v_size[rank], MPI_DOUBLE, id_send, TAG_MEAN, comm);
        }
        int first_var_recv = 0;
        for (int i = 0; i < id_recv; i++)
        {
            first_var_recv += v_size[i];
        }
        // step 8: Compute Covariance with others blocks [TODO]

        for (int x = 0; x < v_size[id_recv]; x++)
        {
            for (int y = 0; y < v_size[rank]; y++)
            {
                for (unsigned long int i = 0; i < nb_rows; i++)
                {
                    (*output)[y * nb_cols + first_var_recv + x] += (rot_data[i * v_size[id_recv] + x] - rot_mean[x]) * (data[i * v_size[rank] + y] - mean[y]) / (nb_rows - 1);
                }
            }
        }
    }
    free(rot_mean);
    free(rot_data);
    free(mean);
    // free(variance);
    free(v_size);
}
