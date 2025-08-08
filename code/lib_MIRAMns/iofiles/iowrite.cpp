#include <iofiles/iofiles.hpp>
#include <iomanip>
#include <cmath>

#define TAG_SAVE 0

void save_vector_double(double *matrix, const int nbRow, const int nbCol, const std::string path)
{
    extractData::listVectorDouble listVectors;

    for (int i = 0; i < nbRow; i++)
    {
        extractData::VectorDouble *vectorDouble = listVectors.add_vector();
        for (int j = 0; j < nbCol; j++)
        {
            vectorDouble->add_val(matrix[i * nbCol + j] * 1.0);
        };
    }

    std::fstream output(path, std::fstream::out | std::fstream::binary);
    if (!listVectors.SerializeToOstream(&output))
    {
        std::cerr << "Failed to write the vector" << std::endl;
    }
}

void save_vector_int(double *matrix, const int nbRow, const int nbCol, const std::string path)
{
    extractData::listVectorInt listVectors;

    for (int i = 0; i < nbRow; i++)
    {
        extractData::VectorInt *vectorInt = listVectors.add_vector();
        ;

        for (int j = 0; j < nbCol; j++)
        {
            vectorInt->add_val(matrix[i * nbCol + j]);
        };
    }
    std::cout << "NbRows (in message) : " << listVectors.vector_size() << std::endl;
    std::cout << "NbCols (in message) : " << listVectors.vector(0).val_size() << std::endl;

    std::fstream output(path, std::fstream::out | std::fstream::binary);
    if (!listVectors.SerializeToOstream(&output))
    {
        std::cerr << "Failed to write the vector" << std::endl;
    }
}

void save_master_vector_double(double *matrix, const int nbRow, const int nbCol, const std::string path, const int delim)
{
    extractData::masterListVectorDouble masterListVectors;
    int currentIndex = 0;

    extractData::listVectorDouble listVectors;

    for (int i = 0; i < nbRow; i++)
    {
        if (currentIndex > delim)
        {
            currentIndex = 0;
            std::string tmp_path = path + "." + std::to_string(masterListVectors.listvector_size());
            std::fstream output(tmp_path, std::fstream::out | std::fstream::binary);
            if (!listVectors.SerializeToOstream(&output))
            {
                std::cerr << "Failed to write the vector " << tmp_path << std::endl;
            }
            std::string *value = masterListVectors.add_listvector();
            *value = tmp_path;
            listVectors.clear_vector();
        }
        extractData::VectorDouble *vectorDouble = listVectors.add_vector();
        for (int j = 0; j < nbCol; j++)
        {
            vectorDouble->add_val(matrix[i * nbCol + j] * 1.0);
        };
        currentIndex++;
    }
    std::string tmp_path = path + "." + std::to_string(masterListVectors.listvector_size());
    std::fstream output(tmp_path, std::fstream::out | std::fstream::binary);
    if (!listVectors.SerializeToOstream(&output))
    {
        std::cerr << "Failed to write the vector " << tmp_path << std::endl;
    }
    std::string *value = masterListVectors.add_listvector();
    *value = tmp_path;

    for (int i = 0; i < masterListVectors.listvector_size(); i++)
    {
        std::cout << "File " << i << " : " << masterListVectors.listvector(i) << std::endl;
    }

    std::cout << "Number of files : " << masterListVectors.listvector_size() << std::endl;

    std::fstream outputMaster(path, std::fstream::out | std::fstream::binary);
    if (!masterListVectors.SerializeToOstream(&outputMaster))
    {
        std::cerr << "Failed to write the master vector" << std::endl;
    }
}

// [TODO]
void save_master_vector_int(double *matrix, const int nbRow, const int nbCol, const std::string path, const int delim)
{
    extractData::listVectorInt listVectors;

    for (int i = 0; i < nbRow; i++)
    {
        extractData::VectorInt *vectorInt = listVectors.add_vector();
        ;

        for (int j = 0; j < nbCol; j++)
        {
            vectorInt->add_val(matrix[i * nbCol + j]);
        };
    }
    std::cout << "NbRows (in message) : " << listVectors.vector_size() << std::endl;
    std::cout << "NbCols (in message) : " << listVectors.vector(0).val_size() << std::endl;

    std::fstream output(path, std::fstream::out | std::fstream::binary);
    if (!listVectors.SerializeToOstream(&output))
    {
        std::cerr << "Failed to write the vector" << std::endl;
    }
}

void save_vector_bool(bool *matrix, const int nbRow, const int nbCol, const std::string path)
{
    extractData::listVectorBool listVectors;

    for (int i = 0; i < nbRow; i++)
    {
        extractData::VectorBool *vectorInt = listVectors.add_vector();
        ;

        for (int j = 0; j < nbCol; j++)
        {
            vectorInt->add_val(matrix[i * nbCol + j]);
        };
    }
    std::cout << "NbRows (in message) : " << listVectors.vector_size() << std::endl;
    std::cout << "NbCols (in message) : " << listVectors.vector(0).val_size() << std::endl;

    std::fstream output(path, std::fstream::out | std::fstream::binary);
    if (!listVectors.SerializeToOstream(&output))
    {
        std::cerr << "Failed to write the vector" << std::endl;
    }
}

void save_mtx_dense(double *data, const unsigned long int nb_rows, const unsigned long int nb_cols, const std::string path)
{
    std::fstream file;
    file.open(path, std::ios::out);
    if (file)
    {
        file << nb_rows << " " << nb_cols << " " << nb_cols * nb_rows;
        for (unsigned long int i = 0; i < nb_rows; i++)
        {
            for (unsigned long int j = 0; j < nb_cols; j++)
            {
                file << std::endl
                     << i + 1 << " " << j + 1 << " " << data[i * nb_cols + j];
            }
        }
    }
    else
    {

        std::cerr << "Failed to write the MTX file." << std::endl;
    }
}

// Save a distributed matrice in a mtx file. Consider here the matrice is distributed by blocks of rows, the the same order than the ranks in the communicator.
// Nb_rows and nb_cols represents the global size of the distributed matrix.
void save_mtx_dense(double *data, const unsigned long int nb_rows, const unsigned long int nb_cols, const std::string path, MPI_Comm comm)
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

    if (rank == 0)
    { // Only rank 0 write the file
        std::fstream file;
        file.open(path, std::ios::out);
        file << nb_rows << " " << nb_cols << " " << nb_cols * nb_rows;
        for (unsigned long int i = 0; i < (unsigned long int)v_size[rank]; i++)
        {
            for (unsigned long int j = 0; j < nb_cols; j++)
            {
                file << std::endl
                     << i + 1 << " " << j + 1 << " " << std::setprecision(15) << data[i * nb_cols + j];
            }
        }
        int start = v_size[rank];
        for (int i = 1; i < p; i++)
        {
            double *tmp = (double *)malloc(v_size[i] * nb_cols * sizeof(double));
            MPI_Recv(tmp, v_size[i] * nb_cols, MPI_DOUBLE, i, TAG_SAVE, comm, MPI_STATUS_IGNORE);
            for (int x = 0; x < v_size[i]; x++)
            {
                for (unsigned long int y = 0; y < nb_cols; y++)
                {
                    file << std::endl
                         << start + x + 1 << " " << y + 1 << " " << tmp[x * nb_cols + y];
                }
            }
            start += v_size[i];
            free(tmp);
        }
        // file.close();
    }
    else
    {
        MPI_Send(data, v_size[rank] * nb_cols, MPI_DOUBLE, 0, TAG_SAVE, comm);
    }
    free(v_size);
}

void save_mtx_dense(Eigen::MatrixXcd A, const std::string path)
{
    std::fstream file;
    file.open(path, std::ios::out);
    if (file)
    {
        // Check if complex or not before save in file
        bool some_complex = false;
        for (int i = 0; i < A.rows(); i++)
        {
            for (int j = 0; j < A.cols(); j++)
            {
                if (std::imag(A.coeff(i, j)) != 0.0)
                {
                    some_complex = true;
                    goto endloop;
                }
            }
        }
    endloop:
        file << A.rows() << " " << A.cols() << " " << A.rows() * A.cols();
        for (long int i = 0; i < A.rows(); i++)
        {
            for (long int j = 0; j < A.cols(); j++)
            {
                file << std::endl
                     << i + 1 << " " << j + 1 << " " << std::real(A.coeff(i, j));
                if (some_complex)
                {
                    file << " " << std::imag(A.coeff(i, j));
                }
            }
        }
    }
    else
    {

        std::cerr << "Failed to write the MTX file." << std::endl;
    }
}

void save_mtx_dense(Eigen::MatrixXd A, const std::string path)
{
    std::fstream file;
    file.open(path, std::ios::out);
    if (file)
    {
        file << A.rows() << " " << A.cols() << " " << A.rows() * A.cols();
        for (long int i = 0; i < A.rows(); i++)
        {
            for (long int j = 0; j < A.cols(); j++)
            {
                file << std::endl
                     << i + 1 << " " << j + 1 << " " << A.coeff(i, j);
            }
        }
    }
    else
    {

        std::cerr << "Failed to write the MTX file." << std::endl;
    }
}