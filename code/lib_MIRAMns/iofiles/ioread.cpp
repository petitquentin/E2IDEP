#include <iofiles/iofiles.hpp>

#include <sstream>
#include <vector>
#include <cassert>
#include <cmath>
#include <mpi.h>

extractData::listVectorBool read_protobuf_message(const std::string path)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    extractData::listVectorBool listVectors;
    std::fstream input(path, std::fstream::in | std::fstream::binary);
    if (!listVectors.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse the protobuf message : " << path << std::endl;
    }
    return listVectors;
}

void read_mtx(const std::string path, std::vector<unsigned long int> &row, std::vector<unsigned long int> &col, unsigned long int &nb_rows, unsigned long int &nb_cols, unsigned long int &nnz)
{
    std::ifstream reader(path);
    if (reader)
    {

        row.clear();
        col.clear();
        while (reader.peek() == '%')
        {
            reader.ignore(2048, '\n');
        }
        // unsigned long nb_rows, nb_cols, nnz;
        reader >> nb_rows >> nb_cols >> nnz;
        std::string s;
        std::getline(reader, s);
        std::getline(reader, s);
        int nb_line = 0;
        std::vector<std::string> firstLine;
        std::istringstream iss(s);
        std::string s2;
        while (getline(iss, s2, ' '))
        {
            nb_line++;
            firstLine.push_back(s2);
        }

        unsigned long int r = std::stoul(firstLine[0]);
        unsigned long int c = std::stoul(firstLine[1]);

        if (r <= nb_rows && c <= nb_cols)
        {
            row.push_back(r - 1);
            col.push_back(c - 1);
        }
        for (unsigned long i = 0; i < (nnz - 1); i++)
        {
            reader >> r >> c;
            if (r <= nb_rows && c <= nb_cols)
            {
                row.push_back(r - 1);
                col.push_back(c - 1);
            }
        }

        reader.close();
    }
    else
    {
        std::cerr << "Impossible to open the file " << path << std::endl;
    }
}

void read_mtx(const std::string path, std::vector<unsigned long int> &row, std::vector<unsigned long int> &col, std::vector<double> &val, unsigned long int &nb_rows, unsigned long int &nb_cols, unsigned long int &nnz)
{
    std::ifstream reader(path);
    if (reader)
    {

        row.clear();
        col.clear();
        val.clear();
        while (reader.peek() == '%')
        {
            reader.ignore(2048, '\n');
        }
        // unsigned long nb_rows, nb_cols, nnz;
        reader >> nb_rows >> nb_cols >> nnz;
        std::string s;
        std::getline(reader, s);
        std::getline(reader, s);
        int nb_line = 0;
        std::vector<std::string> firstLine;
        std::istringstream iss(s);
        std::string s2;
        while (getline(iss, s2, ' '))
        {
            nb_line++;
            firstLine.push_back(s2);
        }

        unsigned long int r = std::stoul(firstLine[0]);
        unsigned long int c = std::stoul(firstLine[1]);
        double v = std::stod(firstLine[2]);

        if (r <= nb_rows && c <= nb_cols)
        {
            row.push_back(r - 1);
            col.push_back(c - 1);
            val.push_back(v);
        }
        for (unsigned long i = 0; i < (nnz - 1); i++)
        {
            reader >> r >> c >> v;
            if (r <= nb_rows && c <= nb_cols)
            {
                row.push_back(r - 1);
                col.push_back(c - 1);
                val.push_back(v);
            }
        }

        reader.close();
    }
    else
    {
        std::cerr << "Impossible to open the file " << path << std::endl;
    }
}

// Read a .mtx file and store the result in a double array.
void read_mtx_dense(const std::string path, double **matrix, unsigned long int &nb_rows, unsigned long int &nb_cols)
{
    std::ifstream reader(path);
    if (reader)
    {

        while (reader.peek() == '%')
        {
            reader.ignore(2048, '\n');
        }
        unsigned long int nnz;
        reader >> nb_rows >> nb_cols >> nnz;

        if ((*matrix) != NULL)
        {
            free(*matrix);
        }
        (*matrix) = (double *)malloc(nb_rows * nb_cols * sizeof(double));

        std::string s;
        std::getline(reader, s);
        std::getline(reader, s);
        int nb_line = 0;
        std::vector<std::string> firstLine;
        std::istringstream iss(s);
        std::string s2;
        while (getline(iss, s2, ' '))
        {
            nb_line++;
            firstLine.push_back(s2);
        }

        unsigned long int r = std::stoul(firstLine[0]);
        unsigned long int c = std::stoul(firstLine[1]);
        double v = std::stod(firstLine[2]);

        if (r <= nb_rows && c <= nb_cols)
        {
            (*matrix)[(r - 1) * nb_cols + (c - 1)] = v;
        }
        for (unsigned long i = 0; i < (nnz - 1); i++)
        {
            reader >> r >> c >> v;
            if (r <= nb_rows && c <= nb_cols)
            {
                (*matrix)[(r - 1) * nb_cols + (c - 1)] = v;
            }
        }

        reader.close();
    }
    else
    {
        std::cerr << "Impossible to open the file " << path << std::endl;
    }
}

// Read a .mtx file and store the result in a double array (distributed version).
void read_mtx_dense(const std::string path, double **matrix, unsigned long int &nb_rows, unsigned long int &nb_cols, MPI_Comm comm)
{
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    std::ifstream reader(path);

    if (reader)
    {

        while (reader.peek() == '%')
        {
            reader.ignore(2048, '\n');
        }
        unsigned long int nnz;
        reader >> nb_rows >> nb_cols >> nnz;

        if ((*matrix) != NULL)
        {
            free(*matrix);
        }
        (*matrix) = (double *)malloc(nb_rows * nb_cols * sizeof(double));

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

        std::string s;
        std::getline(reader, s);
        std::getline(reader, s);
        int nb_line = 0;
        std::vector<std::string> firstLine;
        std::istringstream iss(s);
        std::string s2;
        while (getline(iss, s2, ' '))
        {
            nb_line++;
            firstLine.push_back(s2);
        }

        unsigned long int r = std::stoul(firstLine[0]);
        unsigned long int c = std::stoul(firstLine[1]);
        double v = std::stod(firstLine[2]);
        r = r - 1;
        c = c - 1;

        if (r <= nb_rows && c >= startRow && c < startRow + v_size[rank])
        {
            (*matrix)[r * v_size[rank] + c - startRow] = v;
        }
        for (unsigned long i = 0; i < (nnz - 1); i++)
        {
            reader >> r >> c >> v;

            r = r - 1;
            c = c - 1;
            // std::cout << "r: " << r << ", c: " << c << ", v: " << v << std::endl;
            if (r <= nb_rows && c >= startRow && c < startRow + v_size[rank])
            {
                // std::cout << v << " " << r * v_size[rank] + c - startRow << std::endl;
                (*matrix)[r * v_size[rank] + c - startRow] = v;
            }
        }
        free(v_size);
        reader.close();
    }
    else
    {
        std::cerr << "Impossible to open the file " << path << std::endl;
    }
}

extractData::listVectorDouble read_listDouble(const std::string path)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    extractData::listVectorDouble listVectors;
    std::fstream input(path, std::fstream::in | std::fstream::binary);
    if (!listVectors.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse the protobuf message : " << path << std::endl;
    }
    return listVectors;
}

extractData::listVectorInt read_listInt(const std::string path)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    extractData::listVectorInt listVectors;
    std::fstream input(path, std::fstream::in | std::fstream::binary);
    if (!listVectors.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse the protobuf message : " << path << std::endl;
    }
    return listVectors;
}

extractData::masterListVectorDouble read_masterListDouble(std::string path)
{
    extractData::masterListVectorDouble file;
    std::fstream input(path, std::fstream::in | std::fstream::binary);
    if (!file.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse the protobuf message : " << path << std::endl;
    }
    return file;
}

extractData::masterListVectorInt read_masterListInt(std::string path)
{
    extractData::masterListVectorInt file;
    std::fstream input(path, std::fstream::in | std::fstream::binary);
    if (!file.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse the protobuf message : " << path << std::endl;
    }
    return file;
}

extractData::masterListVectorBool read_masterListBool(std::string path)
{
    extractData::masterListVectorBool file;
    std::fstream input(path, std::fstream::in | std::fstream::binary);
    if (!file.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse the protobuf message : " << path << std::endl;
    }
    return file;
}

// Read a .mtx file and store the result in a double array.
void get_mtx_dim(const std::string path, unsigned long int &nb_rows, unsigned long int &nb_cols)
{
    std::ifstream reader(path);
    if (reader)
    {

        while (reader.peek() == '%')
        {
            reader.ignore(2048, '\n');
        }
        unsigned long int nnz;
        reader >> nb_rows >> nb_cols >> nnz;

        reader.close();
    }
    else
    {
        std::cerr << "Impossible to open the file " << path << std::endl;
    }
}