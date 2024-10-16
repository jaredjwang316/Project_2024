#include <iostream>
#include <vector>
#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <random>

#define MASTER 0
#define MAX_VAL 1000

std::vector<int> getCount(const std::vector<int> &arr, int offset, int chunkSize, int max)
{
    std::vector<int> counts(max + 1, 0);

    for (int i = offset; i < std::min(offset + chunkSize, static_cast<int>(arr.size())); i++)
    {
        counts[arr[i]]++;
    }

    return counts;
}

std::vector<int> prefixSum(const std::vector<int> &counts)
{
    std::vector<int> prefixSums(counts.size());
    prefixSums[0] = 0;

    for (int i = 1; i < static_cast<int>(counts.size()); i++)
    {
        prefixSums[i] = prefixSums[i - 1] + counts[i - 1];
    }

    return prefixSums;
}

bool isSorted(const std::vector<int> &arr)
{
    for (size_t i = 1; i < arr.size(); i++)
    {
        if (arr[i] < arr[i - 1])
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;

    int arrSize;

    if (argc == 2)
    {
        arrSize = atoi(argv[1]);
    }
    else
    {
        std::cout << "\n Please provide the size of the array to be sorted" << std::endl;
        return 0;
    }

    int size, rank;
    std::vector<int> array;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Metadata
    adiak::init(nullptr);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    adiak::value("algorithm", std::string("radix"));
    adiak::value("programming_model", std::string("mpi"));
    adiak::value("data_type", std::string("int"));
    adiak::value("size_of_data_type", sizeof(int));
    adiak::value("input_size", arrSize);
    adiak::value("input_type", std::string("Random"));
    adiak::value("num_procs", size);
    adiak::value("scalability", std::string("strong"));
    adiak::value("group_num", 5);
    adiak::value("implementation_source", std::string("handwritten"));

    MPI_Comm comm_dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_dup);

    // Data initialization
    CALI_MARK_BEGIN("data_init_runtime");
    if (rank == MASTER)
    {
        array.resize(arrSize);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, MAX_VAL);
        for (int i = 0; i < arrSize; i++)
        {
            array[i] = dis(gen);
        }
    }
    CALI_MARK_END("data_init_runtime");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("comm_small");

    // Calculate chunk size
    int chunkSize = (arrSize + size - 1) / size;
    std::vector<int> localCounts(MAX_VAL + 1, 0);
    std::vector<int> globalCounts(MAX_VAL + 1, 0);

    // Broadcast the array to all processes
    CALI_MARK_BEGIN("comm_large");
    if (rank == MASTER)
    {
        MPI_Bcast(array.data(), arrSize, MPI_INT, MASTER, MPI_COMM_WORLD);
    }
    else
    {
        array.resize(arrSize);
        MPI_Bcast(array.data(), arrSize, MPI_INT, MASTER, MPI_COMM_WORLD);
    }
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Local counting
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    int startIdx = rank * chunkSize;
    localCounts = getCount(array, startIdx, chunkSize, MAX_VAL);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Combine counts
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    MPI_Reduce(localCounts.data(), globalCounts.data(), MAX_VAL + 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Master process creates final sorted array
    if (rank == MASTER)
    {
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_small");
        std::vector<int> prefixSums = prefixSum(globalCounts);
        std::vector<int> result(arrSize);

        // Build sorted array
        for (int i = 0; i < arrSize; i++)
        {
            int value = array[i];
            result[prefixSums[value]] = value;
            prefixSums[value]++;
        }
        CALI_MARK_END("comp_small");

        /*

        // Print Out Input
        std::cout << "Input: ";
        for (int i = 0; i < arrSize; i++)
        {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;

        // Print Out Result
        std::cout << "Result: ";
        for (int i = 0; i < arrSize; i++)
        {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;

        */
        CALI_MARK_END("comp");

        // Check correctness
        CALI_MARK_BEGIN("correctness_check");
        if (isSorted(result))
        {
            std::cout << "Array is sorted correctly." << std::endl;
        }
        else
        {
            std::cout << "Array is NOT sorted correctly." << std::endl;
        }
        CALI_MARK_END("correctness_check");
    }

    adiak::fini();
    MPI_Finalize();

    return 0;
}