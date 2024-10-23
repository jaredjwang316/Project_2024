#include <iostream>
#include <vector>
#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <random>
#include <algorithm>

#define MASTER 0

enum InputType
{
    RANDOM,
    SORTED,
    REVERSE_SORTED,
    PERTURBED
};

std::vector<int> generateData(int chunkSize, InputType inputType, int rank, int totalSize)
{
    std::vector<int> chunk;
    chunk.resize(chunkSize);

    long startIdx = rank * chunkSize;

    switch (inputType)
    {
    case RANDOM:
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, totalSize);
        for (int i = 0; i < chunkSize; i++)
        {
            chunk[i] = dis(gen);
        }
        break;
    }
    case SORTED:
    {
        for (int i = 0; i < chunkSize; i++)
        {
            chunk[i] = startIdx + i;
        }
        break;
    }
    case REVERSE_SORTED:
    {
        for (int i = 0; i < chunkSize; i++)
        {
            chunk[i] = totalSize - (startIdx + i) - 1;
        }
        break;
    }
    case PERTURBED:
    {
        for (int i = 0; i < chunkSize; i++)
        {
            chunk[i] = startIdx + i;
        }

        int perturbCount = std::max(1, chunkSize / 100);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, chunkSize - 1);
        for (int i = 0; i < perturbCount; i++)
        {
            int idx1 = dis(gen);
            int idx2 = dis(gen);
            std::swap(chunk[idx1], chunk[idx2]);
        }
        break;
    }
    }

    /*

    // Print Out Chunks
    std::cout << "Chunk " << rank << ": ";
    for (int i = 0; i < chunkSize; i++)
    {
        std::cout << chunk[i] << " ";
    }
    std::cout << std::endl;

    */

    return chunk;
}

std::vector<int> getCount(const std::vector<int> &arr, int chunkSize, int max)
{
    std::vector<int> counts(max + 1, 0);

    for (int i = 0; i < chunkSize; i++)
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
    InputType inputType = RANDOM;

    if (argc >= 2)
    {
        arrSize = atoi(argv[1]);
    }
    else
    {
        std::cout << "\n Please provide the size of the array to be sorted" << std::endl;
        return 0;
    }

    if (argc == 3)
    {
        std::string inputTypeStr = argv[2];
        if (inputTypeStr == "sorted")
        {
            inputType = SORTED;
        }
        else if (inputTypeStr == "reverse")
        {
            inputType = REVERSE_SORTED;
        }
        else if (inputTypeStr == "perturbed")
        {
            inputType = PERTURBED;
        }
        else
        {
            inputType = RANDOM;
        }
    }

    int size, rank;

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
    adiak::value("input_type", std::string((inputType == SORTED ? "Sorted" : inputType == REVERSE_SORTED ? "Reverse Sorted"
                                                                         : inputType == PERTURBED        ? "1% Perturbed"
                                                                                                         : "Random")));
    adiak::value("num_procs", size);
    adiak::value("scalability", std::string("strong"));
    adiak::value("group_num", 5);
    adiak::value("implementation_source", std::string("handwritten"));

    MPI_Comm comm_dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_dup);

    // Data initialization
    CALI_MARK_BEGIN("data_init_runtime");
    int chunkSize = (arrSize + size - 1) / size;
    std::vector<int> localChunk = generateData(chunkSize, inputType, rank, arrSize);
    CALI_MARK_END("data_init_runtime");

    // Local counting & prefix sum
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    std::vector<int> localCounts = getCount(localChunk, chunkSize, arrSize);
    std::vector<int> localPrefixSum = prefixSum(localCounts);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Combine prefix sums
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    std::vector<int> globalPrefixSum(arrSize + 1, 0);
    MPI_Reduce(localPrefixSum.data(), globalPrefixSum.data(), arrSize + 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Master process creates final sorted array
    if (rank == MASTER)
    {
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");

        std::vector<int> result(arrSize);
        int position = 0;

        for (int value = 0; value <= arrSize; value++)
        {
            int start = globalPrefixSum[value];
            int end = (value == arrSize) ? arrSize : globalPrefixSum[value + 1];

            for (int i = start; i < end; i++)
            {
                result[i] = value;
            }
        }
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        /*

        // Print Out Result
        std::cout << "Result: ";
        for (int i = 0; i < arrSize; i++)
        {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;

        */

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