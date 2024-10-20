#include <iostream>
#include <vector>
#include <string>
#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// Function to merge two sorted arrays
std::vector<int> merge(const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> merged;
    size_t i = 0, j = 0;
    
    // Merge elements from both halves
    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            merged.push_back(left[i]);
            i++;
        } else {
            merged.push_back(right[j]);
            j++;
        }
    }
    // Append any remaining elements from left or right
    while (i < left.size()) {
        merged.push_back(left[i]);
        i++;
    }
    while (j < right.size()) {
        merged.push_back(right[j]);
        j++;
    }
    
    return merged;
}

// Custom merge sort implementation
std::vector<int> merge_sort(std::vector<int>& arr) {
    if (arr.size() <= 1) {
        return arr;  // Base case
    }

    // Define pointers
    size_t mid = arr.size() / 2;  // Find the midpoint
    std::vector<int> left(arr.begin(), arr.begin() + mid);  // Left half
    std::vector<int> right(arr.begin() + mid, arr.end());   // Right half

    // Recursively sort the left and right halves
    left = merge_sort(left);
    right = merge_sort(right);

    // Merge the sorted halves back together
    return merge(left, right);
}

// Function to check if the array is sorted
bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}


int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    const int MASTER = 0;

    int sizeOfArray, array_type;

    if (argc == 3) {
        sizeOfArray = atoi(argv[1]);
        array_type = atoi(argv[2]);
    } else {
        std::cout << "Please provide the size of the array and the array type." << std::endl;
        return 1;
    }



    int numtasks, taskid;
    std::vector<int> array;
    std::vector<int> subarray;


    // Initialize MPI
    CALI_MARK_BEGIN("MPI_Init");
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    
    CALI_MARK_END("MPI_Init");  // End MPI_Init

    std::string input_type = "";
    if (taskid == MASTER) {
        array.resize(sizeOfArray);

        switch (array_type) {
            case 1:
                // Generate random array
                for (int i = 0; i < sizeOfArray; i++) {
                    array[i] = rand() % 1000;
                }
                input_type = "Random";
                break;
            case 2:
                // Generate sorted array
                for (int i = 0; i < sizeOfArray; i++) {
                    array[i] = i;
                }
                input_type = "Sorted";
                break;
            case 3:
                // Generate reverse sorted array
                for (int i = 0; i < sizeOfArray; i++) {
                    array[i] = sizeOfArray - i;
                }
                input_type = "Reverse Sorted";
                break;
            case 4:
                // Generate 1% perturbed array
                for (int i = 0; i < sizeOfArray; i++) {
                    array[i] = i;
                }
                // Swap 1% of elements
                for (int i = 0; i < sizeOfArray / 100; i++) {
                    int idx1 = rand() % sizeOfArray;
                    int idx2 = rand() % sizeOfArray;
                    std::swap(array[idx1], array[idx2]);
                }
                input_type = "1% Perturbed";
                break;
            default:
                std::cout << "Invalid array type. Please use 1 for random, 2 for sorted, 3 for reverse sorted, or 4 for 1% perturbed." << std::endl;
                return 1;
        }
        
        std::cout << "Generated " << input_type << " array of size " << sizeOfArray << std::endl;
    }


    // Start Adiak and register metadata
    adiak::init(nullptr);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();  // Command line used to launch the job
    adiak::clustername();   // Name of the cluster

    // Algorithm-specific metadata
    std::string algorithm = "merge";
    std::string programming_model = "mpi";
    std::string data_type = "int";
    int size_of_data_type = sizeof(int);
    std::string scalability = "strong";  
    int group_number = 5;  
    std::string implementation_source = "handwritten";  

    adiak::value("algorithm", algorithm);
    adiak::value("programming_model", programming_model);
    adiak::value("data_type", data_type);
    adiak::value("size_of_data_type", size_of_data_type);
    adiak::value("input_size", sizeOfArray);
    adiak::value("input_type", input_type);
    adiak::value("num_procs", numtasks);
    adiak::value("scalability", scalability);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);


    CALI_MARK_BEGIN("MPI_Comm_dup");  // Begin MPI_Comm_dup
    MPI_Comm comm_dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_dup);
    CALI_MARK_END("MPI_Comm_dup");  // End MPI_Comm_dup

    // Calculate chunk size and extra elements
    int chunk_size = sizeOfArray / numtasks;
    int remainder = sizeOfArray % numtasks;

    // Resize subarray to receive its portion of data
    int subarray_size;
    
    if (taskid < remainder)
    {
      subarray_size = chunk_size + 1;
    }
    else
    {
      subarray_size = chunk_size;
    }
    
    subarray.resize(subarray_size);

    // Data initialization runtime (assuming data initialization code here)
    CALI_MARK_BEGIN("data_init_runtime");

    if (taskid == MASTER) {
        array.resize(sizeOfArray);
        for (int i = 0; i < sizeOfArray; i++) {
            array[i] = rand() % 1000;  // Random initialization of array
        }
    }
    CALI_MARK_END("data_init_runtime");  // End data_init_runtime

    // Barrier to synchronize
    CALI_MARK_BEGIN("comm");
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("comm");

    // Prepare data for scattering: send counts and displacements
    std::vector<int> send_counts(numtasks);
    std::vector<int> displacements(numtasks);
    int offset = 0;

    for (int i = 0; i < numtasks; i++) {
        if (i < remainder)
        {
          send_counts[i] = chunk_size + 1;
        }
        else    
        {
          send_counts[i] = chunk_size;
        }
        
        displacements[i] = offset;
        offset += send_counts[i];
    }

    // Scatter data to workers with varying counts
    CALI_MARK_BEGIN("comm_large");
    MPI_Scatterv(array.data(), send_counts.data(), displacements.data(), MPI_INT, subarray.data(), subarray.size(), MPI_INT, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");

    // Computation section
    CALI_MARK_BEGIN("comp");
    subarray = merge_sort(subarray);
    CALI_MARK_END("comp");

    // Prepare receive counts and displacements for gathering
    std::vector<int> recv_counts(numtasks);
    std::vector<int> recv_displacements(numtasks);
    offset = 0;
    for (int i = 0; i < numtasks; i++) {
        recv_counts[i] = send_counts[i]; // Same as the send counts
        recv_displacements[i] = displacements[i];
    }
    
    // Resize the gathered array on the MASTER process before gathering data
    if (taskid == MASTER) {
        array.resize(sizeOfArray);
    }

    // Gather sorted data back to master
    CALI_MARK_BEGIN("comp_large");
    MPI_Gatherv(subarray.data(), subarray.size(), MPI_INT, array.data(), recv_counts.data(), recv_displacements.data(), MPI_INT, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END("comp_large");

    // Check correctness
    CALI_MARK_BEGIN("correctness_check");
    if (taskid == MASTER) {
        // The master process needs to merge the sorted subarrays into a single sorted array
        std::vector<int> sorted_array;

        // Merge the sorted subarrays back into the final sorted array
        for (int i = 0; i < numtasks; ++i) {
            // Extract the subarray that belongs to this task
            std::vector<int> current_subarray(array.begin() + recv_displacements[i], 
                                               array.begin() + recv_displacements[i] + recv_counts[i]);
            
            // Merge current subarray into the sorted_array
            sorted_array = merge(sorted_array, current_subarray);
        }

        // Now sorted_array contains the fully sorted array
        array = sorted_array;
        
        if (is_sorted(array)) {
            std::cout << "Array is sorted correctly." << std::endl;
        } else {
            std::cout << "Array is NOT sorted correctly." << std::endl;
        }
    }
    CALI_MARK_END("correctness_check");

    // Finalize MPI and Adiak
    CALI_MARK_BEGIN("MPI_Finalize");
    adiak::fini();
    MPI_Finalize();
    CALI_MARK_END("MPI_Finalize");

    return 0;
}