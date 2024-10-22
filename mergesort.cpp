#include <iostream>
#include <vector>
#include <string>
#include <cstdlib> // For rand()
#include <ctime>   // For time()
#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// Function to merge two sorted arrays
std::vector<int> merge(const std::vector<int>& left, const std::vector<int>& right) {
    size_t total_size = left.size() + right.size();
    std::vector<int> merged;
    merged.resize(total_size); // Preallocate memory

    size_t i = 0, j = 0, k = 0;

    // Merge elements from both halves
    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            merged[k++] = left[i++];
        } else {
            merged[k++] = right[j++];
        }
    }

    // Append any remaining elements from left or right
    while (i < left.size()) {
        merged[k++] = left[i++];
    }
    while (j < right.size()) {
        merged[k++] = right[j++];
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
bool is_sorted(const int* arr, int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

// Function to generate an array based on user input
void generate_array(int* subarray, int sizeOfArray, int array_type) {
    srand(static_cast<unsigned int>(time(0))); // Seed random number generator
    std::string input_type = "";
    for (int i = 0; i < sizeOfArray; i++) {
        switch (array_type) {
            case 1:
                // Generate random array
                subarray[i] = rand() % 1000;
                input_type = "Random";
                break;
            case 2:
                // Generate sorted array
                subarray[i] = i; // global_index is i here
                input_type = "Sorted";
                break;
            case 3:
                // Generate reverse sorted array
                subarray[i] = sizeOfArray - i - 1; // global_index is i here
                input_type = "Reverse Sorted";
                break;
            case 4:
                // Generate 1% perturbed array
                subarray[i] = i; // Initially sorted
                if (rand() % 100 == 0) { // 1% chance to swap
                    int idx_to_swap = rand() % sizeOfArray;
                    std::swap(subarray[i], subarray[idx_to_swap]);
                }
                input_type = "1% Perturbed";
                break;
            default:
                std::cerr << "Invalid array type. Please use 1 for random, 2 for sorted, "
                          << "3 for reverse sorted, or 4 for 1% perturbed." << std::endl;
                exit(1); // Exit if invalid type
        }
    }
    adiak::value("input_type", input_type);
}

// Parallel merge sort function
void parallel_merge_sort(int* A, int N, int rank, int size) {
    // Allocate space for the local array
    std::vector<int> local_array(A, A + N);
    
    // Each process sorts its own sub-array
    CALI_MARK_BEGIN("comp_large"); // Start of computation region
    local_array = merge_sort(local_array);
    CALI_MARK_END("comp_large"); // End of computation region

    // Merging step using iterative merging (recursive doubling)
    int step = 1; // Start with a step size of 1 for merging
    while (step < size) {
        if (rank % (2 * step) == 0) { // Even-ranked process
            if (rank + step < size) {
                int received_size = N / size; // Size of the received array
                std::vector<int> received_array(received_size);

                // Use comm_large for larger communication
                CALI_MARK_BEGIN("comm_large"); // Start of large communication region
                MPI_Recv(received_array.data(), received_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END("comm_large"); // End of large communication region
                
                // Merge the received array
                CALI_MARK_BEGIN("comp"); // Start of merge computation
                local_array = merge(local_array, received_array);
                CALI_MARK_END("comp"); // End of merge computation
            }
        } else {
            // Use comm_large for larger communication
            CALI_MARK_BEGIN("comm_large"); // Start of large communication region

            MPI_Send(local_array.data(), N / size, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            CALI_MARK_END("comm_large"); // End of large communication region
            break; // Exit the loop after sending the data
        }
        step *= 2; // Double the step size for the next iteration
    }
}

int main(int argc, char** argv) {

    
    // Mark the function entry for Caliper
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

    int rank, size;
    CALI_MARK_BEGIN("MPI_Init");
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    CALI_MARK_END("MPI_Init");


    // Start Adiak and register metadata
    adiak::init(nullptr);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();  // Command line used to launch the job
    adiak::clustername();   // Name of the cluster

    // Validate input
    if (array_type < 1 || array_type > 4) {
        std::cerr << "Invalid input! Please enter a value between 1 and 4." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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

    // Initialize data
    int* A = nullptr;
    if (rank == 0) {
        CALI_MARK_BEGIN("data_init_runtime"); // Start data initialization region
        A = new int[sizeOfArray];
        generate_array(A, sizeOfArray, array_type); // Generate the array based on user input
        CALI_MARK_END("data_init_runtime"); // End data initialization region
    }

    adiak::value("num_procs", size);
    adiak::value("scalability", scalability);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    CALI_MARK_BEGIN("MPI_Comm_dup");  // Begin MPI_Comm_dup
    MPI_Comm comm_dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_dup);
    CALI_MARK_END("MPI_Comm_dup");  // End MPI_Comm_dup

    // Barrier to synchronize
    CALI_MARK_BEGIN("comm");
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("comm");


    // Calculate sub-array size and allocate space for local sub-array
    int sub_array_size = sizeOfArray / size;
    int* local_array = new int[sub_array_size];

    // Scatter the data across all processes
    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(A, sub_array_size, MPI_INT, local_array, sub_array_size, MPI_INT, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");

    // Call the parallel merge sort function
    parallel_merge_sort(local_array, sub_array_size, rank, size);

    // Gather the sorted sub-arrays back to the root process
    int* sorted_array = nullptr;
    if (rank == 0) {
        sorted_array = new int[sizeOfArray];
    }

    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(local_array, sub_array_size, MPI_INT, sorted_array, sub_array_size, MPI_INT, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    
    // Communication - gather the sorted arrays (final merge)
    if (rank == 0) {
        // Check correctness
        CALI_MARK_BEGIN("correctness_check"); // Start correctness check region
        if (is_sorted(sorted_array, sizeOfArray)) {
            std::cout << "Array is sorted." << std::endl;
        } else {
            std::cout << "Array is NOT sorted." << std::endl;
        }
        CALI_MARK_END("correctness_check"); // End correctness check region

        // Cleanup
        delete[] A; // Free allocated memory
        delete[] sorted_array;
    }

    delete[] local_array; // Free local array memory

    CALI_MARK_BEGIN("MPI_Finalize");
    adiak::fini();
    MPI_Finalize(); // Finalize MPI
    CALI_MARK_END("MPI_Finalize");
    return 0;
}
