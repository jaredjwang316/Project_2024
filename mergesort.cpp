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
bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

// Function to generate an array based on user input
void generate_array(int* subarray, int sizeOfArray, int array_type) {
    srand(static_cast<unsigned int>(time(0))); // Seed random number generator
    for (int i = 0; i < sizeOfArray; i++) {
        switch (array_type) {
            case 1:
                // Generate random array
                subarray[i] = rand() % 1000;
                break;
            case 2:
                // Generate sorted array
                subarray[i] = i; // global_index is i here
                break;
            case 3:
                // Generate reverse sorted array
                subarray[i] = sizeOfArray - i - 1; // global_index is i here
                break;
            case 4:
                // Generate 1% perturbed array
                subarray[i] = i; // Initially sorted
                if (rand() % 100 == 0) { // 1% chance to swap
                    int idx_to_swap = rand() % sizeOfArray;
                    std::swap(subarray[i], subarray[idx_to_swap]);
                }
                break;
            default:
                std::cerr << "Invalid array type. Please use 1 for random, 2 for sorted, "
                          << "3 for reverse sorted, or 4 for 1% perturbed." << std::endl;
                exit(1); // Exit if invalid type
        }
    }
}

// Parallel merge sort function
void parallel_merge_sort(int* A, int N, int rank, int size) {
    // Allocate space for the local array
    std::vector<int> local_array(A, A + N);
    
    // Each process sorts its own sub-array
    CALI_MARK_BEGIN("comp_small"); // Start of the computation region
    local_array = merge_sort(local_array);
    CALI_MARK_END("comp_small"); // End of the computation region

    // Merging step using iterative merging (recursive doubling)
    int step = 1; // Start with a step size of 1 for merging
    while (step < size) {
        if (rank % (2 * step) == 0) { // Even-ranked process
            if (rank + step < size) {
                int received_size = N / size; // Size of the received array
                std::vector<int> received_array(received_size);
                CALI_MARK_BEGIN("comm_large"); // Start of large communication region
                MPI_Recv(received_array.data(), received_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END("comm_large"); // End of large communication region
                local_array = merge(local_array, received_array);
            }
        } else {
            CALI_MARK_BEGIN("comm_small"); // Start of small communication region
            MPI_Send(local_array.data(), N / size, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            CALI_MARK_END("comm_small"); // End of small communication region
            break; // Exit the loop after sending the data
        }
        step *= 2; // Double the step size for the next iteration
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    // Mark the function entry for Caliper
    CALI_CXX_MARK_FUNCTION;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Input: Specify the total number of elements and array type
    int N = 100; // Total number of elements, can be replaced by actual input
    int array_type = 0; // 1 for random, 2 for sorted, 3 for reverse sorted, 4 for 1% perturbed

    if (rank == 0) {
        std::cout << "Enter the array type (1: Random, 2: Sorted, 3: Reverse Sorted, 4: 1% Perturbed): ";
        std::cin >> array_type;

        // Validate input
        if (array_type < 1 || array_type > 4) {
            std::cerr << "Invalid input! Please enter a value between 1 and 4." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast the selected array type to all processes
    MPI_Bcast(&array_type, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Initialize data
    int* A = nullptr;

    if (rank == 0) {
        CALI_MARK_BEGIN("data_init"); // Start data initialization region
        A = new int[N];
        generate_array(A, N, array_type); // Generate the array based on user input
        CALI_MARK_END("data_init"); // End data initialization region
    }

    // Broadcast the size of the array to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate sub-array size and allocate space for local sub-array
    int sub_array_size = N / size;
    int* local_array = new int[sub_array_size];

    // Communication - distribute the array
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            CALI_MARK_BEGIN("comm_large"); // Start of large communication region
            MPI_Send(A + i * sub_array_size, sub_array_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            CALI_MARK_END("comm_large"); // End of large communication region
        }
        std::copy(A, A + sub_array_size, local_array);
    } else {
        CALI_MARK_BEGIN("comm_small"); // Start of small communication region
        MPI_Recv(local_array, sub_array_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END("comm_small"); // End of small communication region
    }

    // Call the parallel merge sort function
    parallel_merge_sort(local_array, sub_array_size, rank, size);

    // Communication - gather the sorted arrays (final merge)
    if (rank == 0) {
        std::vector<int> sorted_array(N);
        std::copy(local_array, local_array + sub_array_size, sorted_array.begin());

        for (int i = 1; i < size; i++) {
            CALI_MARK_BEGIN("comm_large"); // Start of large communication region
            MPI_Recv(sorted_array.data() + i * sub_array_size, sub_array_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CALI_MARK_END("comm_large"); // End of large communication region
        }

        // Final merge on the root process (optional, depending on your setup)
        // sorted_array = merge_sort(sorted_array); // Uncomment if a final merge is needed
        
        // Check correctness
        CALI_MARK_BEGIN("correctness_check"); // Start correctness check region
        if (is_sorted(sorted_array)) {
            std::cout << "Array is sorted." << std::endl;
        } else {
            std::cout << "Array is NOT sorted." << std::endl;
        }
        CALI_MARK_END("correctness_check"); // End correctness check region

        // Cleanup
        delete[] A; // Free allocated memory
    }

    delete[] local_array; // Free local array memory

    MPI_Finalize(); // Finalize MPI
    return 0;
}
