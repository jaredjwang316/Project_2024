#include <iostream>
#include <vector>
#include <string>
#include <cstdlib> // For rand()
#include <ctime>   // For time()
#include "mpi.h"

// Function to merge two sorted arrays
std::vector<int> merge(const std::vector<int>& left, const std::vector<int>& right) {
    size_t total_size = left.size() + right.size();
    std::vector<int> merged(total_size);  // Preallocate memory

    size_t i = 0, j = 0, k = 0;

    // Merge elements from both halves
    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            merged[k] = left[i];
            i++;
        } else {
            merged[k] = right[j];
            j++;
        }
        k++;
    }

    // Append any remaining elements from left or right
    while (i < left.size()) {
        merged[k] = left[i];
        i++;
        k++;
    }
    while (j < right.size()) {
        merged[k] = right[j];
        j++;
        k++;
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
}

// Parallel merge sort function
void parallel_merge_sort(int* A, int N, int rank, int size) {
    // Step 1: Create displacements and send counts for Scatterv/Gatherv
    int* send_counts = new int[size];
    int* displacements = new int[size];

    int base_size = N / size;
    int remainder = N % size;

    // Initialize send_counts and displacements
    for (int i = 0; i < size; i++) {
        send_counts[i] = base_size + (i < remainder ? 1 : 0); // Distribute remainder
        displacements[i] = i == 0 ? 0 : displacements[i - 1] + send_counts[i - 1]; // Cumulative sum
    }

    // Step 2: Allocate space for the local array
    std::vector<int> local_array(send_counts[rank]);
    MPI_Scatterv(A, send_counts, displacements, MPI_INT, local_array.data(), send_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Step 4: Each process sorts its own sub-array
    local_array = merge_sort(local_array);

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 5: Merging step using iterative merging (recursive doubling)
    int step = 1; // Start with a step size of 1 for merging
    while (step < size) {
        
    
        if (rank % (2 * step) == 0) { // Even-ranked process
            if (rank + step < size) {
                std::vector<int> received_array(local_array.size()); // Allocate buffer for incoming data
                MPI_Recv(received_array.data(), received_array.size(), MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Merge received array
                local_array = merge(local_array, received_array);
            }
        } else {
            // Send the local array to the previous rank
            MPI_Send(local_array.data(), local_array.size(), MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2; // Double the step size for the next iteration
        
    }

    // Step 6: Gather the sorted sub-arrays back to the root process
    if (rank == 0) {
        std::copy(local_array.begin(), local_array.end(), A);
    } else {
        // Send the local sorted array back to root
        MPI_Gatherv(local_array.data(), local_array.size(), MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Cleanup
    delete[] send_counts;
    delete[] displacements;
}

int main(int argc, char** argv) {
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
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Validate input
    if (array_type < 1 || array_type > 4) {
        std::cerr << "Invalid input! Please enter a value between 1 and 4." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize data
    int* A = nullptr;
    if (rank == 0) {
        A = new int[sizeOfArray];
        generate_array(A, sizeOfArray, array_type); // Generate the array based on user input
        
    }
    
    //MPI_Barrier(MPI_COMM_WORLD);
    // Call the parallel merge sort function
    parallel_merge_sort(A, sizeOfArray, rank, size);

    // Check correctness and print the sorted array if rank is 0
    if (rank == 0) {
        if (is_sorted(A, sizeOfArray)) {
            std::cout << "Array is sorted." << std::endl;
            std::cout << "Sorted Array: ";
            for (int i = 0; i < sizeOfArray; i++) {
                std::cout << A[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Array is NOT sorted." << std::endl;
            for (int i = 0; i < sizeOfArray; i++) {
                std::cout << A[i] << " ";
            }
            std::cout << std::endl;
        }

        // Cleanup
        delete[] A; // Free allocated memory
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}