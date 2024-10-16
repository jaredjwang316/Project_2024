#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

void local_bitonic_sort(std::vector<int>& arr, int low, int count, bool ascending);
void local_bitonic_merge(std::vector<int>& arr, int low, int count, bool ascending);

void swap(int& a, int& b) { 
    int temp = a;
    a = b;
    b = temp;
}

bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

void print_arr(const std::vector<int>& arr) {
    std::cout << "[";
    for (int i = 0; i < arr.size(); i++) {
        std::cout << arr[i];
        if (i < arr.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;

    int global_size, sub_arr_size, rank, n_procs, mtype;

    if (argc == 2) {
        global_size = atoi(argv[1]);
    } else {
        std::cout << "\n Received " << argc << " arguments" << std::endl;
        std::cout << "\n Usage: ./bitonic_sort <array_size>" << std::endl;
        return 0;
    }
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    // Start Adiak and register metadata
    adiak::init(nullptr);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();  // Command line used to launch the job
    adiak::clustername();   // Name of the cluster

    // Algorithm-specific metadata
    std::string algorithm = "bitonic";
    std::string programming_model = "mpi";
    std::string data_type = "int";
    int size_of_data_type = sizeof(int);
    std::string input_type = "Random";
    std::string scalability = "strong";  
    int group_number = 5;  
    std::string implementation_source = "handwritten";  

    adiak::value("algorithm", algorithm);
    adiak::value("programming_model", programming_model);
    adiak::value("data_type", data_type);
    adiak::value("size_of_data_type", size_of_data_type);
    adiak::value("input_size", global_size);
    adiak::value("input_type", input_type);
    adiak::value("num_procs", n_procs);
    adiak::value("scalability", scalability);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    MPI_Comm comm_dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_dup);

    std::vector<int> arr;
    std::vector<int> local_arr;
    sub_arr_size = global_size / n_procs;

    CALI_MARK_BEGIN("data_init_runtime");
    if (rank == MASTER) {
        arr.resize(global_size);
        for (int i = 0; i < global_size; ++i) {
            arr[i] = rand() % 1000;
        }
    }
    local_arr.resize(sub_arr_size);
    CALI_MARK_END("data_init_runtime");

    CALI_MARK_BEGIN("comm");
        // Create barrier for synchronization
        MPI_Barrier(MPI_COMM_WORLD);

        // Distribute array across processes
        CALI_MARK_BEGIN("comm_large");
            MPI_Scatter(arr.data(), sub_arr_size, MPI_INT, local_arr.data(), sub_arr_size, MPI_INT, MASTER, MPI_COMM_WORLD);
        CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // bitonically sort local distributed arrays (ascending if rank is even)
    CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        local_bitonic_sort(local_arr, 0, local_arr.size(), rank % 2 == 0);
        CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
    
    // Gather sorted data back to master
    CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
            MPI_Gather(local_arr.data(), sub_arr_size, MPI_INT, arr.data(), sub_arr_size, MPI_INT, MASTER, MPI_COMM_WORLD);
        CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    
    if (rank == MASTER) {
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        for (int step = 2; step <= n_procs; step *= 2) {
            int pairwise_count = sub_arr_size * step;
            for (int i = 0; i < n_procs / step; i++) {
                local_bitonic_merge(arr, i * pairwise_count, pairwise_count, i % 2 == 0);
            }
        }
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        CALI_MARK_BEGIN("correctness_check");
        if (is_sorted(arr)) {
            std::cout << "Array is sorted correctly." << std::endl;
        } else {
            std::cout << "Array is NOT sorted correctly." << std::endl;
        }
        CALI_MARK_END("correctness_check");
    }

    adiak::fini();
    MPI_Finalize();
    return 0;
}

void local_bitonic_sort(std::vector<int>& arr, int low, int count, bool ascending) {
    if (count > 1) {
        int k = count / 2;
        local_bitonic_sort(arr, low, k, true);
        local_bitonic_sort(arr, low + k, k, false);
        local_bitonic_merge(arr, low, count, ascending);
    }
}

void local_bitonic_merge(std::vector<int>& arr, int low, int count, bool ascending) {
    if (count <= 1) return;
    int k = count / 2;
    for (int i = low; i < low + k; ++i) {
        if ((arr[i] > arr[i + k]) == ascending) {
            swap(arr[i], arr[i + k]);
        }
    }
    local_bitonic_merge(arr, low, k, ascending);
    local_bitonic_merge(arr, low + k, k, ascending);
}
