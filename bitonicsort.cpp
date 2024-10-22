#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
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

void initialize_sorted(std::vector<int>& arr, int n, int rank);
void initialize_random(std::vector<int>& arr, int n, int rank);
void initialize_perturbed(std::vector<int>& arr, int n, int rank, float prob);
void initialize_reverse(std::vector<int>& arr, int n, int rank, int n_proc);

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;

    int global_size, sub_arr_size, rank, n_procs, mtype;
    std::string data_init_method;

    if (argc == 3) {
        global_size = atoi(argv[1]);
        data_init_method = std::string(argv[2]);
    } else {
        std::cout << "\n Received " << argc << " arguments" << std::endl;
        std::cout << "\n Usage: ./bitonic_sort <array_size> <data_init_method>" << std::endl;
        return 0;
    }
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Status status;

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

    CALI_MARK_BEGIN("data_init_runtime");
    std::vector<int> local_arr(global_size);
    sub_arr_size = global_size / n_procs;
    if (data_init_method == "sorted") {
        initialize_sorted(local_arr, sub_arr_size, rank);
    } else if (data_init_method == "reverse") {
        initialize_reverse(local_arr, sub_arr_size, rank, n_procs);
    } else if (data_init_method == "perturbed") {
        initialize_perturbed(local_arr, sub_arr_size, rank, 0.01f);
    } else { // default to random initialization
        initialize_random(local_arr, sub_arr_size, rank);
    }
    CALI_MARK_END("data_init_runtime");

    // bitonically sort local distributed arrays (ascending if rank is even)
    CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
            local_bitonic_sort(local_arr, 0, local_arr.size(), rank % 2 == 0);
        CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
        // Create barrier for synchronization
        MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("comm");

    // iteratively and parallelly merge bitonic pairs of locally sorted subarrays
    int step = 1;
    while (step < n_procs) {
        int step_size = step * sub_arr_size;
        if (rank % (step * 2) == 0) { // receive from partner
            CALI_MARK_BEGIN("comm");
                CALI_MARK_BEGIN("comm_large");
                    MPI_Recv(local_arr.data() + step_size, step_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, &status);
                CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");

            CALI_MARK_BEGIN("comp");
                CALI_MARK_BEGIN("comp_large");
                    local_bitonic_merge(local_arr, 0, step_size * 2, rank % (step * 4) == 0);
                CALI_MARK_END("comp_large");
            CALI_MARK_END("comp");
        }
        else { // send to partner and quit
            CALI_MARK_BEGIN("comm");
                CALI_MARK_BEGIN("comm_large");
                    MPI_Send(local_arr.data(), step_size, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
                CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");
            break;
        }
        step *= 2;
    }
    
    if (rank == MASTER) {
        CALI_MARK_BEGIN("correctness_check");
        if (is_sorted(local_arr)) {
            std::cout << "Array of length " << local_arr.size() << " is sorted correctly." << std::endl;
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

void initialize_sorted(std::vector<int>& arr, int n, int rank) {
    for (int i = 0; i < n; i++) {
        arr[i] = rank * n + i;
    }
}

void initialize_random(std::vector<int>& arr, int n, int rank) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }
}

void initialize_perturbed(std::vector<int>& arr, int n, int rank, float prob) {
    for (int i = 0; i < n; i++) {
        arr[i] = rank * n + i;
    }

    int n_perturb = static_cast<int>(n * prob);

    std::mt19937 rng(10);
    std::uniform_int_distribution<int> dist(0, n - 1);

    for (size_t i = 0; i < n_perturb; i++) {
        int i1 = dist(rng);
        int i2 = dist(rng);
        std::swap(arr[i1], arr[i2]);
    }
}

void initialize_reverse(std::vector<int>& arr, int n, int rank, int n_proc) {
    for (int i = 0; i < n; i++) {
        arr[i] = (n_proc - rank) * n - i;
    }
}
