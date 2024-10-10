#include <iostream>
#include <vector>
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
std::vector<int> merge_sort(std::vector<int> arr) {
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

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    const int MASTER = 0;
    const int FROM_MASTER = 1;
    const int FROM_WORKER = 2;

    int sizeOfArray;
    if (argc == 2) {
        sizeOfArray = atoi(argv[1]);
    } else {
        std::cout << "\n Please provide the size of the array to be sorted" << std::endl;
        return 0;
    }

    int numtasks, taskid, numworkers, source, dest, mtype;
    int averow, extra, offset, i, rc;
    std::vector<int> array;
    std::vector<int> subarray;
    std::vector<int> sorted_array;
    MPI_Status status;
    double worker_receive_time, worker_calculation_time, worker_send_time = 0;
    double whole_computation_time, master_initialization_time, master_send_receive_time = 0;
    const char* whole_computation = "whole_computation";
    const char* master_initialization = "master_initialization";
    const char* master_send_receive = "master_send_recieve";
    const char* worker_receive = "worker_recieve";
    const char* worker_calculation = "worker_calculation";
    const char* worker_send = "worker_send";

    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2) {
        std::cout << "Need at least two MPI tasks. Quitting..." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    numworkers = numtasks - 1;

    MPI_Comm workers_comm;
    int color;  // Defines the color for the MPI_Comm_split
    int key;    // Defines the rank order within the new communicator
    
    if (taskid == MASTER) {
        color = MPI_UNDEFINED;  // Exclude the master from the new communicator
    } else {
        color = 1;  // All workers will have the same color
    }
    key = taskid;  // Use task ID to define the rank ordering within the new communicator
    
    // Create a new communicator with the workers
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &workers_comm);

    // Timing and profiling initialization
    double whole_computation_start = MPI_Wtime();
    CALI_MARK_BEGIN(whole_computation);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start(); // Start profiling

    /**************************** master task ************************************/
    if (taskid == MASTER) {
        double master_init_start = MPI_Wtime();
        CALI_MARK_BEGIN(master_initialization);

        // Allocate and initialize the array
        array.resize(sizeOfArray);
        for (i = 0; i < sizeOfArray; i++) {
            array[i] = rand() % 100; // Initialize with random values
        }

        double master_init_end = MPI_Wtime();
        master_initialization_time = master_init_end - master_init_start;
        CALI_MARK_END(master_initialization);

        double master_send_receive_start = MPI_Wtime();
        CALI_MARK_BEGIN(master_send_receive);

        // Distribute array segments to worker tasks
        averow = sizeOfArray / numworkers;
        extra = sizeOfArray % numworkers;
        offset = 0;
        mtype = FROM_MASTER;

        for (dest = 1; dest <= numworkers; dest++) {
            int elements = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&elements, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(array.data() + offset, elements, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset += elements;
        }

        // Collect sorted segments from worker tasks
        sorted_array.resize(sizeOfArray);
        offset = 0;
        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++) {
            int elements;
            MPI_Recv(&elements, 1, MPI_INT, i, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(sorted_array.data() + offset, elements, MPI_INT, i, mtype, MPI_COMM_WORLD, &status);
            offset += elements;
        }

        double master_send_receive_end = MPI_Wtime();
        master_send_receive_time = master_send_receive_end - master_send_receive_start;
        CALI_MARK_END(master_send_receive);

        // Merge the sorted segments
        sorted_array = merge_sort(sorted_array);  // Use the correct merge_sort call here

        // Optionally print the sorted array
        /*
        std::cout << "Sorted Array:" << std::endl;
        for (i = 0; i < sizeOfArray; i++) {
            std::cout << sorted_array[i] << " ";
        }
        std::cout << std::endl;
        */

    }

    /**************************** worker task ************************************/
    if (taskid > MASTER) {
        double worker_receive_start = MPI_Wtime();
        CALI_MARK_BEGIN(worker_receive);

        int elements;
        mtype = FROM_MASTER;
        MPI_Recv(&elements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        subarray.resize(elements);
        MPI_Recv(subarray.data(), elements, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        double worker_receive_end = MPI_Wtime();
        worker_receive_time = worker_receive_end - worker_receive_start;
        CALI_MARK_END(worker_receive);

        double worker_calculation_start = MPI_Wtime();
        CALI_MARK_BEGIN(worker_calculation);

        // Sort the subarray using merge sort
        subarray = merge_sort(subarray);  

        double worker_calculation_end = MPI_Wtime();
        worker_calculation_time = worker_calculation_end - worker_calculation_start;
        CALI_MARK_END(worker_calculation);

        double worker_send_start = MPI_Wtime();
        CALI_MARK_BEGIN(worker_send);

        mtype = FROM_WORKER;
        MPI_Send(&elements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(subarray.data(), elements, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);

        double worker_send_end = MPI_Wtime();
        worker_send_time = worker_send_end - worker_send_start;
        CALI_MARK_END(worker_send);

    }

    double whole_computation_end = MPI_Wtime();
    whole_computation_time = whole_computation_end - whole_computation_start;
    CALI_MARK_END(whole_computation);


    double worker_receive_time_max, worker_receive_time_min, worker_receive_time_sum, worker_receive_time_average;
    double worker_calculation_time_max, worker_calculation_time_min, worker_calculation_time_sum, worker_calculation_time_average;
    double worker_send_time_max, worker_send_time_min, worker_send_time_sum, worker_send_time_average;

    // Initialize Adiak
    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_procs", numtasks);
    adiak::value("program_name", "mpi_merge_sort");


    // Use MPI_Reduce to calculate statistics for worker processes
    if (workers_comm != MPI_COMM_NULL) {
        MPI_Reduce(&worker_receive_time, &worker_receive_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, workers_comm);
        MPI_Reduce(&worker_receive_time, &worker_receive_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, workers_comm);
        MPI_Reduce(&worker_receive_time, &worker_receive_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, workers_comm);
        
        MPI_Reduce(&worker_send_time, &worker_send_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, workers_comm);
        MPI_Reduce(&worker_send_time, &worker_send_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, workers_comm);
        MPI_Reduce(&worker_send_time, &worker_send_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, workers_comm);
        
        MPI_Reduce(&worker_calculation_time, &worker_calculation_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, workers_comm);
        MPI_Reduce(&worker_calculation_time, &worker_calculation_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, workers_comm);
        MPI_Reduce(&worker_calculation_time, &worker_calculation_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, workers_comm);     
    }

    if (taskid == 0) {
        // Master process reports timing information
        printf("******************************************************\n");
        printf("Master Times:\n");
        printf("Whole Computation Time: %f \n", whole_computation_time);
        printf("Master Initialization Time: %f \n", master_initialization_time);
        printf("Master Send and Receive Time: %f \n", master_send_receive_time);
        printf("\n******************************************************\n");

        // Add values to Adiak
        adiak::value("MPI_Reduce-whole_computation_time", whole_computation_time);
        adiak::value("MPI_Reduce-master_initialization_time", master_initialization_time);
        adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);

        // Receive statistics from the first worker
        MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_receive_time_average, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_calculation_time_average, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_send_time_average, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Log worker statistics
        adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
        adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
        adiak::value("MPI_Reduce-worker_receive_time_average", worker_receive_time_average);
        adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
        adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
        adiak::value("MPI_Reduce-worker_calculation_time_average", worker_calculation_time_average);
        adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
        adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
        adiak::value("MPI_Reduce-worker_send_time_average", worker_send_time_average);
    } else if (taskid == 1) {
        // Worker process calculates and sends statistics to the master
        worker_receive_time_average = worker_receive_time_sum / (double)numworkers;
        worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;
        worker_send_time_average = worker_send_time_sum / (double)numworkers;

        printf("******************************************************\n");
        printf("Worker Times:\n");
        printf("Worker Receive Time Max: %f \n", worker_receive_time_max);
        printf("Worker Receive Time Min: %f \n", worker_receive_time_min);
        printf("Worker Receive Time Average: %f \n", worker_receive_time_average);
        printf("Worker Calculation Time Max: %f \n", worker_calculation_time_max);
        printf("Worker Calculation Time Min: %f \n", worker_calculation_time_min);
        printf("Worker Calculation Time Average: %f \n", worker_calculation_time_average);
        printf("Worker Send Time Max: %f \n", worker_send_time_max);
        printf("Worker Send Time Min: %f \n", worker_send_time_min);
        printf("Worker Send Time Average: %f \n", worker_send_time_average);
        printf("\n******************************************************\n");

        // Send the computed statistics back to the master
        MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_receive_time_average, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_calculation_time_max, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_calculation_time_min, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_calculation_time_average, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_send_time_max, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_send_time_min, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_send_time_average, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    // Finalize MPI
    MPI_Finalize();
    return 0;
}