#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <vector>
#include <algorithm>
using namespace std;

bool check_whether_sorted(vector<int>& a){
	if (a.size() <= 1){
		return true;
	}
	for (int i = 1; i < a.size(); i++){
		if (a[i] < a[i-1]){
			return false;
		}
	}
	return true;
}

int main(int argc, char *argv[]){
	CALI_CXX_MARK_FUNCTION;

	int n, s, rank, num_processes;
	string input_type;
	if (argc == 3){
		n = atoi(argv[1]);
		input_type = argv[2]; 
	}
	else{
		printf("Invalid input");
	}


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes); 

	cali::ConfigManager mgr;
	mgr.start();
	
	CALI_MARK_BEGIN("data_init_runtime");        
	double data_init_begin = MPI_Wtime();
	vector<int> a;
	a.resize(n/num_processes);
	if (input_type == "sorted"){
		for (int i = 0; i < n/num_processes; i++){
			a[i] = i + rank * n/num_processes;
		}
	}
	else if (input_type == "random"){
		// Each process generates n/num_processes data points
		for (int i = 0; i < n/num_processes; i++){
			a[i] = rand() % 10000;
		}
	}
	else if (input_type == "reverse"){
		for (int i = 0; i < n/num_processes; i++){
			a[i] = ((n - 1) - i) - rank * n/num_processes; 
		}
	}
	else if (input_type == "perturbed"){
		int num_perturbed = (n/num_processes)/100;
		for (int i = 0; i < n/num_processes; i++){
			a[i] = i + rank * n/num_processes;
		}
		for (int i = 0; i < num_perturbed; i++){
			int index1 = rand() % (n/num_processes);
			int index2 = rand() % (n/num_processes);
			int temp = a[index1];
			a[index1] = a[index2];
			a[index2] = temp; 
		}
	}
	double data_init_end = MPI_Wtime();
	double data_init_time = data_init_end - data_init_begin;
	CALI_MARK_END("data_init_runtime");
	
	if (rank == 0){
		printf("Sorting %s input of size %d with %d processors. \n", input_type.c_str(), n, num_processes); 
	}

	double min_data_init_time, max_data_init_time;
	MPI_Reduce(&data_init_time, &min_data_init_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&data_init_time, &max_data_init_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0){
		printf("Min data init time: %f \n", min_data_init_time);
		printf("Max data init time: %f \n", max_data_init_time);
	}

	adiak::init(NULL);
	adiak::launchdate();    // launch date of the job
	adiak::libraries();     // Libraries used
	adiak::cmdline();       // Command line used to launch the job
	adiak::clustername();   // Name of the cluster
	adiak::value("algorithm", "sample"); // The name of the algorithm you are using (e.g., "merge", "bitonic")
	adiak::value("programming_model", "mpi"); // e.g. "mpi"
	adiak::value("data_type", "int"); // The datatype of input elements (e.g., double, int, float)
	adiak::value("size_of_data_type", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
	adiak::value("input_size", n); // The number of elements in input dataset (1000)
	if (input_type == "sorted"){
		adiak::value("input_type", "Sorted"); // For sorting, this would be choices: ("Sorted", "ReverseSorted", "Random", "1_perc_perturbed")

	}
	else if (input_type == "random"){
		adiak::value("input_type", "Random"); // For sorting, this would be choices: ("Sorted", "ReverseSorted", "Random", "1_perc_perturbed")

	}
	else if (input_type == "reverse"){
		adiak::value("input_type", "ReverseSorted"); // For sorting, this would be choices: ("Sorted", "ReverseSorted", "Random", "1_perc_perturbed")

	}
	else if (input_type == "perturbed"){
		adiak::value("input_type", "1_perc_perturbed"); // For sorting, this would be choices: ("Sorted", "ReverseSorted", "Random", "1_perc_perturbed")
	
	}

	adiak::value("num_procs", num_processes); // The number of processors (MPI ranks)
	adiak::value("scalability", "strong"); // The scalability of your algorithm. choices: ("strong", "weak")
	adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
	adiak::value("implementation_source", "handwritten"); // Where you got the source code of your algorithm. choices: ("online", "ai", "handwritten").

	
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
	sort(a.begin(), a.end());
	CALI_MARK_END("comp_large");	
	CALI_MARK_END("comp");

			
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
	//16 seems like a decent sample size and is also a power of 2 which is convenient.
	s = 16;
	// s samples will be drawn from the local portion of the array for each process, these local samples will eventually be combined and put into global_samples
	vector<int> local_samples;
	vector<int> global_samples;
	// Only the root process(process 0) needs to access global_samples array. It will then scatter the splitters to all processes later.
	if (rank == 0){
		global_samples.resize(s*num_processes);
	}
	// We need to sample s elements from the current process
	// It makes sense to select them in a roughly evenly distributed way once the local array is sorted
	for (int i = 0; i < a.size(); i++){
		if ((i + 1) % ((a.size() - s)/(s)) == 0){
			local_samples.push_back(a[i]);		
		}
	}
	CALI_MARK_END("comp_large");
	CALI_MARK_END("comp");
	
	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_small");
	// Gather all the local samples from among the processes and combine into a vector in the root process: global_samples
	MPI_Gather(local_samples.data(), s, MPI_INT, global_samples.data(), s, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END("comm_small");
	CALI_MARK_END("comm");	

	
	vector<int> splitters;
	splitters.resize(num_processes-1);
	vector<int> temp;
	// Select splitters
	if (rank == 0){
		CALI_MARK_BEGIN("comp");
		CALI_MARK_BEGIN("comp_small");
		sort(global_samples.begin(), global_samples.end());
		int gap = (global_samples.size() - (num_processes - 1))/num_processes;
		// Want to select every gap + 1 th elemen
		
		// example: 32 processors
		// 32*8 gloabl samples
		// Want the splitters to have a roughly equal amount of elements in between them
		// 32*8 - 7 elements that are not splitters
		// divide the non splitter elements by the number of processes to see how much the gap should be roughly
		// First splitter is at index gap, next is that plus gap

		// Select num_processes elements
		int k = 0;
		for (int i = 0; i < global_samples.size(); i++){
			if ((i+1) % s  == 0){
				splitters[k] = global_samples[i];
				k++;
			}
		}
		
		CALI_MARK_END("comp_small");
		CALI_MARK_END("comp");
		
	}

	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_small");
	MPI_Bcast(splitters.data(), num_processes - 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	CALI_MARK_END("comm_small");
	CALI_MARK_END("comm");	

	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
	
	double local_bucket_placing_begin, local_bucket_placing_end, local_bucket_placing_time;
	local_bucket_placing_begin = MPI_Wtime();	
	vector<vector<int>> local_buckets;
	local_buckets.resize(num_processes);
	// Could probably do binary search here to assign elemetns into buckets but speedup probably not too dramatic.
	// Each process places the local array elements into buckets
	
	for (int i = 0; i < a.size(); i++){
		if (a[i] < splitters[0]){
			local_buckets[0].push_back(a[i]);
		}
		else if (a[i] >= splitters[splitters.size()-1]){
			local_buckets[num_processes-1].push_back(a[i]);
		}		
		for (int j = 1; j < num_processes - 1; j++){
			if (a[i] >= splitters[j-1] && a[i] < splitters[j]){
				local_buckets[j].push_back(a[i]);
			}		
		}
	}
	local_bucket_placing_end = MPI_Wtime();
	local_bucket_placing_time = local_bucket_placing_end - local_bucket_placing_begin;
	CALI_MARK_END("comp_large");
	CALI_MARK_END("comp");

	double min_local_bucket_placing_time, max_local_bucket_placing_time;
	MPI_Reduce(&local_bucket_placing_time, &min_local_bucket_placing_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&local_bucket_placing_time, &max_local_bucket_placing_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0){
		printf("Min local bucket placing time: %f \n", min_local_bucket_placing_time);
		printf("Max local bucket placing time: %f \n", max_local_bucket_placing_time);
	}
	
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_small");
	// Computing bucket sizes for process i's local buckets
	vector<int> local_bucket_sizes;
	local_bucket_sizes.resize(num_processes);
	for (int i = 0; i < num_processes; i++){
		local_bucket_sizes[i] = local_buckets[i].size();	
	}
	CALI_MARK_END("comp_small");
	CALI_MARK_END("comp");
	
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_small");
	// Eventually, process rank will become bucket rank globally
	int current_bucket_size = 0;
	for (int i = 0; i < num_processes; i++){	
		MPI_Reduce(&local_bucket_sizes[i], &current_bucket_size, 1, MPI_INT, MPI_SUM, i, MPI_COMM_WORLD);
	}
	//printf("rank %d bucket size: %d \n", rank, current_bucket_size);
	CALI_MARK_END("comp_small");
	CALI_MARK_END("comp");
	
	
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_small");
	vector<int> current_bucket;
	current_bucket.resize(current_bucket_size);
	// Want to take the contents of bucket rank from among all processes and give it to process with rank r to sort
	// At the ith location, this stores the size of bucket rank in the process with rank i
	vector<int> local_sizes_for_current_bucket;
	local_sizes_for_current_bucket.resize(num_processes);
	// Gather the number of elements in bucket i for each process to use as displacements for the next Gatherv
	// These are the sizes of the local buckets for what will go to process i
	// Find the size of bucket i in process 0, in process 1, ...., in process num_processes - 1 and send all of these sizes to process i
	// Eventually all the elements in these buckets will be sent to process i
	CALI_MARK_END("comp_small");
	CALI_MARK_END("comp");
	
	
	for (int i = 0; i < num_processes; i++){
		CALI_MARK_BEGIN("comm");
		CALI_MARK_BEGIN("comm_small");
		MPI_Gather(&local_bucket_sizes[i], 1, MPI_INT, local_sizes_for_current_bucket.data(), 1, MPI_INT, i, MPI_COMM_WORLD);
		CALI_MARK_END("comm_small");
		CALI_MARK_END("comm");
	}
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_small");
	
	int* displacements = new int[num_processes];
	displacements[0] = 0;
	for (int i = 1; i < num_processes; i++){
		displacements[i] = displacements[i-1] + local_sizes_for_current_bucket[i-1];
	}
	
	int* num_received = new int[num_processes];
	for (int i = 0; i < num_processes; i++){
		num_received[i] = local_sizes_for_current_bucket[i];
	}
	CALI_MARK_END("comp_small");
	CALI_MARK_END("comp");
	
	double gather_bucket_elements_begin, gather_bucket_elements_end, gather_bucket_elements_time;
	// Gather all the data in in bucket i among all processes into the process with rank i
	for (int i = 0; i < num_processes; i++){
		CALI_MARK_BEGIN("comp");
		CALI_MARK_BEGIN("comp_large");
		if (i == rank){
			gather_bucket_elements_begin = MPI_Wtime();
		}
		MPI_Gatherv(local_buckets[i].data(), local_buckets[i].size(), MPI_INT, current_bucket.data(), num_received, displacements, MPI_INT, i, MPI_COMM_WORLD);
		if (i == rank){
			gather_bucket_elements_end = MPI_Wtime();
			gather_bucket_elements_time = gather_bucket_elements_end - gather_bucket_elements_begin;
		}
		CALI_MARK_END("comp_large");
		CALI_MARK_END("comp");
	}

	//printf("Rank %d bucket gathering time: %f \n", rank, gather_bucket_elements_time);
	
	// For some reason, the following section causes huge increases in time for sorted and perturbed (almost sorted)
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
	double bucket_quicksort_begin = MPI_Wtime();
	sort(current_bucket.begin(), current_bucket.end());
	double bucket_quicksort_end = MPI_Wtime();
	double bucket_quicksort_time = bucket_quicksort_end - bucket_quicksort_begin;
	// printf("Rank %d bucket sort time: %f \n", rank, bucket_quicksort_time);
	CALI_MARK_END("comp_large");
	CALI_MARK_END("comp");	
	
	MPI_Barrier(MPI_COMM_WORLD);
	double bucket_quicksort_max, bucket_quicksort_min, bucket_quicksort_average;
	MPI_Reduce(&bucket_quicksort_time, &bucket_quicksort_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&bucket_quicksort_time, &bucket_quicksort_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (rank == 0){
		printf("Min bucket quicksort time: %f \n", bucket_quicksort_min);
		printf("Max bucket quicksort time: %f \n", bucket_quicksort_max);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	CALI_MARK_BEGIN("correctness_check");
	// Do correctness check locally
	bool local_piece_is_sorted = check_whether_sorted(a);
	bool all_pieces_are_sorted;
	// Reduce using AND into global variable
	MPI_Reduce(&local_piece_is_sorted, &all_pieces_are_sorted, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
	// Gather first and last element from each process into main
	// Check whether those are sorted
	// If yes, then everything is sorted
	bool last_element_check = false;
	
	if (rank == 0){
		last_element_check = true;
	}
	int most_recent_last = -1;
	for (int i = 1; i < num_processes; i++){
		if (rank == i - 1){
			// Send the last element in the array sorted by process with rank i - 1 to all other processes
			if (a.size() >= 1){
				most_recent_last = a[a.size() - 1];
			}
		}
		MPI_Bcast(&most_recent_last, 1, MPI_INT, i-1, MPI_COMM_WORLD);
		if (rank == i){
			// All processes with lower rank had empty buckets
			if (most_recent_last == -1){
				last_element_check = true;
			}
			else{
				if (a.size() >= 1){
					last_element_check = most_recent_last < a[0];
				}
				else{
					last_element_check = true;
				}
			}
		}

	}
	bool global_last_element_check;

	MPI_Reduce(&last_element_check, &global_last_element_check, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
	
	if (rank == 0){
		if (all_pieces_are_sorted && global_last_element_check){
			printf("Sort successful");
		}
		else{
			printf("Sort unsuccessful");
		}
	}
	CALI_MARK_END("correctness_check");
	
	mgr.stop();
	mgr.flush();	
	MPI_Finalize();
}
