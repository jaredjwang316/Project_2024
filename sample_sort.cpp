#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <vector>
using namespace std;

// Sorts a recursively using quicksort
vector<int> quick_sort(vector<int> a, int n){
	// base case, array is already sorted
	if (n == 1 || n == 0){
		return a;
	}
	// Choose the rightmost element as the pivot
	int pivot = n-1;
	// elements in left will be less than or equal to a[pivot], elements in the right will be greater than a[pivot]
	vector<int> left;
	vector<int> right;

	// Since the pivot is the last elemenet, the loop only iterates until a.size() - 2 since the pivot element is in neither left nor right
	for (int i = 0; i < a.size() - 1; i++){
		if (i == pivot){
			continue;
		}
		if (a[i] <= a[pivot]){
			left.push_back(a[i]);
		}
		else{
			right.push_back(a[i]);
		}
	}
	vector<int> sorted_left = quick_sort(left, left.size());
	vector<int> sorted_right = quick_sort(right, right.size());

	sorted_left.push_back(a[pivot]);
	sorted_left.insert(sorted_left.end(), sorted_right.begin(), sorted_right.end());
	return sorted_left;
}

int main(int argc, char *argv[]){
	CALI_CXX_MARK_FUNCTION;

	int n, s, rank, num_processes;
	if (argc == 2){
		n = atoi(argv[1]);
	}
	else{
		printf("Invalid input, please enter the array size");
	}


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes); 

	cali::ConfigManager mgr;
	mgr.start();
	
	CALI_MARK_BEGIN("data_init_runtime");        
	vector<int> a;
	a.resize(n/num_processes);
	// Each process generates n/num_processes data points
	for (int i = 0; i < n/num_processes; i++){
		a[i] = rand() % n/4;
	}
	CALI_MARK_END("data_init_runtime");
	
	adiak::init(NULL);
	adiak::launchdate();    // launch date of the job
	adiak::libraries();     // Libraries used
	adiak::cmdline();       // Command line used to launch the job
	adiak::clustername();   // Name of the cluster

	adiak::value("algorithm", "merge"); // The name of the algorithm you are using (e.g., "merge", "bitonic")
	adiak::value("programming_model", "mpi"); // e.g. "mpi"
	adiak::value("data_type", "int"); // The datatype of input elements (e.g., double, int, float)
	adiak::value("size_of_data_type", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
	adiak::value("input_size", n); // The number of elements in input dataset (1000)
	adiak::value("input_type", "Random"); // For sorting, this would be choices: ("Sorted", "ReverseSorted", "Random", "1_perc_perturbed")
	adiak::value("num_procs", num_processes); // The number of processors (MPI ranks)
	adiak::value("scalability", "strong"); // The scalability of your algorithm. choices: ("strong", "weak")
	adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
	adiak::value("implementation_source", "handwritten"); // Where you got the source code of your algorithm. choices: ("online", "ai", "handwritten").

	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
	vector<int> a_sorted = quick_sort(a, a.size());
	CALI_MARK_END("comp_large");	
	CALI_MARK_END("comp");

	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
	//8 seems like a decent sample size and is also a power of 2 which is convenient.
	s = 8;
	// s samples will be drawn from the local portion of the array for each process, these local samples will eventually be combined and put into global_samples
	vector<int> local_samples;
	vector<int> global_samples;
	// Only the root process(process 0) needs to access global_samples array. It will then scatter the splitters to all processes later.
	if (rank == 0){
		global_samples.resize(s*num_processes);
	}
	// We need to sample s elements from the current process
	// It makes sense to select them in a roughly evenly distributed way once the local array is sorted
	for (int i = 0; i < a_sorted.size(); i++){
		if (i % (a_sorted.size()/s) == 0){
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
		vector<int> global_samples_sorted = quick_sort(global_samples, global_samples.size());
		// Select num_processes elements
		for (int i = 0; i < global_samples_sorted.size(); i++){
			if (i % s  == 0){
				temp.push_back(global_samples_sorted[i]);
			}
		}
		// remove first element from splitters array and we have num_processes - 1 elements
		//temp.erase(temp.begin());
		for (int i = 0; i < num_processes - 1; i++){
			splitters[i] = temp[i+1];
		}
		CALI_MARK_END("comp_small");
		CALI_MARK_END("comp");
		
		CALI_MARK_BEGIN("comm");
		for (int i = 0; i < num_processes; i++){
			CALI_MARK_BEGIN("comm_small");
			MPI_Send(splitters.data(), num_processes - 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			CALI_MARK_END("comm_small");
		}
		CALI_MARK_END("comm");
	}
	MPI_Status status;
	if (rank > 0){
		CALI_MARK_BEGIN("comm");
		CALI_MARK_BEGIN("comm_small");
		MPI_Recv(splitters.data(), num_processes - 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		CALI_MARK_END("comm_small");
		CALI_MARK_END("comm");
	}
	
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");	
	vector<vector<int>> local_buckets;
	local_buckets.resize(num_processes);
	// Could probably do binary search here to assign elemetns into buckets but speedup probably not too dramatic.
	// Each process places the local array elements into buckets
	
	for (int i = 0; i < a_sorted.size(); i++){
		if (a_sorted[i] < splitters[0]){
			local_buckets[0].push_back(a_sorted[i]);
		}
		else if (a_sorted[i] >= splitters[splitters.size()-1]){
			local_buckets[num_processes-1].push_back(a_sorted[i]);
		}		
		for (int j = 1; j < num_processes - 1; j++){
			if (a_sorted[i] >= splitters[j-1] && a_sorted[i] < splitters[j]){
				local_buckets[j].push_back(a_sorted[i]);
			}		
		}
	}
	CALI_MARK_END("comp_large");
	CALI_MARK_END("comp");
	
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_small");
	// Computing bucket sizes for process i's local buckets
	vector<int> local_bucket_sizes;
	for (int i = 0; i < num_processes; i++){
		local_bucket_sizes.push_back(local_buckets[i].size());	
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
	displacements[0] = 0;;
	for (int i = 1; i < num_processes; i++){
		displacements[i] = displacements[i-1] + local_sizes_for_current_bucket[i-1];
	}
	
	int* num_received = new int[num_processes];
	for (int i = 0; i < num_processes; i++){
		num_received[i] = local_sizes_for_current_bucket[i];
	}
	CALI_MARK_END("comp_small");
	CALI_MARK_END("comp");

	// Gather all the data in in bucket i among all processes into the process with rank i
	for (int i = 0; i < num_processes; i++){
		CALI_MARK_BEGIN("comp");
		CALI_MARK_BEGIN("comp_large");
		MPI_Gatherv(local_buckets[i].data(), local_buckets[i].size(), MPI_INT, current_bucket.data(), num_received, displacements, MPI_INT, i, MPI_COMM_WORLD);
		CALI_MARK_END("comp_large");
		CALI_MARK_END("comp");
	}

	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
	vector<int> current_bucket_sorted = quick_sort(current_bucket, current_bucket.size());
	CALI_MARK_BEGIN("comp_large");
	CALI_MARK_BEGIN("comp");	


	// MPI_Barrier();
	vector<int> global_array_sorted;
	int* global_receive_counts = new int[num_processes];
	int* global_displacements = new int[num_processes];
	
	vector<int> global_bucket_sizes;
	global_bucket_sizes.resize(num_processes);

	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_small");
	MPI_Gather(&current_bucket_size, 1, MPI_INT, global_bucket_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END("comm_small");
	CALI_MARK_END("comm");	

	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_small");
	global_displacements[0] = 0;
	for (int i = 0; i < num_processes; i++){
		global_receive_counts[i] = global_bucket_sizes[i];
	}
	for (int i = 1; i < num_processes; i++){
		global_displacements[i] = global_displacements[i-1] + global_bucket_sizes[i-1];
	}
	
	if (rank == 0){
		global_array_sorted.resize(n);
	}
	CALI_MARK_END("comp_small");
	CALI_MARK_END("comp");
	
	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_large");
	MPI_Gatherv(current_bucket_sorted.data(), current_bucket_sorted.size(), MPI_INT, global_array_sorted.data(), global_receive_counts, global_displacements, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END("comm_large");
	CALI_MARK_END("comm");

	if (rank == 0){
		// MPI_Barrier? wait till all local sorts done
		CALI_MARK_BEGIN("correctness_check");
		// Check correctness
		bool sorted = true;
		for (int i = 1; i < global_array_sorted.size(); i++){
			if (global_array_sorted[i] < global_array_sorted[i-1]){
				sorted = false;
			}
		}
		if (sorted){
			printf("Sort successful");
		}
		else{
			printf("Sort unsuccessful");
		}
		CALI_MARK_END("correctness_check");	

	}
	
	mgr.stop();
	mgr.flush();	
	MPI_Finalize();

}
