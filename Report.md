# CSCE 435 Group project

## 0. Group number: 5

## 1. Group members:
1. Jared Wang
2. Kevin Tang
3. Aaron Matthews
4. Surya Jasper

## 2. Project topic (e.g., parallel sorting algorithms)

### 2a. Brief project description (what algorithms will you be comparing and on what architectures)

- Bitonic Sort: SPMD (CPU only)
- Sample Sort:
- Merge Sort:  SPMD (CPU only)  
- Radix Sort:

### 2b. Pseudocode for each parallel algorithm
- For MPI programs, include MPI calls you will use to coordinate between processes

Merge Sort Pseudocode:

    def parallel_merge_sort(A, N):  
      MPI_Init()
      rank = MPI_Comm_rank()  # Get the rank of the current process
      size = MPI_Comm_size()   # Get the total number of processes
  
      # Divide the array among processes
      if rank == 0:
          sub_array_size = N // size  # Calculate the size of each sub-array
          # Send portions of the array to each process
          for i in range(1, size):
              # Sending sub-arrays to other processes
              MPI_Send(A[i * sub_array_size:(i + 1) * sub_array_size], dest=i)
          local_array = A[0:sub_array_size]  # Root keeps its portion of the array
      else:
          # Receive the assigned sub-array for non-root processes
          local_array = MPI_Recv(source=0)
  
      # Custom merge sort on the sub-array
      merge_sort(local_array)  # Each process sorts its own sub-array
  
      # Merging step using iterative merging (recursive doubling)
      step = 1  # Start with a step size of 1 for merging
      while step < size:  # Continue until the step size is greater than the number of processes
          if rank % (2 * step) == 0:  # Check if the process is an even-ranked process for merging
              if rank + step < size:  # Ensure there is a process to receive the data
                  # Receive the sorted sub-array from the partner process
                  received_array = MPI_Recv(source=rank + step)
                  # Merge the local sorted array with the received array
                  local_array = merge(local_array, received_array)
          else:
              # Send the local sorted array to the partner process
              MPI_Send(local_array, dest=rank - step)
              break  # Exit the loop after sending the data
          step *= 2  # Double the step size for the next iteration
  
      # Finalize MPI
      MPI_Finalize()  # Clean up the MPI environment before exiting

    def merge_sort(arr): 
       # Custom merge sort implementation 
       if len(arr) > 1:
          #define pointers
          mid = len(arr) // 2   # Find the midpoint of the array
          left = arr[:mid]      # Split the array into two halves
          right = arr[mid:]

          # Recursively sort the left and right halves
          merge_sort(left)
          merge_sort(right)

          # Merging the sorted halves back together
          i = j = k = 0  # Initialize indices for left, right, and merged arrays
          while i < len(left) and j < len(right):
              if left[i] < right[j]:  # Compare elements from both halves
                  arr[k] = left[i]     # Add the smaller element to the merged array
                  i += 1
              else:
                  arr[k] = right[j]    # Add the smaller element to the merged array
                  j += 1
              k += 1  # Move to the next position in the merged array

          # Add any remaining elements from the left half
          while i < len(left):
              arr[k] = left[i]
              i += 1
              k += 1

          # Add any remaining elements from the right half
          while j < len(right):
              arr[k] = right[j]
              j += 1
              k += 1

      return arr

Bitonic Sort Pseudocode:
```python
def parallel_bitonic_sort(arr):
    MPI_Init()
    rank = MPI_Comm_rank()   # Get the rank of the current process
    n_procs = MPI_Comm_size()   # Get the total number of processes

    # Divide array for parallel sorting
    local_arr = []
    if rank == 0:   
        # Divide & dispatch subarrays in host process  
        sub_arr_size = len(arr) // n_procs
        local_arr = arr[:sub_arr_size]
        for i in range(1, n_procs):
            worker_arr = arr[i * sub_arr_size : (i + 1) * sub_arr_size]
            MPI_Send(worker_arr, dest=i)
    else:   
        # Receive local subarray from host process in worker process
        local_arr = MPI_Recv(source=0)

    # Sort local subarrays into bitonically alternating pairs (ascending-descending-ascending-etc.) 
    if rank % 2 == 0:
        local_bitonic_sort(local_arr, 0, len(local_arr), ascending=True)
    else:
        local_bitonic_sort(local_arr, 0, len(local_arr), ascending=False)

    # Iteratively & parallelly merge bitonic pairs of locally sorted subarrays
    step = 1
    while step < n_procs:
        if rank % (step * 2) == 0:
            # Receive data from partner process and bitonic merge sort into local array
            recv_arr = MPI_Recv(source=rank + step)
            local_arr.extend(recv_arr)
            local_bitonic_merge(local_arr, 0, len(local_arr), True)
        else:
            # Send local array to process waiting to merge and exit
            MPI_Send(local_arr, dest=rank - step)
            break
        step = step * 2 # Double step for next iteration

    # Host process now contains fully-sorted array
    if rank == 0:
        print(local_arr)

    MPI_Finalize()

def local_bitonic_sort(arr, low, count, ascending):
    if count > 1:
        k = count // 2
        local_bitonic_sort(arr, low, k, True)        # Sort first half in ascending order
        local_bitonic_sort(arr, low + k, k, False)    # Sort second half in descending order
        local_bitonic_merge(arr, low, count, ascending) # Bitonic merge sorted halves

def local_bitonic_merge(arr, low, count, ascending):
    if count <= 1:
        return  # Break recursion on base case
    k = count // 2
    # Iteratively swap such that all elements in the first half are:
    #   - less than all elements in the second half if ascending
    #   - greater than all elements in the second half if descending
    for i in range(low, low + k):
        if arr[i] > arr[k + i] == ascending:
            swap(arr[i], arr[k + i])
    # Recursively merge each half to sort
    local_bitonic_merge(arr, low, k, ascending)
    local_bitonic_merge(arr, low + k, k, ascending)
```

### 2c. Evaluation plan - what and how will you measure and compare
- Input sizes, Input types
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)

1. Input Sizes:

    - Small Input: 10,000 elements
    - Medium Input: 100,000 elements
    - Large Input: 1,000,000 elements

These input sizes allow for testing with varying workloads and evaluating the performance of parallel merge sort under different data sizes.

2. Input Types:

    - Random Integers: Arrays of random integers, which mimic typical use cases of merge sort.
    - Edge Cases: Arrays that are already sorted (ascending and descending), arrays with repeated elements, and small arrays (smaller or equal to the number of processors) to assess how the algorithm performs under edge conditions.

3. Scaling Tests:
   
  Strong Scaling:
      
      - Goal: Measure how execution time changes when increasing the number of processors while keeping the problem size fixed.
      - Processors: Use processor counts of 2, 4, 8, 16, 32, 64, and 128.
      - Input: Fixed-size arrays, such as 1,000,000 elements.
      - Measurement: The primary metric is execution time (time to complete sorting) for each processor count.
      - Expected Outcome: As the number of processors increases, the execution time should decrease, ideally showing faster sorting with more processors. However, at higher processor counts, communication overhead might offset the gains from parallelism, leading to diminishing returns.

5. Output Graph:

    - For each input size, generate a graph with execution time vs. number of processors.
    - The y-axis will represent execution time, and the x-axis will represent the number of processors. P
    - Plot 3 graphs (one for each input size) with lines representing each input size’s performance with varying processor counts.
