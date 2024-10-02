# CSCE 435 Group project

## 0. Group number: 5

## 1. Group members:
1. Jared Wang
2. Kevin Tang
3. Aaron Matthews
4. Surya Jasper

## 2. Project topic (e.g., parallel sorting algorithms)

### 2a. Brief project description (what algorithms will you be comparing and on what architectures)

- Bitonic Sort:
- Sample Sort:
- Merge Sort: A classic divide-and-conquer sorting algorithm that can be effectively parallelized.
  We will implement an iterative version using MPI, where the array is divided among multiple processes,
  and each process sorts its sub-array before merging results.
- Radix Sort:

### 2b. Pseudocode for each parallel algorithm
- For MPI programs, include MPI calls you will use to coordinate between processes

Merge Sort Pseudocode:


  
def parallel_merge_sort(A, N):  

    MPI_Init()
    rank = MPI_Comm_rank()  # Get the rank of the current process
    size = MPI_Comm_size()   # Get the total number of processes

    # Step 3: Divide the array among processes
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



### 2c. Evaluation plan - what and how will you measure and compare
- Input sizes, Input types
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
