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
    # Initialize MPI
    MPI_Init()

    rank = MPI_Comm_rank()
    size = MPI_Comm_size()

    # Step 3: Divide the array
    if rank == 0:
        sub_array_size = N // size
        for i in range(1, size):
            MPI_Send(A[i * sub_array_size:(i + 1) * sub_array_size], dest=i)
        local_array = A[0:sub_array_size]
    else:
        local_array = MPI_Recv(source=0)

    # Custom merge sort on the sub-array
    merge_sort(local_array)

    # Merging step using iterative merging (recursive doubling)
    step = 1
    while step < size:
        if rank % (2 * step) == 0:
            if rank + step < size:
                received_array = MPI_Recv(source=rank + step)
                local_array = merge(local_array, received_array)
        else:
            MPI_Send(local_array, dest=rank - step)
            break
        step *= 2

    # Finalize MPI
    MPI_Finalize()

    def merge_sort(arr):
    # Custom merge sort implementation
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        merge_sort(left)
        merge_sort(right)

        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1


### 2c. Evaluation plan - what and how will you measure and compare
- Input sizes, Input types
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
