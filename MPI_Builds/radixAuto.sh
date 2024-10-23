#!/bin/bash

processor_size="$1"
array_sizes=(65536 262144 1048576 4194304 16777216 67108864 268435456)
input_types=("random" "sorted" "reverse" "perturbed")

for size in "${array_sizes[@]}"; do
  for type in "${input_types[@]}"; do
    sbatch radix.grace_job "$size" "$processor_size" "$type"
  done
done
