#!/bin/bash

algo=$1
batch_mode=0

# check if --batch is specified
if [[ "$2" == "--batch" ]]; then
  batch_mode=1
fi

n_procs=(2 4 8 16 32 64 128 256 512 1024)
array_size=(65536 262144 1048576 4194304 16777216 67108864 268435456)
input_type=("random" "sorted" "reverse" "perturbed")

if [[ $batch_mode -eq 1 ]]; then
  for n in "${n_procs[@]}"; do
    job_file="jobs/${algo}-p${n}-all.grace_job"
    if [[ -f $job_file ]]; then
      echo "Submitting $job_file"
      sbatch "$job_file"
    else
      echo "Job file $job_file does not exist, skipping..."
    fi
  done
else
  for n in "${n_procs[@]}"; do
    for size in "${array_size[@]}"; do
      for type in "${input_type[@]}"; do
        job_file="jobs/${algo}-p${n}-a${size}-t${type}.grace_job"
        if [[ -f $job_file ]]; then
          echo "Submitting $job_file"
          sbatch "$job_file"
        else
          echo "Job file $job_file does not exist, skipping..."
        fi
      done
    done
  done
fi