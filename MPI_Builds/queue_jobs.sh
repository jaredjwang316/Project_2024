#!/bin/bash

algo=$1
n_procs=(2 4 8 16 32 64 128 256 512)

for n in "${n_procs[@]}"; do
  job_file="jobs/${algo}-p${n}-all.grace_job"
  if [[ -f $job_file ]]; then
    echo "Submitting $job_file"
    sbatch "$job_file"
  else
    echo "Job file $job_file does not exist, skipping..."
  fi
done
