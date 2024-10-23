import sys

def generate_grace_job(algo, array_size, processes, data_init_method):
    nodes = max(1, processes // 32)
    procs_per_node = processes // nodes
    mem = 128

    job_name = f'{algo}-p{processes}-a{array_size}-t{data_init_method}'
    output_name = f'out/{algo}-p{processes}-a{array_size}-t{data_init_method}.out'

    return f'''#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE            #Do not propagate environment
#SBATCH --get-user-env=L         #Replicate login environment
#
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name={job_name}       #Set the job name to "JobName"
#SBATCH --time=02:30:00           #Set the wall clock limit
#SBATCH --nodes={nodes}               #Request nodes
#SBATCH --ntasks-per-node={procs_per_node}    # Request tasks/cores per node
#SBATCH --mem={mem}G                 #Request GB per node 
#SBATCH --output={output_name}       #Send stdout/err to "output.[jobID]" 
#
##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=email_address    #Send all emails to email_address 
#
##First Executable Line
#

module load intel/2020b       # load Intel software stack
module load CMake/3.12.1
module load GCCcore/8.3.0
module load PAPI/6.0.0            # Load PAPI (adjust version as needed)

CALI_CONFIG="spot(output=cali/{job_name}.cali, \\
    time.variance,profile.mpi)" \\
mpirun -np {processes} ./{algo}sort {array_size} {data_init_method}
'''

def generate_grace_batch_job(algo, processes, array_sizes, data_init_methods):
    nodes = max(1, processes // 32)
    procs_per_node = processes // nodes
    mem = 128

    job_name = f'{algo}-p{processes}-all'
    output_name = f'out/{job_name}.out'

    base = f'''#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE            #Do not propagate environment
#SBATCH --get-user-env=L         #Replicate login environment
#
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name={job_name}       #Set the job name to "JobName"
#SBATCH --time=00:10:00           #Set the wall clock limit
#SBATCH --nodes={nodes}               #Request nodes
#SBATCH --ntasks-per-node={procs_per_node}    # Request tasks/cores per node
#SBATCH --mem={mem}G                 #Request GB per node 
#SBATCH --output={output_name}       #Send stdout/err to "output.[jobID]" 
#
##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=email_address    #Send all emails to email_address 
#
##First Executable Line
#

module load intel/2020b       # load Intel software stack
module load CMake/3.12.1
module load GCCcore/8.3.0
module load PAPI/6.0.0            # Load PAPI (adjust version as needed)
'''
    for array_size in array_sizes:
        for data_init_method in data_init_methods:
            base += f'''
CALI_CONFIG="spot(output=cali/{algo}-p{processes}-a{array_size}-t{data_init_method}.cali, \\
    time.variance,profile.mpi)" \\
mpirun -np {processes} ./{algo}sort {array_size} {data_init_method}
'''
    return base

if __name__ == '__main__':
    algo = sys.argv[1]
    array_sizes = [int(2 ** exp) for exp in [16, 18, 20, 22, 24, 26, 28]]
    input_types = ['random', 'sorted', 'reverse', 'perturbed']
    processes = [int(2 ** exp) for exp in range(1, 11)]

    if '--batch' in sys.argv:
        for process in processes:
            job_name = f'{algo}-p{process}-all'
            print('Generating', job_name)
            grace_job = generate_grace_batch_job(algo, process, array_sizes, input_types)
            with open(f'jobs/{job_name}.grace_job', 'w') as out:
                out.write(grace_job)
    else:
        for array_size in array_sizes:
            for process in processes:
                for input_type in input_types:
                    job_name = f'{algo}-p{process}-a{array_size}-t{input_type}'
                    print('Generating', job_name)
                    grace_job = generate_grace_job(algo, array_size, process, input_type)
                    with open(f'jobs/{job_name}.grace_job', 'w') as out:
                        out.write(grace_job)
