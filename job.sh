#!/bin/bash
#SBATCH --job-name=fluids-optimised
#SBATCH --time=00:01:00
#SBATCH --partition=1CN192C24G1H_MI300A_Ubuntu22
#SBATCH --cpus-per-task=100
#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22

module load rocm/6.3.0

export OMP_NUM_THREADS=100
# export OMP_TARGET_OFFLOAD=MANDATORY

cd "$SLURM_SUBMIT_DIR"

make clean
make
srun ./build/main 512
