#!#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --job-name=deepseek_train
#SBATCH --mail-type=ALL
#SBATCH --partition=cuda

###### 1 Load the module
module load nvidia/cudasdk/10.1
module load intel/python/3

# 2 Activate the virtual environment 
source luxia/bin/activate

# 3 Run the python script
python3 main.py --savedir legion_erfnet_maxent 
