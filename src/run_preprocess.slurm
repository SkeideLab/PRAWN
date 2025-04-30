#!/bin/bash -l
#
# Multithreading example job script for MPCDF Cobra.
# In addition to the Python example shown here, the script
# is valid for any multi-threaded program, including
# plain OpenMP, parallel Matlab, Julia, and similar cases.
#
#SBATCH -o /u/kroma/PRAWN/logs/preprocess_py_%j.out
#SBATCH -e /u/kroma/PRAWN/logs/preprocess_py_%j.out
#SBATCH -D ./
#SBATCH -J preprocess
#SBATCH --nodes=1             # request a full node
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=72    # assign all the cores to that first task to make room for multithreading
#SBATCH --time=01:00:00
#SBATCH --mail-user=kessler@cbs.mpg.de
#SBATCH --mail-type=ALL

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 sub_ses_str"
    exit 1
fi

# Access the arguments and set variables
sub_ses_str="$1"

conda activate prawn
conda info --envs
cd /u/kroma/PRAWN
srun python3 /u/kroma/PRAWN/src/preprocess.py $sub_ses_str

