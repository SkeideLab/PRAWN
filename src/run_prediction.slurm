#!/bin/bash -l
#
#SBATCH -o /u/kroma/PRAWN/logs/prediction_py_%j.out
#SBATCH -e /u/kroma/PRAWN/logs/prediction_py_%j.out
#SBATCH -D ./
#SBATCH -J prediction
#SBATCH --nodes=1             # request a full node
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=72 # was 40    # assign all the cores to that first task to make room for multithreading
#SBATCH --time=02:00:00  # EN 8h, TR: 0:30
#SBATCH --mail-user=kessler@cbs.mpg.de
#SBATCH --mail-type=ALL
##SBATCH --mem=256GB


# set number of OMP threads per process
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1 

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 sub_ses_str model netz context"
    exit 1
fi

# Access the arguments and set variables
sub_ses_str="$1"
model="$2"
net="$3"
context="$4"

conda activate prawn
conda info --envs


# Conditional logic for 'braindecode'
if [ "$model" == "braindecode_single_merge" ]; then # merged session 1 and 2
    srun python3 /u/kroma/PRAWN/src/single_braindecode_merge.py $sub_ses_str $context merge

# Conditional logic for 'time resolved'
elif [ "$model" == "timeresolved_single_merge" ]; then # merged session 1 and 2
    srun python3 /u/kroma/PRAWN/src/single_timeresolved_merge.py $sub_ses_str merge

# R1: for single session analysis despite two sessions available:
elif [ "$model" == "braindecode_single" ]; then 
    srun python3 /u/kroma/PRAWN/src/single_braindecode_merge.py $sub_ses_str $context single

# R1: for single session analysis despite two sessions available:
elif [ "$model" == "timeresolved_single" ]; then 
    srun python3 /u/kroma/PRAWN/src/single_timeresolved_merge.py $sub_ses_str single

else # TODO timeresolved single session
    echo "Unknown model: $model"
    exit 1

fi