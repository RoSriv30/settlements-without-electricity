#!/bin/bash
#SBATCH --job-name=<REPLACE_ME!!>   # TODO: Name of job
#SBATCH -n 1                        # Number of tasks to run (equal to 1 cpu/core per task)
#SBATCH -N 1                        # Ensure that all cores are on one machine
#SBATCH -t 0-00:10                  # TODO: Max Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p opengpu.p                # Partition to submit to
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --mail-type=all             # Send mail when job begins and ends
#SBATCH --mail-user=<REPLACE_ME!!>  # TODO: Email address of job author
#SBATCH -o scripts/sh/slurm_files/slurm-%x-%j.out   # File to which STDOUT will be written,  %x inserts job name, %j inserts jobid
#SBATCH -e scripts/sh/slurm_files/slurm-%x-%j.err    # File to which STDERR will be written,  %x inserts job name, %j inserts jobid

# More about filename patterns @ https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E

# Printing job info
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: ${PWD}"
echo "Current node(s): ${SLURM_NODELIST}"

# Prepare the Environment
module load cuda
source ~/PATH/TO/YOUR/esdenv/bin/activate   # TODO (this may be in a different hw repo)
pip install --upgrade -r ${PWD}/requirements.txt
source .env

# Export your required environmnet variables (defined in .env file)
export WANDB_API_KEY=$WB_API_KEY
export WANDB_USERNAME=$WB_UNAME

# Debug
python --version

# TODO: Job Submission filename
# NOTE: If you are using this with sweeps.yml, replace the list of parser args with --sweep_file="${PWD}/scripts/sweeps.yml"
python -m <REPLACE_ME> <LIST OF PARSER ARGS (E.G. --model_type=SegmentationCNN --max_epochs=1 ...)>