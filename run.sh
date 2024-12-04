#!/bin/bash

#SBATCH --job-name=main_train      # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=32         # Number of CPU cores per task
#SBATCH --gpus=1                   # Number of GPUs per node
#SBATCH --constraint="I|K"         # Node constraints
#SBATCH --time=4:00:00                # Time limit (4 hours)
#SBATCH --output=job_%j.log        # Standard output and error log

# Set threading environment variables
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32

# Navigate to project directory
cd /scratch/gilbreth/tnadolsk/592rli-llm-project

# Confirm setup
echo "Environment setup complete. Starting training..."

# Run the training script
/scratch/gilbreth/tnadolsk/rli_llm/rli_env/bin/python main_train_unrolled.py