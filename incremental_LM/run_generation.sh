#!/bin/bash
#
#SBATCH --job-name=gen
#SBATCH --output=logsgen/run_%j.txt  # output file
#SBATCH -e logsgen/run_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

python run_generation.py  --model_type=gpt2 --length=20 --model_name_or_path=gpt2