#!/bin/bash
#
#SBATCH --job-name=xl
#SBATCH --output=logsxl/run_%j.txt  # output file
#SBATCH -e logsxl/run_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

# python test.py
python run_transfo_xl.py --work_dir logsxl