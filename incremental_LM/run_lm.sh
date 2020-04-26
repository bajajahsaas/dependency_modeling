#!/bin/bash
#
#SBATCH --job-name=lm
#SBATCH --output=logslm/run_%j.txt  # output file
#SBATCH -e logslm/run_%j.err        # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --mem=40GB
#
#SBATCH --ntasks=1

python run_lm.py --model_type transfo-xl --model_name_or_path ./model/ --do_eval --eval_data_file ../../data/wikitext-103/valid.txt --line_by_line --output_dir logslm