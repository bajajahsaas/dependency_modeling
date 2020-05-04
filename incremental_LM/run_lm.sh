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


export MODEL_PATH=/mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/models/transfo-xl-wt103/

tgts=(8 16 32 64 128 256 512 1024)
for tgt in "${tgts[@]}"
do
    export TGT_LEN=$tgt
    python run_lm.py --model_type transfo-xl --model_name_or_path ${MODEL_PATH} --block_size ${TGT_LEN} --do_eval --eval_data_file ../../data/wikitext-103/test.txt --line_by_line --output_dir outlm
    # python run_lm.py --model_type gpt2 --model_name_or_path gpt2 --block_size ${TGT_LEN} --do_eval --eval_data_file ../../data/wikitext-103/test.txt --line_by_line --output_dir outlm
    # python run_lm.py --model_type xlnet --model_name_or_path xlnet-base-cased --block_size ${TGT_LEN} --do_eval --eval_data_file ../../data/wikitext-103/test.txt --line_by_line --output_dir outlm
    # python run_lm.py --model_type bert --model_name_or_path bert-base-cased --mlm --block_size ${TGT_LEN} --do_eval --eval_data_file ../../data/wikitext-103/test.txt --line_by_line --output_dir outlm
    # python run_lm.py --model_type roberta --model_name_or_path roberta-base --mlm --block_size ${TGT_LEN} --do_eval --eval_data_file ../../data/wikitext-103/test.txt --line_by_line --output_dir outlm

done