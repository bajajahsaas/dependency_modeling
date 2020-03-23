#!/bin/bash

export MODEL_NAME_OR_PATH=roberta-base #xlnet-base-cased #/mnt/nfs/work1/miyyer/tuvu/torch/xlm-mlm-en-2048 #
# bert-base-cased, bert-large-cased,
# roberta-base, roberta-large,
# xlnet-base-cased, xlnet-large-cased
# xlm-mlm-en-2048
#export MODEL_NAME_OR_PATH=/Users/tutvu/Downloads/spanbert_hf_base
export DO_LOWER_CASE=False
export CACHE_DIR=../cache
export MAX_CONTEXT_SIZE=249
export MAX_SEQUENCE_LENGTH=512
export SPAN_LENGTH=1
export MAX_SPAN_LENGTH=1
export MAX_NUM_EXAMPLES=100000 #3000
export MODEL_NAME=roberta # bert, xlnet, xlm, roberta
export DATA_NAME=aclImdb
export TAGS=FREQ,INFREQ,VERB,PROPN,SYM,CONJ,NOUN,SPACE,PART,INTJ,PUNCT,ADP,ADJ,X,ADV,DET,NUM,PUNCT,PRON
#export TAGS=FREQ
export FREQUENCY_THRESHOLD=11851

export OUTPUT_DIR=../data/${DATA_NAME}/
#../data/aclImdb/train \

python longterm_dependencies_tags.py \
  --data_name ${DATA_NAME} \
  --model_type ${MODEL_NAME} \
  --data_dir ../data/${DATA_NAME}/train \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --do_lower_case ${DO_LOWER_CASE} \
  --max_seq_length ${MAX_SEQUENCE_LENGTH} \
  --max_context_size ${MAX_CONTEXT_SIZE} \
  --max_span_length ${MAX_SPAN_LENGTH} \
  --span_length ${SPAN_LENGTH} \
  --max_num_examples ${MAX_NUM_EXAMPLES} \
  --tags ${TAGS} \
  --frequency_threshold ${FREQUENCY_THRESHOLD} \
  --output_dir ${OUTPUT_DIR} \
  --cache_dir ${CACHE_DIR}

