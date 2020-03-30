#!/bin/bash

#export MODEL_NAME_OR_PATH=/Users/tutvu/Downloads/spanbert_hf_base
export DO_LOWER_CASE=False
export CACHE_DIR=../cache
export MAX_CONTEXT_SIZE=249
export MAX_SEQUENCE_LENGTH=512
export SPAN_LENGTH=1
export MAX_SPAN_LENGTH=1
export MAX_NUM_EXAMPLES=3000 #3000
export MODEL_NAME=xlnet # bert, xlnet, xlm, roberta
export MODEL_NAME_OR_PATH=xlnet-base-cased
# bert-base-cased, bert-large-cased,
# roberta-base, roberta-large,
# xlnet-base-cased, xlnet-large-cased
# /mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/models/xlm-mlm-en-2048

export DATA_NAME=RACE # RACE, aclImdb
export TAG=$1 # NOUN,PROPN,PRON,VERB,ADJ,ADP,DET,FREQ,INFREQ (get from run_all_tags.sh)

export CHECK_XLM="xlm"
if [ "${MODEL_NAME}" = "${CHECK_XLM}" ]
then export OUTPUT_DIR=../output_dir/xlm-mlm-en-2048-${DATA_NAME}-${TAG}
else
  export OUTPUT_DIR=../output_dir/${MODEL_NAME_OR_PATH}-${DATA_NAME}-${TAG}
fi

#../data/aclImdb/train \

python dependency_modeling_tags.py \
  --data_name ${DATA_NAME} \
  --model_type ${MODEL_NAME} \
  --data_dir ../data/${DATA_NAME}/ \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --do_lower_case ${DO_LOWER_CASE} \
  --max_seq_length ${MAX_SEQUENCE_LENGTH} \
  --max_context_size ${MAX_CONTEXT_SIZE} \
  --max_span_length ${MAX_SPAN_LENGTH} \
  --span_length ${SPAN_LENGTH} \
  --max_num_examples ${MAX_NUM_EXAMPLES} \
  --tag ${TAG} \
  --output_dir ${OUTPUT_DIR} \
  --cache_dir ${CACHE_DIR}

