#!/bin/bash

export DO_LOWER_CASE=False
export CACHE_DIR=../cache
export MAX_CONTEXT_SIZE=249
export MAX_SEQUENCE_LENGTH=512
export SPAN_LENGTH=1
export MAX_SPAN_LENGTH=1
export MAX_NUM_EXAMPLES=3000

#model_name_or_path=('bert-base-cased', 'bert-large-cased', 'roberta-base', 'roberta-large', 'xlnet-base-cased', 'xlnet-large-cased', '/mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/models/xlm-mlm-en-2048')
#model_name=('bert', 'bert', 'roberta', 'roberta', 'xlnet', 'xlnet', 'xlm')
#data_name=('RACE', 'aclImdb')

export MODEL_NAME_OR_PATH=$1
export MODEL_NAME=$2 # bert, xlnet, xlm, roberta
export DATA_NAME=$3 # RACE, aclImdb

export CHECK_XLM="xlm"
if [ "${MODEL_NAME}" = "${CHECK_XLM}" ]
then export OUTPUT_DIR=/mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/my_output_dir/xlm-mlm-en-2048-${DATA_NAME}
else
  export OUTPUT_DIR=/mnt/nfs/work1/696ds-s20/abajaj/nlplab/long-term-context/my_output_dir/${MODEL_NAME_OR_PATH}-${DATA_NAME}
fi

python dependency_modeling.py \
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
  --output_dir ${OUTPUT_DIR} \
  --cache_dir ${CACHE_DIR}
