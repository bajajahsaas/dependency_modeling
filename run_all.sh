#!/usr/bin/env bash

model_name_or_path=('bert-base-cased' 'bert-large-cased' 'roberta-base' 'roberta-large' 'xlnet-base-cased' 'xlnet-large-cased' 'xlm-mlm-en-2048')
model_name=('bert' 'bert' 'roberta' 'roberta' 'xlnet' 'xlnet' 'xlm')
data_name=('RACE' 'aclImdb')

for data in "${data_name[@]}"
do
  for ((i=0;i<${#model_name_or_path[@]};i++));
  do
     export DATA=$data
     export MODEL_NAME_PATH=${model_name_or_path[i]}
     export MODEL_NAME=${model_name[i]}
#     echo $DATA $MODEL_NAME_PATH $MODEL_NAME
     sbatch --job-name=${MODEL_NAME}_${DATA} --gres=gpu:1 --partition=1080ti-long --mem=40GB --output=logsdepmodel/dependency_modeling_${MODEL_NAME}_${DATA}.txt dependency_modeling.sh ${MODEL_NAME_PATH} ${MODEL_NAME} ${DATA}
  done
done

#for tag in "${model_name_or_path[@]}"
#do
#    sbatch --gres=gpu:1 --partition=1080ti-long --mem=40GB --output=dependency_modeling_${tag}_reberta_aclImdb.txt dependency_modeling_tags.sh ${tag}
#done
