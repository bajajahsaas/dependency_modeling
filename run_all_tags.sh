#!/usr/bin/env bash

#for tag in "FREQ" "INFREQ" "VERB" "PROPN" "SYM" "NOUN" "PART" "INTJ" "PUNCT" "ADP" "ADJ" "X" "ADV" "DET" "NUM" "PUNCT" "PRON"
for tag in "PROPN" "NOUN" "ADJ" "ADV" "DET" "NUM"
do
    sbatch --gres=gpu:1 --partition=m40-long --mem=40GB --output=logsdepmodel_tags/dependency_modeling_${tag}_xlm_RACE.txt dependency_modeling_tags.sh ${tag}
done

# change model name, type, dataset in dependency_modeling_tags.sh