#!/usr/bin/env bash

for tag in "FREQ" #"INFREQ" "VERB" "PROPN" "SYM" "NOUN" "PART" "INTJ" "PUNCT" "ADP" "ADJ" "X" "ADV" "DET" "NUM" "PUNCT" "PRON"
do
    sbatch --gres=gpu:1 --partition=1080ti-long --mem=40GB --output=logsdepmodel_tags/dependency_modeling_${tag}_reberta_aclImdb.txt dependency_modeling_tags.sh ${tag}
done
