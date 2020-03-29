#!/usr/bin/env bash

#for tag in "FREQ" "INFREQ" "VERB" "PROPN" "SYM" "NOUN" "PART" "INTJ" "PUNCT" "ADP" "ADJ" "X" "ADV" "DET" "NUM" "PUNCT" "PRON"
#for tag in "PROPN" "NOUN" "ADJ" "ADV" "DET" "NUM"
for tag in "NOUN"
do
    sbatch --gres=gpu:1 --partition=1080ti-long --mem=40GB --output=logsdepmodel_tags/dependency_modeling_${tag}_xlnet_aclImdb.txt dependency_modeling_tags.sh ${tag}
done
