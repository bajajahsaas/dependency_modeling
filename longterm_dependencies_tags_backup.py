# coding=utf-8
# Copyright (c) 2019, Tu Vu.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
import math

from pytorch_transformers import (WEIGHTS_NAME,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  XLNetConfig, XLNetLMHeadModel, XLNetTokenizer,
                                  XLMConfig, XLMWithLMHeadModel, XLMTokenizer)

from data_processing import get_texts
from pathlib import Path
from collections import Counter

import spacy
nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
    'xlm': (XLMConfig, XLMWithLMHeadModel, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_name", default=None, type=str, required=True,
                        help="The dataset name.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", default=False, type=eval,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_context_size", default=128, type=int,
                        help="The maximum context size.")
    parser.add_argument("--max_span_length", default=10, type=int,
                        help="The maximum span length.")
    parser.add_argument("--span_length", default=1, type=int,
                        help="The span length.")
    parser.add_argument("--max_num_examples", default=1000, type=int,
                        help="The maximum number of examples.")
    parser.add_argument("--tags", default="", type=str,
                        help="List of tag names to mask out")
    parser.add_argument("--frequency_threshold", default=800, type=int,
                        help="The threshold for frequent words.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3.")
    args = parser.parse_args()

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.data_dir):
        raise ValueError("Data directory does not exist!")

    data_path = Path(args.data_dir)
    texts = get_texts(args.data_name, data_path)

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.info("device: {} n_gpu: {}".format(args.device, args.n_gpu))

    # Set seed
    set_seed(args)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    TAGS = args.tags.split(",")

    original_texts = {}
    tokenized_texts = []
    pos_maps = []

    count = 0
    for text in texts:
        count += 1
        if count % 100 == 0:
            logger.info("  Loaded and processed %d examples", count)

        doc = nlp(text)
        pos_map = {}
        for i in range(len(doc)-1):
            if str(doc[i]).lower() not in pos_map:
                pos_map[str(doc[i]).lower()] = {}
            pos_map[str(doc[i]).lower()][str(doc[i + 1]).lower()] = str(doc[i].pos_)
        pos_maps.append(pos_map)
        text = tokenizer.cls_token + " " + text.strip()
        tokenized_text = tokenizer.tokenize(text)
        tokenized_texts.append(tokenized_text)
        original_texts[' '.join(tokenized_text[:args.max_seq_length - 2])] = text

    logger.info("***** Finished tokenizing data *****")
    word_frequency = Counter(w for s in tokenized_texts for w in s)
    import pdb
    pdb.set_trace()
    frequent_words = {w: None for w, f in word_frequency.most_common() if f >= args.frequency_threshold}

    selected_tokenized_texts = []
    selected_pos_maps = []
    for i in range(len(tokenized_texts)):
        if len(tokenized_texts[i]) >= 1 + 2 * args.max_context_size + args.max_span_length:
            selected_tokenized_texts.append(tokenized_texts[i][:args.max_seq_length - 2])
            selected_pos_maps.append(pos_maps[i])

    tokenized_texts = selected_tokenized_texts[:]
    pos_maps = selected_pos_maps[:]

    outputs = {}
    for tag in TAGS:
        outputs[tag] = []
    count = 0
    for i in range(len(tokenized_texts)):
        tokenized_text = tokenized_texts[i]
        pos_map = pos_maps[i]
        count += 1
        if count > 1:
            logger.info("***** Finished %d examples *****", count - 1)

        for j in range(1 + args.max_context_size, 1 + len(tokenized_text) - args.max_context_size - args.span_length):
            masked_index = j
            token = tokenized_text[masked_index]
            next_token = tokenized_text[masked_index + 1]

            example = {}
            example["tokenized_text"] = tokenized_text
            example["original_text"] = original_texts[' '.join(tokenized_text)]
            example["masked_index"] = masked_index
            # frequent/infrequent words
            if token in frequent_words:
                outputs["FREQ"].append(example)
            else:
                outputs["INFREQ"].append(example)

            if args.model_type not in ['bert']:
                token = str(tokenizer.convert_tokens_to_string(token)).strip()
                next_token = str(tokenizer.convert_tokens_to_string(next_token)).strip()
            if token.lower() in pos_map and next_token.lower() in pos_map[token.lower()]:
                tag = pos_map[token.lower()][next_token.lower()]
                if tag in TAGS:
                    example = {}
                    example["tokenized_text"] = tokenized_text
                    example["original_text"] = original_texts[' '.join(tokenized_text)]
                    example["masked_index"] = masked_index
                    outputs[tag].append(example)

    np.save(os.path.join(args.output_dir,
                         'tags_{}.npy'.format(args.model_type)), np.array(outputs))

if __name__ == "__main__":
    main()
