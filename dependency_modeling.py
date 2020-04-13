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

from transformers import (WEIGHTS_NAME,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          XLNetConfig, XLNetLMHeadModel, XLNetTokenizer,
                          XLMConfig, XLMWithLMHeadModel, XLMTokenizer)

from data_processing import get_texts
from pathlib import Path

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)),
    ())

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
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--no_cuda", default=False, type=eval,
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
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3.")
    args = parser.parse_args()

    logger.info('Printing training configuration...')
    logger.info('Data: ', args.data_name, ' model_type: ', args.model_type, ' model_name_path: ', args.model_name_or_path, 'output_dir: ', args.output_dir)

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
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.info("device: {} n_gpu: {}".format(args.device, args.n_gpu))

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    original_texts = []
    tokenized_texts = []
    example_ids = []

    count = 0
    for text in texts:
        count += 1
        if count % 100 == 0:
            logger.info("  Loaded and processed %d examples", count)

        text = tokenizer.cls_token + " " + text.strip()

        tokenized_text = tokenizer.tokenize(text)
        if len(tokenized_text) >= 1 + 2 * args.max_context_size + args.max_span_length:
            original_texts.append(text)
            tokenized_texts.append(tokenized_text[:args.max_seq_length - 2])  # two positions for special tokens
            example_ids.append(count)  # to refer back to original dataset, which examples got in


    # import pdb
    # pdb.set_trace()

    print('max_num_examples: ', args.max_num_examples)
    print('tokenized text length: ', len(tokenized_texts))

    tokenized_texts = tokenized_texts[:args.max_num_examples]
    logger.info("***** Finished tokenizing data *****")
    logger.info("  Num examples = %d", len(tokenized_texts))
    model.eval()
    model.to(args.device)

    # ideally only write [:args.max_num_examples] version of lists below
    np.save(os.path.join(args.output_dir,
                         'original_texts_{}.npy'.format(args.span_length)), np.array(original_texts))

    np.save(os.path.join(args.output_dir,
                         'examples_ids_{}.npy'.format(args.span_length)), np.array(example_ids))

    all_tokenized_texts = []
    all_acc_context_size = []
    all_ppl_context_size = []
    all_loss_context_size = []
    all_prob_context_size = []
    all_rank_context_size = []
    all_pred_toks = []
    all_probs_true = []
    all_probs_pred = []
    all_ranks_true = []
    all_masked_toks = []

    count = 0
    for tokenized_text in tokenized_texts:
        count += 1
        if count > 1:
            logger.info("***** Finished %d examples *****", count - 1)

        span_start_index = random.randrange(1 + args.max_context_size,
                                            1 + len(tokenized_text) - args.max_context_size - args.span_length)
        original_masked_indices = list(range(span_start_index, span_start_index + args.span_length))
        span_end_index = original_masked_indices[-1]

        masked_tokens = []

        for masked_index in original_masked_indices:
            masked_tokens.append(tokenized_text[masked_index])
            tokenized_text[masked_index] = tokenizer.mask_token

        all_tokenized_texts.append(tokenized_text)
        all_masked_toks.append(masked_tokens)

        original_masked_indices = torch.tensor(original_masked_indices)
        correct_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
        correct_ids = torch.tensor(correct_ids)

        # If you have a GPU, put everything on cuda
        original_masked_indices = original_masked_indices.to(args.device)
        correct_ids = correct_ids.to(args.device)

        acc_context_size = []
        ppl_context_size = []
        loss_context_size = []
        prob_context_size = []
        rank_context_size = []
        pred_toks = []
        probs_true = []
        probs_pred = []
        ranks_true = []

        context_sizes = [1, 2, 3] + list(range(5, 30, 5)) + list(range(30, args.max_context_size, 10))
        # for context_size in range(1, 1 + args.max_context_size):
        for context_size in context_sizes:
            context_masked_text = tokenized_text[:]
            context_masked_text = context_masked_text[span_start_index - context_size:span_end_index + context_size + 1]
            # Convert token to vocabulary indices
            if args.model_type not in ['bert', 'roberta', 'xlm']:
                masked_indices = original_masked_indices - (span_start_index - context_size)
                # indexed_tokens = tokenizer.convert_tokens_to_ids(context_masked_text)
            else:
                masked_indices = original_masked_indices - (span_start_index - context_size) + 1
            indexed_tokens = tokenizer.build_inputs_with_special_tokens(
                tokenizer.convert_tokens_to_ids(context_masked_text))
            # Define sentence
            segments_ids = [0] * len(indexed_tokens)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            if args.model_type in ['xlnet']:
                perm_mask = torch.zeros((1, tokens_tensor.shape[1], tokens_tensor.shape[1]), dtype=torch.float)
                perm_mask[:, :, masked_indices] = 1.0

                target_mapping = torch.zeros((1, len(masked_indices), tokens_tensor.shape[1]),
                                             dtype=torch.float)
                target_mapping[0, 0, masked_indices] = 1.0
                perm_mask = perm_mask.to(args.device)
                target_mapping = target_mapping.to(args.device)
            # If you have a GPU, put everything on cuda
            tokens_tensor = tokens_tensor.to(args.device)
            segments_tensors = segments_tensors.to(args.device)

            if args.model_type in ['xlnet']:
                inputs = {'input_ids': tokens_tensor,
                          'attention_mask': None,
                          'token_type_ids': segments_tensors,
                          'perm_mask': perm_mask,
                          'target_mapping': target_mapping,
                          }
            else:
                inputs = {'input_ids': tokens_tensor,
                          'attention_mask': None,
                          'token_type_ids': segments_tensors if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM and RoBERTa don't use segment_ids
                          }
            # Predict all tokens
            with torch.no_grad():
                outputs = model(**inputs)
                if args.model_type in ['xlnet']:
                    # Return target_mapping.shape[1] (already corresponds to masked_indices size)
                    predictions = outputs[0].view(-1, config.vocab_size)
                else:
                    # Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
                    predictions = outputs[0].view(-1, config.vocab_size)[masked_indices]

            masked_probabilities = predictions.softmax(dim=1)
            probabilities_correct = masked_probabilities.gather(1, correct_ids.view(-1, 1))

            # mean over all examples in that context size
            prob = probabilities_correct.mean().item()

            ranks = predictions.size(1) - torch.zeros_like(predictions).long().scatter_(1, predictions.argsort(dim=1),
                                                                                        torch.arange(
                                                                                            predictions.size(1),
                                                                                            device=args.device).repeat(
                                                                                            predictions.size(0), 1))
            ranks = ranks.gather(1, correct_ids.view(-1, 1)).float()
            rank = ranks.mean().item()

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(predictions, correct_ids.view(-1)).item()
            predicted_ids = torch.argmax(predictions, dim=1)
            predicted_id_list = predicted_ids.tolist()
            probabilities_predicted = masked_probabilities.gather(1, predicted_ids.view(-1, 1))
            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_id_list)

            n_matches = sum([1 if p == y else 0 for p, y in zip(predicted_tokens, masked_tokens)])
            acc = n_matches * 100.0 / args.span_length
            ppl = math.exp(masked_lm_loss)

            acc_context_size.append(acc)
            ppl_context_size.append(ppl)
            loss_context_size.append(masked_lm_loss)
            prob_context_size.append(prob)
            rank_context_size.append(rank)

            pred_toks.append(predicted_tokens)
            probs_true.append(probabilities_correct.cpu().numpy())
            probs_pred.append(probabilities_predicted.cpu().numpy())
            ranks_true.append(ranks.cpu().numpy())

        all_acc_context_size.append(acc_context_size)
        all_ppl_context_size.append(ppl_context_size)
        all_loss_context_size.append(loss_context_size)
        all_prob_context_size.append(prob_context_size)
        all_rank_context_size.append(rank_context_size)

        all_pred_toks.append(pred_toks)
        all_probs_true.append(probs_true)
        all_probs_pred.append(probs_pred)
        all_ranks_true.append(ranks_true)

    print('Finished execution, saving files..')
    np.save(os.path.join(args.output_dir,
                         'all_tokenized_texts_{}.npy'.format(args.span_length)), np.array(all_tokenized_texts))
    np.save(os.path.join(args.output_dir,
                         'all_acc_context_size_{}.npy'.format(args.span_length)), np.array(all_acc_context_size))
    np.save(os.path.join(args.output_dir,
                         'all_ppl_context_size_{}.npy'.format(args.span_length)), np.array(all_ppl_context_size))
    np.save(os.path.join(args.output_dir,
                         'all_loss_context_size_{}.npy'.format(args.span_length)), np.array(all_loss_context_size))
    np.save(os.path.join(args.output_dir,
                         'all_prob_context_size_{}.npy'.format(args.span_length)), np.array(all_prob_context_size))
    np.save(os.path.join(args.output_dir,
                         'all_rank_context_size_{}.npy'.format(args.span_length)), np.array(all_rank_context_size))
    np.save(os.path.join(args.output_dir,
                         'all_pred_toks_{}.npy'.format(args.span_length)), np.array(all_pred_toks))
    np.save(os.path.join(args.output_dir,
                         'all_probs_true_{}.npy'.format(args.span_length)), np.array(all_probs_true))
    np.save(os.path.join(args.output_dir,
                         'all_probs_pred_{}.npy'.format(args.span_length)), np.array(all_probs_pred))
    np.save(os.path.join(args.output_dir,
                         'all_ranks_true_{}.npy'.format(args.span_length)), np.array(all_ranks_true))
    np.save(os.path.join(args.output_dir,
                         'all_masked_toks_{}.npy'.format(args.span_length)), np.array(all_masked_toks))

    print('Files saved in..', args.output_dir)

if __name__ == "__main__":
    main()
