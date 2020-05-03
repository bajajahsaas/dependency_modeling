# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" PyTorch Transformer XL model evaluation script.
    Adapted from https://github.com/kimiyoung/transformer-xl.
    In particular https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/eval.py

    This script with default values evaluates a pretrained Transformer-XL on WikiText 103
"""


import argparse
import logging
import math
import time
import numpy as np
import torch
import os

from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers.tokenization_transfo_xl import TransfoXLCorpus

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Language Model")
    parser.add_argument("--model_name", type=str, default="transfo-xl-wt103", help="pretrained model name")
    parser.add_argument(
        "--split", type=str, default="valid", choices=["all", "valid", "test"], help="which split to evaluate"
    )
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--tgt_len", type=int, default=128, help="number of tokens to predict")
    parser.add_argument("--ext_len", type=int, default=0, help="length of the extended context") # as per git issue, this was just for experimentation and never used
    parser.add_argument("--mem_len", type=int, default=1600, help="length of the retained previous heads")
    parser.add_argument("--clamp_len", type=int, default=1000, help="max positional embedding index")
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA even though CUA is available")
    parser.add_argument("--work_dir", type=str, required=True, help="path to the work_dir")
    parser.add_argument("--no_log", action="store_true", help="do not log the eval result")
    parser.add_argument("--same_length", action="store_true", help="set same length attention with masking")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--no_write", action="store_true", help="Don't write predictions, just calculate perplexity")
    args = parser.parse_args()
    assert args.ext_len >= 0, "extended context length must be non-negative"


    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    # Load a pre-processed dataset
    # You can also build the corpus yourself using TransfoXLCorpus methods
    # The pre-processing involve computing word frequencies to prepare the Adaptive input and SoftMax
    # and tokenizing the dataset
    # The pre-processed corpus is a convertion (using the conversion script )
    
    corpus = TransfoXLCorpus.from_pretrained(args.model_name)
    tokenizer = TransfoXLTokenizer.from_pretrained(args.model_name)



    # corpus.vocab : TransfoXLTokenizer object from TransfoXLTokenizer.from_pretrained()
    ntokens = len(corpus.vocab)
    logger.info("n_tokens: {}".format(ntokens))
    logger.info("train shape: {}".format(corpus.train.shape))
    logger.info("test shape: {}".format(corpus.test.shape))

    va_iter = corpus.get_iterator("valid", args.batch_size, args.tgt_len, device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator("test", args.batch_size, args.tgt_len, device=device, ext_len=args.ext_len)

    # Load a pre-trained model
    model = TransfoXLLMHeadModel.from_pretrained(args.model_name)
    model = model.to(device)

    logger.info(
        "Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {} no_write {}".format(
            args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len, args.no_write
        )
    )

    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    if args.clamp_len > 0:
        model.clamp_len = args.clamp_len
    if args.same_length:
        model.same_length = True

    # model.resize_token_embeddings(len(tokenizer))

    ###############################################################################
    # Evaluation code
    ###############################################################################
    def evaluate(eval_iter, args):

        sentences = []
        targets = []
        pred_toks = []
        probs_true = []
        probs_pred = []
        ranks_true = []
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_len, total_loss = 0, 0.0
        total_loss1, total_loss2, total_loss3, total_loss4 = 0.0, 0.0, 0.0, 0.0
        total_examples = 0
        num_examples = 0
        start_time = time.time()
        with torch.no_grad():
            mems = None
            mems1 = None
            for idx, (data, target, seq_len) in enumerate(eval_iter):
                # for bsz = 1. sentence/target are dims: (bsz, seqlen)
                # logger.info("sentence: {}".format(" ".join(tokenizer.convert_ids_to_tokens(data[0]))))
                # logger.info("target: {}".format(" ".join(tokenizer.convert_ids_to_tokens(target[0]))))

    
                # shifting happens inside the model, so labels = data. target is shifted version of data
                # check forward() def. of TransfoXLLMHeadModel
                ret = model(data, labels=data, mems=mems1)
                loss, _ , mems1 = ret
                # dims of loss: (bsz vs seqlen-1)

                num_examples += loss.shape[0]

                if seq_len < args.tgt_len:
                    # add 0 in losses
                    logger.info("Received a batch with outlying seq_len")   
                                    
                    continue
                
                thresh1 = int((seq_len/4))
                thresh2 = 2*thresh1
                thresh3 = 3*thresh1
            
                loss1 = loss[:, :thresh1]
                loss2 = loss[:, thresh1: thresh2]
                loss3 = loss[:, thresh2: thresh3]
                loss4 = loss[:, thresh3:]
                
                loss1 = loss1.mean()
                loss2 = loss2.mean()
                loss3 = loss3.mean()
                loss4 = loss4.mean()

                loss = loss.mean()

                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss3 += loss3.item()
                total_loss4 += loss4.item()
                
                total_loss += loss.item()
                
                total_examples += 1 # over which mean taken, to be divided by same. If bsz > 1, .mean() takes mean over all values. So at last divide by number of batches
                
            total_time = time.time() - start_time
        
        
        logger.info("Time : {:.2f}s, {:.2f}ms/segment".format(total_time, 1000 * total_time / (idx + 1)))
        logger.info("Total number of examples: {}".format(num_examples))

        avloss1, avloss2, avloss3, avloss4, avloss = total_loss1 / total_examples, total_loss2 / total_examples, total_loss3 / total_examples, total_loss4 / total_examples, total_loss / total_examples
        return avloss1, avloss2, avloss3, avloss4, avloss

    # Run on test data.
   
    if args.split == "all":
        test_loss = evaluate(te_iter, os.path.join(args.work_dir, args.tgt_len), args)
        valid_loss = evaluate(va_iter, os.path.join(args.work_dir, args.tgt_len), args)
    elif args.split == "valid":
        valid_loss = evaluate(va_iter, args)
        test_loss = None
    elif args.split == "test":
        test_loss = evaluate(te_iter, args)
        valid_loss = None

    def compute_perpl(losses):
        l1, l2, l3, l4, l = losses
        return math.exp(l1), math.exp(l2), math.exp(l3), math.exp(l4), math.exp(l)

    def format_log(losses, split):
        ppl1, ppl2, ppl3, ppl4, ppl = compute_perpl(losses)
        log_str = "|{0} ppl1 {1:9.3f} ppl2 {2:9.3f} ppl3 {3:9.3f} ppl4 {4:9.3f}, ppl {5:9.3f}".format(split, ppl1, ppl2, ppl3, ppl4, ppl)
        return log_str

    log_str = ""
    if valid_loss is not None:
        log_str += format_log(valid_loss, "valid")
    if test_loss is not None:
        log_str += format_log(test_loss, "test")

    logger.info("=" * 100)
    logger.info(log_str)
    logger.info("=" * 100)

    logger.info("Process Completed")

if __name__ == "__main__":
    main()
