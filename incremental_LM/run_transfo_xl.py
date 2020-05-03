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
    parser.add_argument("--model_name", type=str, default="transfo-xl", help="pretrained model name")
    parser.add_argument("--model_path", type=str, default="transfo-xl-wt103", help="pretrained model path")
    parser.add_argument(
        "--split", type=str, default="test", choices=["all", "valid", "test"], help="which split to evaluate"
    )
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--tgt_len", type=int, default=128, help="number of tokens to predict")
    parser.add_argument("--ext_len", type=int, default=0, help="length of the extended context")
    parser.add_argument("--mem_len", type=int, default=1600, help="length of the retained previous heads")
    parser.add_argument("--clamp_len", type=int, default=400, help="max positional embedding index")
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
    
    # this path has corpus.bin (wikitext-103 processed for transformer-xl)
    corpus = TransfoXLCorpus.from_pretrained(args.model_path)

    # corpus tokenized only for transformer-xl
    
    # corpus.vocab : TransfoXLTokenizer object from TransfoXLTokenizer.from_pretrained()
    ntokens = len(corpus.vocab)
    logger.info("n_tokens: {}".format(ntokens))
    logger.info("train shape: {}".format(corpus.train.shape))
    logger.info("test shape: {}".format(corpus.test.shape))

    va_iter = corpus.get_iterator("valid", args.batch_size, args.tgt_len, device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator("test", args.batch_size, args.tgt_len, device=device, ext_len=args.ext_len)

    # Load a pre-trained model


    model = TransfoXLLMHeadModel.from_pretrained(args.model_path)
    model = model.to(device)

    tokenizer = TransfoXLTokenizer.from_pretrained(args.model_path)

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
    def evaluate(eval_iter, outDir, args):

        sentences = []
        targets = []
        pred_toks = []
        probs_true = []
        probs_pred = []
        ranks_true = []
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_len, total_loss = 0, 0.0
        start_time = time.time()
        with torch.no_grad():
            mems = None
            mems1 = None
            for idx, (data, target, seq_len) in enumerate(eval_iter):
                # print('data.shape', data.shape, target.shape)
                # for bsz = 1. sentence/target are dims: (bsz, seqlen)
                # logger.info("sentence: {}".format(" ".join(tokenizer.convert_ids_to_tokens(data[0]))))
                # logger.info("target: {}".format(" ".join(tokenizer.convert_ids_to_tokens(target[0]))))

                if args.no_write is False:
                    # Write predictions
                    prediction_scores, mems = model(data, mems=mems)  # dims: (bsz, seqlen, config.vocab)

                    prediction_scores = prediction_scores.view(-1, ntokens) # dims: (seqlen, vocab_sz)
                    predictions = prediction_scores.softmax(dim=1) # dims: (seqlen, vocab_sz)

                    predicted_ids = torch.argmax(predictions, dim=1) # dims: (seqlen)
                    predicted_id_list = predicted_ids.tolist()
                    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_id_list) # dims: (seqlen)
                    # logger.info("Predicted_tokens: {}".format(" ".join(predicted_tokens)))
                    probabilities_predicted = predictions.gather(1, predicted_ids.view(-1, 1)) # dims: (seqlen, 1)
                    probabilities_predicted = torch.squeeze(probabilities_predicted) # dims: (seqlen)

                    correct_ids = target[0]
                    correct_ids = torch.tensor(correct_ids)
                    correct_ids = correct_ids.to(device)
                    probabilities_correct = predictions.gather(1, correct_ids.view(-1, 1))
                    probabilities_correct = torch.squeeze(probabilities_correct)

                    ranks = prediction_scores.size(1) - torch.zeros_like(prediction_scores).long().scatter_(1, prediction_scores.argsort(dim=1),
                                                                                            torch.arange(
                                                                                                prediction_scores.size(1),
                                                                                                device=device).repeat(
                                                                                                prediction_scores.size(0), 1))
                    ranks = ranks.gather(1, correct_ids.view(-1, 1)).float()
                    ranks = torch.squeeze(ranks)
                    
                    sent = tokenizer.convert_ids_to_tokens(data[0])
                    if len(sent) == args.tgt_len:
                        # last sentence may be shorter
                        sentences.append(sent)
                        targets.append(tokenizer.convert_ids_to_tokens(target[0]))
                        pred_toks.append(predicted_tokens)
                        probs_true.append(probabilities_correct.cpu().numpy())
                        probs_pred.append(probabilities_predicted.cpu().numpy())
                        ranks_true.append(ranks.cpu().numpy())


                # shifting happens inside the model, so labels = data. target is shifted version of data
                # check forward() def. of TransfoXLLMHeadModel
                ret = model(data, labels=data, mems=mems1)
                loss, _ , mems1 = ret
                # dims of loss: (bsz vs seqlen-1)
                loss = loss.mean()
                # logger.info("idx: {0}, seq_len: {1}, loss.item(): {2}".format(idx, seq_len, loss.item()))
                 
                total_loss += seq_len * loss.item()
                total_len += seq_len
            
            total_time = time.time() - start_time
        
        if args.no_write is False:
            # Write predictions
            np.save(os.path.join(outDir, 'sentences.npy'), np.array(sentences))
            np.save(os.path.join(outDir, 'targets.npy'), np.array(targets))
            np.save(os.path.join(outDir, 'pred_toks.npy'), np.array(pred_toks))
            np.save(os.path.join(outDir, 'probs_true.npy'), np.array(probs_true))
            np.save(os.path.join(outDir, 'probs_pred.npy'), np.array(probs_pred))
            np.save(os.path.join(outDir, 'ranks_true.npy'), np.array(ranks_true))
        
        
        logger.info("Time : {:.2f}s, {:.2f}ms/segment".format(total_time, 1000 * total_time / (idx + 1)))
        return total_loss / total_len

    # Run on test data.
    outDir = os.path.join(args.work_dir, args.split, str(args.mem_len))
    
    if os.path.exists(outDir):
        if args.no_write is False:
            # empty directory if to be written further
            logger.info("Emptying outDir: {} to be ready for writing".format(outDir))
            filelist = [f for f in os.listdir(outDir)]
            for f in filelist:
                os.remove(os.path.join(outDir, f))
    else:
        os.makedirs(outDir)

    if args.split == "all":
        test_loss = evaluate(te_iter, os.path.join(args.work_dir, "test", args.tgt_len), args)
        valid_loss = evaluate(va_iter, os.path.join(args.work_dir, "valid", args.tgt_len), args)
    elif args.split == "valid":
        valid_loss = evaluate(va_iter, outDir, args)
        test_loss = None
    elif args.split == "test":
        test_loss = evaluate(te_iter, outDir, args)
        valid_loss = None

    def format_log(loss, split):
        log_str = "| {0} loss {1:5.2f} | {0} ppl {2:9.3f} ".format(split, loss, math.exp(loss))
        return log_str

    log_str = ""
    if valid_loss is not None:
        log_str += format_log(valid_loss, "valid")
    if test_loss is not None:
        log_str += format_log(test_loss, "test")



    logger.info("=" * 100)
    logger.info(log_str)
    logger.info("=" * 100)

    output_eval_file = os.path.join(outDir, "eval_results_lm.txt")
    with open(output_eval_file, "w") as writer:
        writer.write(log_str + "\n")

    logger.info("Process Completed")

if __name__ == "__main__":
    main()
