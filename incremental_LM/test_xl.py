
import argparse
import logging
import math
import time
import sys
import torch

from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers.tokenization_transfo_xl import TransfoXLCorpus


tokenizer = TransfoXLTokenizer.from_pretrained(sys.argv[1])
model = TransfoXLLMHeadModel.from_pretrained(sys.argv[1])
model.eval()
   
def score(sentence):
    with torch.no_grad():
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss, prediction_score, _ = model(tensor_input, labels=tensor_input)
        # prediction_score is None
        loss = loss.mean()
        return math.exp(loss.item())


a=['there is a book on the desk',
                'there is a plane on the desk',
                        'there is a book in the desk']
# print perplexity
print([score(i) for i in a])

# transformer-xl [149.1977885032318, 276.4760489861424, 135.4560911474157]
# gpt [21.316539840591485, 61.459132414379624, 26.249264459730522]

'''
print(loss)
tensor([[ 2.7568,  1.3658,  7.8468,  3.2799,  2.4584, 12.3240]]) torch.Size([1, 6])
tensor(5.0053) torch.Size([])
'''