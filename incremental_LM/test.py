
import argparse
import logging
import math
import time

import torch

from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers.tokenization_transfo_xl import TransfoXLCorpus


tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
   
def score(sentence):
    with torch.no_grad():
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss, _ = model(tensor_input, labels=tensor_input)
        print(loss.shape)
        loss = loss.mean()
        print(loss.shape)
        return math.exp(loss.item())


a=['there is a book on the desk',
                'there is a plane on the desk',
                        'there is a book in the desk']
# print perplexity
print([score(i) for i in a])

# transformer-xl [149.1977885032318, 276.4760489861424, 135.4560911474157]
# gpt 