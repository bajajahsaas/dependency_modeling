#!/usr/bin/env python
# coding: utf-8

# In[337]:


import numpy as np
import random
from pathlib import Path
import json

# models = ["bert-base-cased", "roberta-base", "xlnet-base-cased" , "xlm-mlm-en-2048"]
model_name = "xlnet-base-cased"
dataset = "aclImdb"
span_length = 1
max_context_size = 249
measure = "acc" # acc, ppl, loss, prob, rank


# In[338]:


data_dir = "../../output_dir"
data_path = Path(data_dir)
assert data_path.exists(), f'Error: {data_path} does not exist.'


# In[342]:


with open('c_maps.json', 'r') as fp:
    c_maps = json.load(fp)
    
with open('c_revmaps.json', 'r') as fp:
    c_revmaps = json.load(fp)


# In[339]:


result_file = data_path / f'{model_name}-{dataset}' / f'all_{measure}_context_size_{span_length}.npy'
results = np.load(result_file)

original_text_file = data_path / f'{model_name}-{dataset}' / f'original_texts_{span_length}.npy'
original_texts = np.load(original_text_file)
tokenized_text_file = data_path / f'{model_name}-{dataset}' / f'all_tokenized_texts_{span_length}.npy'

tokenized_texts = np.load(tokenized_text_file, allow_pickle=True)
predicted_tokens_file = data_path / f'{model_name}-{dataset}' / f'all_pred_toks_{span_length}.npy'
predicted_tokens = np.load(predicted_tokens_file, allow_pickle=True)

masked_tokens_file = data_path / f'{model_name}-{dataset}' / f'all_masked_toks_{span_length}.npy'
masked_tokens = np.load(masked_tokens_file, allow_pickle=True)

masked_tokens_file = data_path / f'{model_name}-{dataset}' / f'all_masked_toks_{span_length}.npy'
masked_tokens = np.load(masked_tokens_file, allow_pickle=True)

probs_true_file = data_path / f'{model_name}-{dataset}' / f'all_probs_true_{span_length}.npy'
probs_true= np.load(probs_true_file, allow_pickle=True)

probs_pred_file = data_path / f'{model_name}-{dataset}' / f'all_probs_pred_{span_length}.npy'
probs_pred = np.load(probs_pred_file, allow_pickle=True)

ranks_true_file = data_path / f'{model_name}-{dataset}' / f'all_ranks_true_{span_length}.npy'
ranks_true = np.load(ranks_true_file, allow_pickle=True)


# In[340]:


while(1):
    example_id = random.randint(0, results.shape[0])
    if np.sum(results[example_id]) == 0:
        continue
    
#     example_id = 2283
    write2file = False
    if write2file:
        dirpath = model_name + '-' + dataset
        outfile = open('results/' + dirpath + '/' + str(example_id) + '.txt', 'w') 

    print(results.shape) # NUM_EXAMPLES vs context_sizes
    print(example_id, results[example_id])
    break


# In[341]:





# In[328]:


# print(original_texts[example_id])
if write2file:
    print(f"\nAccuracy across different context size: \n{results[example_id]}", file = outfile)
    print(f"\nOriginal text: \n{original_texts[example_id]}", file = outfile)


# In[329]:


# just use below for printing but don't replace because it changes indices
# a = " ".join(tokenized_texts[example_id]).replace(" ##", "") #ROBERTA
a = " ".join(tokenized_texts[example_id]).replace("▁", "")  #XLNET
cleaned_tokenized_text = a.split()
#print(" ".join(cleaned_tokenized_text))
if write2file:
    text_output = " ".join(cleaned_tokenized_text)
    #print(f"\nTokenized text: \n{text_output}", file=outfile)


# In[330]:


span_start_index = None
for i in range(len(tokenized_texts[example_id])):
    if tokenized_texts[example_id][i] == '<mask>':
        span_start_index = i
        break
masked_indices = list(range(span_start_index, span_start_index + span_length))
span_end_index = masked_indices[-1]
        
context_sizes = [1,2,3] + list(range(5,30,5)) + list(range(30, max_context_size,10))
context_size_count = 0
for context_size in context_sizes:
    context_masked_text = tokenized_texts[example_id][:]
    context_masked_text = context_masked_text[span_start_index - context_size:span_end_index + context_size + 1]
    print('##################################################')
    print(f'context_size = {context_sizes[context_size_count]}')
    # print(' '.join(context_masked_text[start:end+1]))
    
    # cleaning context text for printing. Different for different models
    #cleaned_context_masked_text = " ".join(context_masked_text).replace(" ##", "")  #ROBERTA
    cleaned_context_masked_text = " ".join(context_masked_text).replace("▁", "")  #XLNET
    print(cleaned_context_masked_text)
    
    # Write to file
    if write2file:
        print('##################################################', file=outfile)
        print(f'context_size = {context_sizes[context_size_count]}', file=outfile)
        print(cleaned_context_masked_text, file=outfile)
        
    probs_true[example_id][context_size_count]
    print(f'masked token(s) = {masked_tokens[example_id]}')
    print(f'probs for the masked token(s) = {[item for subarr in probs_true[example_id][context_size_count] for item in subarr]}')
    print(f'ranks for the masked token(s) = {[int(item) for subarr in ranks_true[example_id][context_size_count] for item in subarr]}')
    
    print(f'predicted token(s) = {predicted_tokens[example_id][context_size_count]}')
    print(f'probs for the predicted token(s) = {[item for subarr in probs_pred[example_id][context_size_count] for item in subarr]}')
    print(f'{measure} = {results[example_id][context_size_count]}')
    
    if write2file:
        print(f'masked token(s) = {masked_tokens[example_id]}',file=outfile)
        print(f'probs for the masked token(s) = {[item for subarr in probs_true[example_id][context_size_count] for item in subarr]}',file=outfile)
        print(f'ranks for the masked token(s) = {[int(item) for subarr in ranks_true[example_id][context_size_count] for item in subarr]}',file=outfile)

        print(f'predicted token(s) = {predicted_tokens[example_id][context_size_count]}',file=outfile)
        print(f'probs for the predicted token(s) = {[item for subarr in probs_pred[example_id][context_size_count] for item in subarr]}',file=outfile)
        print(f'{measure} = {results[example_id][context_size_count]}',file=outfile)
    
    context_size_count += 1

