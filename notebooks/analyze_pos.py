#!/usr/bin/env python
# coding: utf-8

# In[747]:


import numpy as np
from pathlib import Path
import random 

model_name = "bert-base-cased"
dataset = "aclImdb"
tag = "ADJ"
span_length = 1
max_context_size = 249
measure = "acc" # acc, ppl, loss, prob, rank


# In[748]:


data_dir = "../output_dir"
data_path = Path(data_dir)
assert data_path.exists(), f'Error: {data_path} does not exist.'


# In[749]:


result_file = data_path / f'{model_name}-{dataset}-{tag}' / f'all_{measure}_context_size_{span_length}.npy'
results = np.load(result_file)

original_text_file = data_path / f'{model_name}-{dataset}-{tag}' / f'original_texts_{span_length}.npy'
original_texts = np.load(original_text_file)
tokenized_text_file = data_path / f'{model_name}-{dataset}-{tag}' / f'all_tokenized_texts_{span_length}.npy'

tokenized_texts = np.load(tokenized_text_file, allow_pickle=True)
predicted_tokens_file = data_path / f'{model_name}-{dataset}-{tag}' / f'all_pred_toks_{span_length}.npy'
predicted_tokens = np.load(predicted_tokens_file, allow_pickle=True)

masked_tokens_file = data_path / f'{model_name}-{dataset}-{tag}' / f'all_masked_toks_{span_length}.npy'
masked_tokens = np.load(masked_tokens_file, allow_pickle=True)

masked_tokens_file = data_path / f'{model_name}-{dataset}-{tag}' / f'all_masked_toks_{span_length}.npy'
masked_tokens = np.load(masked_tokens_file, allow_pickle=True)

probs_true_file = data_path / f'{model_name}-{dataset}-{tag}' / f'all_probs_true_{span_length}.npy'
probs_true= np.load(probs_true_file, allow_pickle=True)

probs_pred_file = data_path / f'{model_name}-{dataset}-{tag}' / f'all_probs_pred_{span_length}.npy'
probs_pred = np.load(probs_pred_file, allow_pickle=True)

ranks_true_file = data_path / f'{model_name}-{dataset}-{tag}' / f'all_ranks_true_{span_length}.npy'
ranks_true = np.load(ranks_true_file, allow_pickle=True)


# In[750]:


while(1):
    example_id = random.randint(0,3000)
    if np.sum(results[example_id]) == 0:
        continue
    example_id = 1123
    write2file = True
    if write2file:
        fname = model_name + '-' + dataset + '-' + tag + '-' + str(example_id)
        outfile = open('results-' + tag + '/' + fname + '.txt', 'w') 

    print(results.shape) # NUM_EXAMPLES vs context_sizes
    print(example_id, results[example_id])
    break


# In[751]:


# print(original_texts[example_id])
if write2file:
    print(f"\nAccuracy across different context size: \n{results[example_id]}", file = outfile)
    print(f"\nOriginal text: \n{original_texts[example_id]}", file = outfile)


# In[752]:


a = " ".join(tokenized_texts[example_id]).replace(" ##", "")
cleaned_tokenized_text = a.split()
# print(" ".join(cleaned_tokenized_text))
if write2file:
    text_output = " ".join(cleaned_tokenized_text)
    print(f"\nTokenized text: \n{text_output}", file=outfile)


# In[753]:


span_start_index = None
for i in range(len(cleaned_tokenized_text)):
    if cleaned_tokenized_text[i] == '[MASK]':
        span_start_index = i
        break
masked_indices = list(range(span_start_index, span_start_index + span_length))
span_end_index = masked_indices[-1]
        
context_sizes = [1,2,3] + list(range(5,30,5)) + list(range(30, max_context_size,10))
context_size_count = 0
for context_size in context_sizes:
    context_masked_text = cleaned_tokenized_text[:]
    context_masked_text = context_masked_text[span_start_index - context_size:span_end_index + context_size + 1]
    print('##################################################')
    print(f'context_size = {context_sizes[context_size_count]}')
    # print(' '.join(context_masked_text[start:end+1]))
    print(' '.join(context_masked_text))
    
    ### Write to file
    if write2file:
        print('##################################################',file = outfile)
        print(f'context_size = {context_sizes[context_size_count]}',file = outfile)
        print(' '.join(context_masked_text),file = outfile)
    
    probs_true[example_id][context_size_count]
    print(f'masked token(s) = {masked_tokens[example_id]}')
    print(f'probs for the masked token(s) = {[item for subarr in probs_true[example_id][context_size_count] for item in subarr]}')
    print(f'ranks for the masked token(s) = {[int(item) for subarr in ranks_true[example_id][context_size_count] for item in subarr]}')
    
    print(f'predicted token(s) = {predicted_tokens[example_id][context_size_count]}')
    print(f'probs for the predicted token(s) = {[item for subarr in probs_pred[example_id][context_size_count] for item in subarr]}')
    print(f'{measure} = {results[example_id][context_size_count]}')
    
    ### Write to file
    if write2file:
        print(f'masked token(s) = {masked_tokens[example_id]}',file = outfile)
        print(f'probs for the masked token(s) = {[item for subarr in probs_true[example_id][context_size_count] for item in subarr]}',file = outfile)
        print(f'ranks for the masked token(s) = {[int(item) for subarr in ranks_true[example_id][context_size_count] for item in subarr]}',file = outfile)

        print(f'predicted token(s) = {predicted_tokens[example_id][context_size_count]}',file = outfile)
        print(f'probs for the predicted token(s) = {[item for subarr in probs_pred[example_id][context_size_count] for item in subarr]}',file = outfile)
        print(f'{measure} = {results[example_id][context_size_count]}',file = outfile)
    
    context_size_count += 1
outfile.close()

