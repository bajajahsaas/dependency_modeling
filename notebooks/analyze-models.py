#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import random
from pathlib import Path
import json


span_length = 1
max_context_size = 249
measure = "acc" # acc, ppl, loss, prob, rank


# In[65]:


data_dir = "../../output_dir"
data_path = Path(data_dir)
assert data_path.exists(), f'Error: {data_path} does not exist.'


# In[66]:


models = ["bert-base-cased", "roberta-base", "xlnet-base-cased" , "xlm-mlm-en-2048"]
dataset = 'aclImdb' # Below analysis done only for aclImdb

print("Dataset: {0}".format(dataset))
# combined results
c_results = {}
c_original_texts = {}
c_tokenized_texts = {}
c_masked_tokens = {}
c_predicted_tokens = {}
c_probs_true = {}
c_probs_pred = {}
c_ranks_true = {}
c_example_ids = {}
c_num_examples = {}

num_examples = 1e7
for model_name in models:
    print('Model: {0}'.format(model_name))
    result_file = data_path / f'{model_name}-{dataset}' / f'all_{measure}_context_size_{span_length}.npy'
    results = np.load(result_file)
    c_results[model_name] = results
    
    examples_ids_file = data_path / f'{model_name}-{dataset}' / f'examples_ids_{span_length}.npy'
    examples_ids = np.load(examples_ids_file)
    c_example_ids[model_name] = examples_ids
  
    original_text_file = data_path / f'{model_name}-{dataset}' / f'original_texts_{span_length}.npy'
    original_texts = np.load(original_text_file)
    c_original_texts[model_name] = original_texts

    tokenized_text_file = data_path / f'{model_name}-{dataset}' / f'all_tokenized_texts_{span_length}.npy'
    tokenized_texts = np.load(tokenized_text_file, allow_pickle=True)
    c_tokenized_texts[model_name] = tokenized_texts

    predicted_tokens_file = data_path / f'{model_name}-{dataset}' / f'all_pred_toks_{span_length}.npy'
    predicted_tokens = np.load(predicted_tokens_file, allow_pickle=True)
    c_predicted_tokens[model_name] = predicted_tokens

    masked_tokens_file = data_path / f'{model_name}-{dataset}' / f'all_masked_toks_{span_length}.npy'
    masked_tokens = np.load(masked_tokens_file, allow_pickle=True)
    c_masked_tokens[model_name] = masked_tokens

    probs_true_file = data_path / f'{model_name}-{dataset}' / f'all_probs_true_{span_length}.npy'
    probs_true= np.load(probs_true_file, allow_pickle=True)
    c_probs_true[model_name] = probs_true

    probs_pred_file = data_path / f'{model_name}-{dataset}' / f'all_probs_pred_{span_length}.npy'
    probs_pred = np.load(probs_pred_file, allow_pickle=True)
    c_probs_pred[model_name] = probs_pred

    ranks_true_file = data_path / f'{model_name}-{dataset}' / f'all_ranks_true_{span_length}.npy'
    ranks_true = np.load(ranks_true_file, allow_pickle=True)
    c_ranks_true[model_name] = ranks_true
    
    print('Examples: ', results.shape[0])
    c_num_examples[model_name] = results.shape[0]
    num_examples= min(results.shape[0], num_examples)
    


# In[68]:


# test if respective IDs have same examples in different models

c_lists = {}
total_lists = []
for model in models:
    # c_original_texts[model] and c_example_ids[model] have same length
    this_list = c_example_ids[model].tolist()[:num_examples]
    c_lists[model] = this_list
    total_lists.append(this_list)
    
common_ids = set(total_lists[0]).intersection(*total_lists[1:])
print('Length of common IDs', len(list(common_ids)))

# example id is common across different models
c_maps = {} # example id to local index [0, num_examples]
c_revmaps = {} # local index to example id


for model in models:
    c_maps[model] = {}
    c_revmaps[model] = {}
    example_ids = c_lists[model]
    for i, example in enumerate(example_ids):
        c_maps[model][example] = i
        c_revmaps[model][i] = example
        # example is the example number in the original data (common across models)
        # i is the index in rest of the arrays (result, original_text etc.)

with open('c_maps.json', 'w') as fp:
    json.dump(c_maps, fp)
    
with open('c_revmaps.json', 'w') as fp:
    json.dump(c_revmaps, fp)


# In[53]:


def convert2bool(lis):
    values = []
    for l in lis:
        l = int(l)
        values.append(True if l == 100 else False)
    return values

# have a list of example ids per context size
all_wrong_egs = {}
all_right_egs = {}

for ex in common_ids:    
    results = []
    for model in models:
        idx = c_maps[model][ex]  # idx is the index in rest of the arrays (result, original_text etc.)
        assert(idx >= 0 and idx < num_examples)
        # print(model, c_original_texts[model][idx][:100]) # all prints the same text..Yayy!
        this_result = c_results[model][idx] # returns a list of size (30): num of ctx windows
        results.append(this_result)
    
    ctx = 0
    for a, b, c, d in zip(*results):
        # iterate over context sizes
        # in order bert, roberta, xlnet, xlm
        a, b, c, d = convert2bool([a, b, c, d])
        
        if ctx not in all_wrong_egs:
            all_wrong_egs[ctx] = []
        if ctx not in all_right_egs:
            all_right_egs[ctx] = []
        
        if (not a) and (not b) and (not c) and (not d):
            all_wrong_egs[ctx].append(ex)
        
        if (a and b and c and d):
            all_right_egs[ctx].append(ex)
            
        ctx += 1

ctx_map = {}
context_sizes = [1,2,3] + list(range(5,30,5)) + list(range(30, max_context_size,10))
for i, ctx in enumerate(context_sizes):
    ctx_map[i] = ctx

print('Total common examples', len(list(common_ids)))
for k in all_right_egs.keys():
    print('Context size: {0} => All Correct: {1}, All Incorrect: {2}'.format(ctx_map[k], len(all_right_egs[k]), len(all_wrong_egs[k])))


# In[62]:


# models = ["bert-base-cased", "roberta-base", "xlnet-base-cased" , "xlm-mlm-en-2048"]
model = 'xlnet-base-cased'
check_for_ctx = 10

print('Context sizes: ', ctx_map.values())
for ctx in all_wrong_egs.keys():
    list_examples = all_wrong_egs[ctx]
    if ctx_map[ctx] == check_for_ctx:
        print('Failing examples: ', len(list_examples))
        for eg in list_examples:
            idx = c_maps[model][eg]
            print(c_masked_tokens[model][idx])


# In[20]:


this_model = ''

while(1):
    example_id = random.randint(0, results.shape[0])
    if np.sum(results[example_id]) == 0:
        continue
#     example_id = 220 # example id from the data
    write2file = False
    
    example_id = c_maps[this_model][example_id]  # change to local example_id (within num_examples)
    if write2file:
        fname = str(example_id) + "-" + model_name + '-' + dataset
        outfile = open('results/comparison' + fname + '.txt', 'w') 

    print(results.shape) # NUM_EXAMPLES vs context_sizes
    print(example_id, results[example_id])
    break


# In[6]:


# print(original_texts[example_id])
if write2file:
    print(f"\nAccuracy across different context size: \n{results[example_id]}", file = outfile)
    print(f"\nOriginal text: \n{original_texts[example_id]}", file = outfile)


# In[7]:


# just use below for printing but don't replace because it changes indices
a = " ".join(tokenized_texts[example_id]).replace(" ##", "")
cleaned_tokenized_text = a.split()
print(" ".join(cleaned_tokenized_text))
if write2file:
    text_output = " ".join(cleaned_tokenized_text)
#     print(f"\nTokenized text: \n{text_output}", file=outfile)


# In[8]:


span_start_index = None
for i in range(len(tokenized_texts[example_id])):
    if tokenized_texts[example_id][i] == '[MASK]':
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
    cleaned_context_masked_text = " ".join(context_masked_text).replace(" ##", "")
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

