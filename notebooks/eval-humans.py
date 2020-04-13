#!/usr/bin/env python
# coding: utf-8

# In[102]:


import os
import pandas as pd
import json
import numpy as np


# In[99]:


filename = "../../../data/human-ann-Tu/paragraphs.result.json"
file = open(filename, 'r')
data = json.load(file)
values = data.values()

texts = []
humans = []
gold = []
model = []
ctx = []
empty_ann = 0
count = 0
model_corr_couunt = 0
human_corr_count = 0

for ann_list in values:
    # just one value
    print('Total items in dictionary:', len(ann_list))
    for ann in ann_list:
        # Keys: ['annotations', 'tag', 'paragraph_id', 'text', 'ground_truth', 'bert_ranks_true', 'bert_predictions', 'results']
        
        this_text = ann['annotations'] # list of annotators
        if len(this_text) == 0:
            empty_ann += 1
        for this_ann in this_text:
            count += 1
            # multiple human annotation possible for same text config
            text = ann['text']
            human = this_ann['annotation']
            ground_truth = ann['ground_truth']
            model_pred = ann['bert_predictions']
            
            print('len', len(text))
            print('mask index', text.find('<mask>'))
            if human == ground_truth:
                human_corr_count += 1
            
            if model_pred == ground_truth:
                model_corr_couunt += 1
            
            texts.append(ann['text'])
            humans.append(this_ann['annotation'])
            gold.append(ann['ground_truth'])
            model.append(ann['bert_predictions'])
            ctx.append('')


print('Total samples: ', df.shape[0])
assert(count == df.shape[0])
print('Human acc: {0:0.4f}'.format((human_corr_count * 100.0)/count))
print('Model acc: {0:0.4f}'.format((model_corr_couunt * 100.0)/count))
                          
d = {'texts': texts, 'gold': gold, 'bert': model, 'human': humans, 'ctx': ctx}
df = pd.DataFrame(d)

df.to_csv('results/human_eval.csv', index = False)
df.head(15)


# In[97]:


data = pd.read_csv('results/human_eval.csv')

print('Total samples: ', data.shape[0])

human_corr = data[data['human'] == data['gold']]
model_corr = data[data['bert'] == data['gold']]


human_corr_indx = set(list(human_corr.index.values))
model_corr_indx = set(list(model_corr.index.values))

print('Size of intersection is {0}'.format(len(human_corr_indx & model_corr_indx)))

human_not_model_indx = human_corr_indx.difference(model_corr_indx)
model_not_human_indx = model_corr_indx.difference(human_corr_indx)

print('Size of human_not_model is {0}'.format(len(human_not_model_indx)))
print('Size of model_not_human is {0}'.format(len(model_not_human_indx)))

human_not_model = data.iloc[list(human_not_model_indx)]
model_pred = data.iloc[list(human_not_model_indx)]['bert']
human_not_model['Gen Prediction'] = model_pred

model_not_human = data.iloc[list(model_not_human_indx)]
human_pred = data.iloc[list(model_not_human_indx)]['human']
model_not_human['Human Prediction'] = human_pred

human_not_model.to_csv('results/human_not_model.csv', index = False)
model_not_human.to_csv('results/model_not_human.csv', index = False)

assert(human_not_model.shape[0] == len(human_not_model_indx))
assert(model_not_human.shape[0] == len(model_not_human_indx))



none_corr = data[data['bert'] != data['gold']]
none_corr = none_corr[none_corr['human'] != none_corr['gold']]

print('None correct: ', none_corr.shape[0])
none_corr.to_csv('results/none_correct.csv', index = False)


# In[ ]:




