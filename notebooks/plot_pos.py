import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

dataset = "aclImdb" # "RACE" "aclImdb"
model_name = "roberta-base"
max_context_size = 249
measure = "acc" # acc, ppl, loss, prob, rank
start = 1
end = 1

data_dir = "../output_dir"
data_path = Path(data_dir)
assert data_path.exists(), f'Error: {data_path} does not exist.'
context_sizes = [1,2,3] + list(range(5,30,5)) + list(range(30, max_context_size,10))
context_sizes = [str(cs) for cs in context_sizes]

selected_context_sizes = list(range(10,max_context_size,10))
selected_context_sizes = [str(e) for e in selected_context_sizes]
selected_indices = [0] + [context_sizes.index(e) for e in selected_context_sizes]

context_sizes = np.array(context_sizes)[selected_indices]
data_dict = {}
#tag_list = ["FREQ", "INFREQ", "RAND", "VERB","PROPN","NOUN","PUNCT","ADP","ADJ","ADV","DET","NUM","PUNCT","PRON"]
#plot_tag = "all"

setting = 2 #2, 3

if setting == 1:
    tag_list = ["FREQ", "INFREQ", "RAND", "VERB","PROPN","NOUN","PUNCT","ADP","ADJ","ADV","DET","NUM","PUNCT","PRON"]
    plot_tag = "all"
elif setting == 2:
    tag_list = ["VERB","PROPN","NOUN","PUNCT","ADP","ADJ","ADV","DET","NUM","PUNCT","PRON"]
    plot_tag = "pos"
elif setting == 3:
    tag_list = ["FREQ", "INFREQ"]
    plot_tag = "_".join([t.split(" ")[0].lower() for t in tag_list])
else:
    raise ValueError("Invalid setting!")

#for tag in ["FREQ", "INFREQ", "RAND"]:
for tag in tag_list:
    for span_length in range(start, end + 1):
        data_file = data_path / f'{model_name}-{dataset}-{tag}' / f'all_{measure}_context_size_{span_length}.npy'
        data = np.load(data_file)
        data = data[:, selected_indices]
        data_mean = data.mean(axis=0)

        assert len(data_mean) == len(context_sizes)
        data_dict[tag] = data_mean

df = pd.DataFrame.from_dict(data_dict, orient='index')

if setting == 1:
    style = ['+-','o-','.--'] + ['-'] * (len(tag_list) - 3)
elif setting == 2:
    style = ['-'] #['.--'] + ['-'] * (len(tag_list) - 1)
elif setting == 3:
    style = ['-']#['+-', 'o-', '.--']

ax = df.T.plot(xticks=range(len(context_sizes)), figsize=(16,12), style=style)
ax.set_xticklabels(context_sizes, fontsize=40)
ax.set_xlabel("Context window size", fontsize=40)
x_values = ['1', '', '', '', '', '50', '', '', '', '', '100', '', '', '', '', '150', '', '', '', '', '200', '', '', '', '']
ax.set_xticklabels(x_values, fontsize=40)

plt.yticks(fontsize=40)
measure_label = None
if measure == "acc":
    measure_label = "Accuracy"
elif measure == "ppl":
    measure_label = "Perplexity"
elif measure == "loss":
    measure_label = "Loss"
elif measure == "prob":
    measure_label = "Probability"
elif  measure == "rank":
    measure_label = "Rank"

ax.set_ylabel(measure_label, fontsize=40)
plt.legend(title="Part-of-speech tag", loc='best', title_fontsize=28, fontsize=28, ncol=2)
#plt.show()
plot_path = "../plots/"
plt.savefig(plot_path + f'z_{dataset}_{measure}_{start}_{end}_{plot_tag}_{model_name.split("-")[0].lower()}.pdf', bbox_inches='tight')

