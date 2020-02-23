import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

model_name = "xlm-mlm-en-2048" #""xlnet-base-cased" #"roberta-base" # "bert-base-cased"
dataset = "RACE" # "RACE" "aclImdb"
max_context_size = 249
measure = "rank" # acc, ppl, loss, prob, rank
start = 1
end = 10
start_label_index = 0
data_dir = "../output_dir"
data_path = Path(data_dir)
assert data_path.exists(), f'Error: {data_path} does not exist.'
context_sizes = [1,2,3] + list(range(5,30,5)) + list(range(30, max_context_size,10))
context_sizes = [str(cs) for cs in context_sizes]
context_sizes = context_sizes[start_label_index:]
data_dict = {}
for span_length in range(start, end + 1):
    data_file = data_path / f'{model_name}-{dataset}' / f'all_{measure}_context_size_{span_length}.npy'
    data = np.load(data_file)
    data = data[:,start_label_index:]
    data_mean = data.mean(axis=0)

    assert len(data_mean) == len(context_sizes)
    data_dict[span_length] = data_mean

df = pd.DataFrame.from_dict(data_dict, orient='index')
ax = df.T.plot(xticks=range(len(context_sizes)), figsize=(16,12))
ax.set_xticklabels(context_sizes)
ax.set_xlabel("Context size")

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

ax.set_ylabel(measure_label)
plt.legend(title="Span length")
#plt.show()

plt.savefig(f'{model_name}_{dataset}_{measure}_{start}_{end}.pdf', bbox_inches='tight')

