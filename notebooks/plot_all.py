import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

dataset = "RACE" # "RACE" "aclImdb"
max_context_size = 249
measure = "loss" # acc, ppl, loss, prob, rank
start = 1
end = 1
start_label_index = 0
data_dir = "../../output_dir"
data_path = Path(data_dir)
assert data_path.exists(), f'Error: {data_path} does not exist.'
context_sizes = [1,2,3] + list(range(5,30,5)) + list(range(30, max_context_size,10))
context_sizes = [str(cs) for cs in context_sizes]
context_sizes = context_sizes[start_label_index:]

data_dict = {}
# for model_name in ["bert-base-cased", "roberta-base", "xlm-mlm-en-2048", "xlnet-base-cased"]:
for model_name in ["bert-base-cased", "roberta-base", "xlnet-base-cased"]:
    for span_length in range(start, end + 1):
        data_file = data_path / f'{model_name}-{dataset}' / f'all_{measure}_context_size_{span_length}.npy'
        data = np.load(data_file)  # dim:(examples, context sizes)
        data = data[:, start_label_index:]
        data_mean = data.mean(axis=0)  # Taking mean across different examples, getting one value per context size
        data_mean = data.mean(axis=0)  # Taking mean across different examples, getting one value per context size
        assert len(data_mean) == len(context_sizes)
        data_dict[model_name.split("-")[0].upper()] = data_mean

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
plt.legend(title="Model")
#plt.show()
filename = f'{dataset}_{measure}_{start}_{end}.pdf'
plot_path = "../../plots/"
plt.savefig(plot_path + filename, bbox_inches='tight')


