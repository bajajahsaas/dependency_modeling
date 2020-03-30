import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
#plt.style.use('ggplot')

'''
Plot for all models (particular POS). Context size with buckets
'''


dataset = "aclImdb" # "RACE" "aclImdb"
max_context_size = 249
measure = "acc" # acc, ppl, loss, prob, rank
start = 1
end = 1
data_dir = "../../output_dir"
data_path = Path(data_dir)
assert data_path.exists(), f'Error: {data_path} does not exist.'
context_sizes = [1,2,3] + list(range(5,30,5)) + list(range(30, max_context_size,10))
context_sizes = [str(cs) for cs in context_sizes]

selected_context_sizes = list(range(10,max_context_size,10))
selected_context_sizes = [str(e) for e in selected_context_sizes]
selected_indices = [0] + [context_sizes.index(e) for e in selected_context_sizes]

context_sizes = np.array(context_sizes)[selected_indices]
model_name_dict = {}
model_name_dict["bert-base-cased"] = "BERT"
model_name_dict["roberta-base"] = "RoBERTa"
model_name_dict["xlm-mlm-en-2048"] = "XLM"
model_name_dict["xlnet-base-cased"] = "XLNet"

tags = ["PROPN", "NOUN", "ADJ", "ADV", "DET", "NUM"]
for pos in tags:
    data_dict = {}
    for model_name in ["bert-base-cased", "roberta-base", "xlm-mlm-en-2048", "xlnet-base-cased"]:
        for span_length in range(start, end + 1):
            data_file = data_path / f'{model_name}-{dataset}-{pos}' / f'all_{measure}_context_size_{span_length}.npy'
            data = np.load(data_file)
            data = data[:, selected_indices]
            data_mean = data.mean(axis=0)

            assert len(data_mean) == len(context_sizes)
            data_dict[model_name_dict[model_name]] = data_mean

    df = pd.DataFrame.from_dict(data_dict, orient='index')
    ax = df.T.plot(xticks=range(len(context_sizes)),
                   figsize=(16,12))
    ax.set_xticklabels(context_sizes)

    ax.set_xlabel("Context window size", fontsize=40)
    x_values = ['1', '', '', '', '', '50', '', '', '', '', '100', '', '', '', '', '150', '', '', '', '', '200', '', '', '', '']
    ax.set_xticklabels(x_values, fontsize=40)

    #y_values = ['', '', '10', '20', '30', '40', '50', '60', '70', '80', '90']
    #ax.set_yticklabels(y_values, fontsize=30)
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

    ax.set_ylabel(measure_label + "-" + pos, fontsize=40)
    plt.legend(title="Model", loc='best', title_fontsize=40, fontsize=40)
    #plt.show()
    plot_path = "../../plots/"
    plt.savefig(plot_path + f'new_{dataset}_{pos}_{measure}_{start}_{end}.pdf', bbox_inches='tight')