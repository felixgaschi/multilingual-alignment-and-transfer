import pandas as pd
import numpy as np
import tabulate

model_to_layers = {
    "xlm-roberta-base": 12,
    "bert-base-multilingual-cased": 12,
    "distilbert-base-multilingual-cased": 6
}

def print_results(model: str, method: str):
    fname = f"scripts/2025_alignfreeze_continuation/data/single_layer_realignment/{model}__opus100__{method}.csv"

    if method == "single_layer_realignment":
        prefix = "before_realign_only"
    elif method == "single_layer_freezing":
        prefix = "freeze_realign_unfreeze"
    else:
        raise NotImplementedError(f"method {method} not expected by print_results")

    df = pd.read_csv(fname)

    langs = list(map(lambda x: x.split("_")[2], filter(lambda x: x.startswith("final_eval") and "same" not in x and "avg" not in x, df.columns)))
    langs.sort()
    
    n_layers = model_to_layers[model]

    headers = ["lang", *map(lambda x: f"layer {x}", range(n_layers + 1))]

    table = []

    for lang in ["avg", *langs]:
        line = [lang]
        results = []
        for i in range(n_layers + 1):
            values = df[df.method == f"{prefix}_{i}_{i+1}_dico"][f"final_eval_{lang}_accuracy"]
            results.append(np.mean(values))
        i_max = np.argmax(results)
        for i, res in enumerate(results):
            line.append(f"{'**' if i == i_max else ''}{res * 100:.1f}{'**' if i == i_max else ''}")
        table.append(line)
    
    print(tabulate.tabulate(table, headers=headers, tablefmt="github"))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--method", type=str, default="single_layer_realignment")
    args = parser.parse_args()    

    print_results(args.model, args.method)