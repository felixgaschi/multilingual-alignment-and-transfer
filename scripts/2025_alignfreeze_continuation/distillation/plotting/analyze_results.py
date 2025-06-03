import pandas as pd
import numpy as np
import tabulate

import os

res_dir = "scripts/2025_alignfreeze_continuation/distillation/plotting/mean_and_std_res"
for file in os.listdir(res_dir):
    #Extract info
    if file.endswith(".csv"):
        model_name, seed_str, _ = file.split("_")
        df = pd.read_csv(os.path.join(res_dir, file), index_col=0, header=[0,1])

        table = []
        langs = df.index

        methods = df.columns.get_level_values(0).unique()
        headers = ["Language", *[method.replace("_", " ").title() for method in methods]]
        
        for lang in langs:
            line = [lang]
            cur_line = df.loc[lang]

            max_mean = max([cur_line.loc[(method, "mean")] for method in methods])
            for method in methods:
                mean = cur_line.loc[(method, "mean")]
                std = cur_line.loc[(method, "std")]
                bold = '**' if mean == max_mean else ''
                line.append(f"{bold}{mean * 100:.2f} ± {std * 100:.2f}{bold}")
            table.append(line)

    print(f"{model_name} with {seed_str}")
    print(tabulate.tabulate(table, headers=headers, tablefmt="github"))
    print()