import pandas as pd
import numpy as np
import tabulate

import os

res_dir = "./scripts/2025_aacl/plotting/mean_and_std_res"
tasks = ["xnli", "xtremer.udpos", "wikiann", "xquad"]
for task in tasks:
    print(f"==========================={task.upper()}===========================")
    with open("/home/leelab-alignfreeze2/nlp_project/scripts/2025_aacl/plotting/results.md", "a") as f:
        f.write(f"# {task.upper()}")
        f.write("\n")
    for file in os.listdir(res_dir):
        #Extract info
        if not (file.endswith(".csv") and task in file):
            continue
        model_name, task, seed_str, _ = file.split("_")
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

        # print(f"{model_name} with {seed_str}")
        # print(tabulate.tabulate(table, headers=headers, tablefmt="github"))
        # print()
        with open("/home/leelab-alignfreeze2/nlp_project/scripts/2025_aacl/plotting/results.md", "a") as f:
            f.write(f"{model_name} with {seed_str}\n\n")
            f.write(tabulate.tabulate(table, headers=headers, tablefmt="github"))
            f.write("\n\n\n")