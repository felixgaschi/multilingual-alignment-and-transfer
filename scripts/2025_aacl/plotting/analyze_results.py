import pandas as pd
import numpy as np
import tabulate

import os

from collections import defaultdict

res_dir = "./scripts/2025_aacl/plotting/mean_and_std_res"
tasks = ["xnli", "xtremer.udpos", "wikiann", "xquad"]
overall_dict = defaultdict(lambda: defaultdict(list))
with open("./scripts/2025_aacl/plotting/results.md", "w") as f:
        f.write("")

for task in tasks:
    print(f"==========================={task.upper()}===========================")
    with open("./scripts/2025_aacl/plotting/results.md", "a") as f:
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

            max_mean = max([cur_line.loc[(method, "mean")] for method in methods if not pd.isna(cur_line.loc[(method, "std")])])
            for method in methods:
                mean = cur_line.loc[(method, "mean")]
                std = cur_line.loc[(method, "std")]
                bold = '**' if mean == max_mean else ''
                line.append(f"{bold}{mean * 100:.2f} ± {std * 100:.2f}{bold}")
                if lang == "avg":
                    overall_dict[model_name][method].append(mean * 100)
            table.append(line)

        # print(f"{model_name} with {seed_str}")
        # print(tabulate.tabulate(table, headers=headers, tablefmt="github"))
        # print()
        with open("./scripts/2025_aacl/plotting/results.md", "a") as f:
            f.write(f"### {model_name} with {seed_str}\n\n")
            f.write(tabulate.tabulate(table, headers=headers, tablefmt="github"))
            f.write("\n\n\n")

with open("./scripts/2025_aacl/plotting/results.md", "a") as f:
    f.write(f"# Overall\n\n")

    # Group models by list length
    length_groups = defaultdict(list)
    
    for model_name in overall_dict:
        for method, values in overall_dict[model_name].items():
            list_len = len(values)
            length_groups[list_len].append(model_name)
            break  # Group model once by the first method's length

    # Generate one table per list length
    for list_len, models in sorted(length_groups.items()):
        f.write(f"# Overall (List Length = {list_len})\n\n")

        all_methods = sorted({m for model in models for m in overall_dict[model]})
        table = []

        for model_name in models:
            average_dict = {
                method: sum(values) / len(values)
                for method, values in overall_dict[model_name].items()
                if values
            }

            row_values = [average_dict.get(method, "-") for method in all_methods]

            # Bold the max value
            numeric_vals = [(i, v) for i, v in enumerate(row_values) if isinstance(v, (int, float))]
            if numeric_vals:
                max_idx, _ = max(numeric_vals, key=lambda x: x[1])
                row_values[max_idx] = f"**{row_values[max_idx]:.2f}**"

            # Format other floats
            row_values = [
                f"{v:.2f}" if isinstance(v, float) and not str(v).startswith("**") else v
                for v in row_values
            ]

            table.append([model_name] + row_values)

        headers = ["Model"] + all_methods
        f.write(tabulate.tabulate(table, headers=headers, tablefmt="github"))
        f.write("\n\n\n")