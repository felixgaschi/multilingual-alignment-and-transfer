import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import re

model_list = ["xlm-roberta-base", "bert-base-multilingual-cased", "distilbert-base-multilingual-cased"]
results_dir = "./data/results/"
save_results = True

# 3 subplots for 3 models.
fig, axes = plt.subplots(len(model_list), 1, figsize=(15, 25))
fig.suptitle('Models performance across languages', fontsize=16)

for model, ax in zip(model_list, axes):
    df_list = []
    # Gather results from all methods for the current model
    for method in os.listdir(results_dir):
        # Skip methods without realignment on Japanese and Chinese for now. (half|onethird|twothird|onesixth|fivesixth)(_noembs|_withembs)?(_adjacent|_discrete)?
        if "nojazh" in method or not re.search(r"^realign_random_(half|onethird|twothird|onesixth|fivesixth)_dico$", method): 
            if method not in ["baseline", "before_dico", "freeze_realign_unfreeze_dico", "freeze_realign_unfreeze_last_half_dico"]:
                print(f"Skipping {method}")
                continue
        method_dir = os.path.join(results_dir, method)
        for file in os.listdir(method_dir):
            if file.endswith(".csv") and file.startswith(model):
                try:
                    df = pd.read_csv(os.path.join(method_dir, file))
                except Exception as e:
                    print(f"Error reading {file}: {e}, skipping...")
                    continue
                df_list.append(df)
    df_merged = pd.concat(df_list, ignore_index=True, sort=False)
    seed_list = sorted(df_merged.seed.unique().tolist())

    final_columns = [col for col in df_merged.columns if "final" in col]

    # Create a long-form DataFrame for seaborn
    long_df = pd.melt(df_merged, id_vars=['method'], value_vars=final_columns,
                        var_name='language', value_name='accuracy')
    long_df['language'] = long_df['language'].str.replace('final_eval_', '').str.replace('_accuracy', '')

    # This save results part mainly for me to format it to Github table format later, so you might not need it.
    if save_results:
        # Define the directory and filename
        directory = f"scripts/2025_alignfreeze_continuation/distillation/plotting/mean_and_std_res"
        seed_str = "seed-" + "-".join(map(str, seed_list))
        filename = f"{model}_{seed_str}_mean-std-results.csv"
        filepath = os.path.join(directory, filename)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save the DataFrame to a CSV file
        output_df = long_df.groupby(['language', 'method'])['accuracy'].agg(['mean', 'std']).unstack('method')
        # Swap levels and sort columns to get the desired super/sub-column structure
        output_df.columns = output_df.columns.swaplevel(0, 1)
        output_df = output_df.sort_index(axis=1)

        output_df.to_csv(filepath, index=True)
        print(f"DataFrame saved to {filepath}")
    print(long_df['method'].unique())
    sns.boxplot(
        x='language', 
        y='accuracy', 
        hue='method', 
        data=long_df,
        ax=ax
    )
    ax.set_title(f"{model}_{df_merged.task.unique()[0]}_{seed_list}")
    ax.set_xlabel("Language")
    ax.set_ylabel("Accuracy")

# plt.xticks(rotation=45)
plt.tight_layout()
# If you ever change the figure size and find that the title is cut off, you can use the following line to adjust the top margin.
plt.subplots_adjust(top=0.96)
plt.savefig(f"scripts/2025_alignfreeze_continuation/distillation/plotting/imgs/boxplot.png", dpi=300, bbox_inches='tight')