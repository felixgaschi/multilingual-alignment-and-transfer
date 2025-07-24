import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

results_dir = "./scripts/2025_aacl/results"
save_results = True
models = ["bert-base-multilingual-cased", "xlm-roberta-base"]
tasks = ["xnli", "xtreme_r.udpos", "wikiann", "xquad"]
figure = None
SEEDS = [42, 66, 17]

def add_df_from_method(df_list, method, task):
    """
    Loads CSV result files for a given method and appends matching DataFrames to df_list.

    Parameters:
    - df_list (list): A list to which the filtered and processed DataFrames will be appended.
    - method (str): The name of the method; used to locate the directory containing result CSVs.

    The function performs the following:
    - Iterates over all CSV files in the corresponding method directory.
    - Skips files that don't start with the expected model prefix or don't end in .csv.
    - Reads each CSV into a DataFrame.
    - Skips the file if the DataFrame doesn't contain the target task.
    - Adds a 'method' column to indicate which method produced the data.
      If 'n_realignment_langs' is present, the method name is suffixed with its value.
    - Appends the processed DataFrame to df_list.
    """
    method_dir = os.path.join(results_dir, method)

    for file_name in os.listdir(method_dir):
        if not (file_name.endswith(".csv") and file_name.startswith(model) and task in file_name):
            continue

        file_path = os.path.join(method_dir, file_name)

        try:
            df = pd.read_csv(file_path)
            df = df[df['seed'].isin(SEEDS)]
            if len(df) < 3:
                raise Exception("Dataframe does not contain enough data.")
            df["method"] = method
            df_list.append(df)

        except Exception as e:
            print(f"[Warning] Failed to read '{file_path}': {e}")

for model in models:
    for task in tasks:
        print(f"==========================={task.upper()}===========================")
        # Initialize list to collect individual method DataFrames
        df_list = []

        # Collect results from each method for the current model
        for method in os.listdir(results_dir):
            add_df_from_method(df_list, method, task)

        print(f"Total method: {len(df_list)}")
        # Merge all DataFrames into one
        df_merged = pd.concat(df_list, ignore_index=True, sort=False)

        # Extract list of seeds used in experiments
        seed_list = sorted(df_merged["seed"].unique())

        # Identify all final evaluation columns
        if task == "xquad":
            final_columns = [col for col in df_merged.columns if "eval" in col]
        else:
            final_columns = [col for col in df_merged.columns if "final" in col]

        # Convert to long-form DataFrame suitable for seaborn plotting
        long_df = pd.melt(
            df_merged,
            id_vars=["method"],
            value_vars=final_columns,
            var_name="language",
            value_name="accuracy"
        )

        # Clean up the language column names
        long_df["language"] = (
            long_df["language"]
            .str.replace("final_eval_", "", regex=False)
            .str.replace("_accuracy", "", regex=False)
        )

        # Remove duplicates just in case
        long_df = long_df.drop_duplicates()

        # This save results part mainly for me to format it to Github table format later, so you might not need it.
        if save_results:
            # Define the directory and filename
            directory = f"./scripts/2025_aacl/plotting/mean_and_std_res"
            seed_str = "seed-" + "-".join(map(str, seed_list))
            filename = f"{model}_{task.replace("_", "")}_{seed_str}_mean-std-results.csv"
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

        if figure:
            if figure == "boxplot":
                print(long_df['method'].unique())
                sns.boxplot(
                    x='language', 
                    y='accuracy', 
                    hue='method', 
                    data=long_df,
                )
                plt.title(f"{model}_{df_merged.task.unique()[0]}_{seed_list}")
                plt.xlabel("Language")
                plt.ylabel("Accuracy")

                # plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"./scripts/2025_aacl/plotting/imgs/boxplot.png", dpi=300, bbox_inches='tight')

            elif figure == "diff_heat_map":
                # Extract the dataset sizes as integers from the 'method' column
                long_df['num_langs'] = long_df['method'].apply(lambda x: int(x.split('_')[-1]))

                # Pivot the table to have languages as index, dataset sizes as columns
                pivot_df = long_df.pivot_table(index='language', columns='num_langs', values='accuracy')
                pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

                first_method_col = sorted(pivot_df.columns)[0]

                # New DataFrame for differences
                diff_df = pivot_df.copy()

                for col in sorted(pivot_df.columns)[1:]:
                    diff_df[col] = pivot_df[col] - pivot_df[first_method_col]

                # Drop the first method's column from the diff_df as its difference to itself is 0
                diff_df = diff_df.drop(columns=[first_method_col])

                # Rename columns for clarity in the heatmap (e.g., 'Diff_vs_3_Size7', 'Diff_vs_3_Size14')
                diff_df.columns = [f'Diff_vs_{first_method_col}_Size{col}' for col in diff_df.columns]

                # --- Plotting the Heatmap ---
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    diff_df,
                    annot=True,      # Show the numerical values on the heatmap
                    fmt=".3f",       # Format the annotations to 3 decimal places
                    cmap="coolwarm", # Color map (coolwarm is good for showing positive/negative differences)
                    linewidths=.5,   # Lines between cells
                    cbar_kws={'label': f'Accuracy Difference from Method with realignment lang set size {first_method_col}'} # Color bar label
                )
                plt.title(f'Accuracy Difference Heatmap (Compared to num langs = {first_method_col})', fontsize=16)
                plt.xlabel('Accuracy Difference', fontsize=12)
                plt.ylabel('Language', fontsize=12)
                plt.tight_layout()
                plt.savefig(f"./scripts/2025_aacl/plotting/imgs/diff_heat.png", dpi=300, bbox_inches='tight')