import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_rel 


# Constants
XNLI_LANGS = [
    "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "th", "tr", "vi", "zh",
    # 'amh', 'eng', 'ewe', 'fra', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna',
    # 'sot', 'swa', 'twi', 'wol', 'xho', 'yor', 'zul'
]

AFRIXNLI_LANGS = [
    #"ar", "bg", "de", "el", "es", "fr", "hi", "ru", "th", "tr", "vi", "zh",
    'amh', 'ewe', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna',
    'sot', 'swa', 'twi', 'wol', 'xho', 'yor', 'zul'
]

UDPOS_LANGS = ['bg', 'cs', 'de', 'es', 'lv', 'af', 'ar', 'ca', 'da', 'el', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'it', 'ja', 'ko', 'lt', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'ta', 'th', 'tr', 'uk', 'vi', 'zh']


MASAKHAPOS_LANGS = [
    'bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos', 'nya', 'pcm',
    'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul'
]

SORT_COLUMNS = ["12_langs", "34_langs", "DIVERSE_URIEL_FEATURAL_3", "DIVERSE_URIEL_FEATURAL_7", "DIVERSE_URIEL_FEATURAL_14",
                "LEAST_DIVERSE_URIEL_FEATURAL_3", "LEAST_DIVERSE_URIEL_FEATURAL_7", "LEAST_DIVERSE_URIEL_FEATURAL_14",
                "random_distinct_family_3", "random_distinct_family_7", "random_distinct_family_14", "indo_3", "indo_7", "indo_14",
                "low_res", "mix_res", "high_res", "random_langs_with_seed_3", "random_langs_with_seed_7", "random_langs_with_seed_14"]

SEEDS = [31, 42, 66]

def get_pvalues_against(results_df, base_exp):
    """
    Compare all entries against base_exp for the same task and model.
    Returns: list of (compared_exp, p_value, is_significant)
    """
    p_values = []

    # Filter for base experiment
    base_row = results_df[results_df['exp'] == base_exp]
    
    if len(base_row) == 0:
        return [-1 for i in range(len(results_df))]

    base_row = base_row.iloc[0]
    base_zipped = sorted(zip(base_row['seeds'], base_row['list_final_avg']))
    base_seeds, base_vals = zip(*base_zipped)  # base_seeds is now the canonical order

    # Iterate over other rows
    for _, row in results_df.iterrows():
        if row['exp'] == base_exp:
            p_values.append(-1)
            continue
        other_zipped = sorted(zip(row['seeds'], row['list_final_avg']))
        other_seeds, other_vals = zip(*other_zipped)

        t_stat, p_val = ttest_rel(base_vals, other_vals)
        p_values.append(p_val)

    return p_values

exps_directories = [d for d in os.listdir("felix_results_lang_selection")]

for model in ["xlm-roberta-base", "bert-base-multilingual-cased"]:
    for task in ['udpos', 'masakhapos', 'xnli', 'afri-xnli']:
        per_experiment_results = []
        if task.endswith("xnli"):
            true_task = "xnli"
        else:
            true_task = "udpos"
        
        if task == "udpos":
            cols_filter = [f"final_eval_{lang}_accuracy" for lang in UDPOS_LANGS]
        elif task == "masakhapos":
            cols_filter = [f"final_eval_{lang}_accuracy" for lang in MASAKHAPOS_LANGS]
        elif task == "xnli":
            cols_filter = [f"final_eval_{lang}_accuracy" for lang in XNLI_LANGS]
        elif task == "afri-xnli":
            cols_filter = [f"final_eval_{lang}_accuracy" for lang in AFRIXNLI_LANGS]
        cols_filter += ['seed']
        
        for exp in exps_directories:
            try:
                df1 = pd.read_csv(f"felix_results_lang_selection/{exp}/{model}__{true_task}.csv")
            
                df1_filtered = df1[df1['seed'].isin([42, 66])]

                df2 = pd.read_csv(f"phuoc_seed_31/{exp}/{model}__opus100__before_dico__{true_task}.csv")
                df2_filtered = df2[df2['seed'] == 31]
                df1_filtered = df1_filtered[cols_filter]
                df2_filtered = df2_filtered[cols_filter]
                assert(len(df1_filtered) == 2)
                assert(len(df2_filtered) == 1)
            except Exception as e:
                continue
            
            combined = pd.concat([df1_filtered, df2_filtered], ignore_index=True)
            
            accuracy_cols = [
                col for col in combined.columns
                if col.startswith("final_eval_")
                and col.endswith("_accuracy")
                and not col.endswith("avg_accuracy")
                and not col.endswith("same_accuracy")
            ]

            # # Compute mean and std per column
            combined[accuracy_cols] *= 100
            combined['final_eval_avg_accuracy'] = combined[accuracy_cols].mean(axis=1)
            
            seeds = combined['seed'].tolist()
            list_acc_final = combined['final_eval_avg_accuracy'].tolist()
            
            mean_per_lang = combined[accuracy_cols + ['final_eval_avg_accuracy']].mean(axis=0)
            std_per_lang = combined[accuracy_cols + ['final_eval_avg_accuracy']].std(axis=0)

            # Format as "mean ± std"
            formatted = {
                col: f"{mean_per_lang[col]:.2f} $\pm$ {std_per_lang[col]:.2f}"
                for col in mean_per_lang.index
            }
            formatted['exp'] = exp
            formatted['task'] = task
            formatted['seeds'] = seeds
            formatted['list_final_avg'] = list_acc_final

            per_experiment_results.append(formatted)
            
        # Create final DataFrame
        results_df = pd.DataFrame(per_experiment_results)
        results_df['exp'] = pd.Categorical(results_df['exp'], categories=SORT_COLUMNS, ordered=True)

        # Then sort the entire DataFrame by 'exp'
        results_df = results_df.sort_values('exp')
        
        if len(results_df) != 0:
            results_df["p_val_12_langs"] = get_pvalues_against(results_df, "12_langs")
            results_df["p_val_34_langs"] = get_pvalues_against(results_df, "34_langs")

            # Print LaTeX format
            results_df.to_csv(f"summary_{model}_{task}.csv", index=False)
 