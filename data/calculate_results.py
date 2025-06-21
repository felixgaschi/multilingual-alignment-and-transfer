import pandas as pd
import numpy as np
import os 


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


SEEDS = [31, 42, 66]

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
            
            mean_per_lang = combined[accuracy_cols + ['final_eval_avg_accuracy']].mean(axis=0)
            std_per_lang = combined[accuracy_cols + ['final_eval_avg_accuracy']].std(axis=0)

            # Format as "mean ± std"
            formatted = {
                col: f"{mean_per_lang[col]:.2f} $\pm$ {std_per_lang[col]:.2f}"
                for col in mean_per_lang.index
            }
            formatted['exp'] = exp
            formatted['task'] = task

            per_experiment_results.append(formatted)
            
        # Create final DataFrame
        results_df = pd.DataFrame(per_experiment_results)

        # Print LaTeX format
        results_df.to_csv(f"summary_{model}_{task}.csv", index=False)
 