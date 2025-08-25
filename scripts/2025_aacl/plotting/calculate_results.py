import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_rel 
from tqdm import tqdm

from langcodes import *

with_en=True

# Constants
MASAKHA_DICT = {
    "Masakha": ['bam', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos', 'nya', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul'],
}
MASAKHA_UNSEEN = ['bbj', 'pcm']
#-----------NLI-----------
NLI_DICT = {
    "XNLI": ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'],
    "AfriXNLI": ['amh', 'ewe', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna', 'sot', 'swa', 'twi', 'wol', 'xho', 'yor', 'zul'],
    "AmericasNLI": ['aym', 'bzd', 'cni', 'gn', 'hch', 'nah', 'oto', 'quy', 'shp', 'tar']
}
NLI_LANGS = [
    lang for lang in NLI_DICT['XNLI'] 
    if Language.get(lang).to_alpha3() not in NLI_DICT['AfriXNLI'] + ["mya", "ind"]
    ] + NLI_DICT['AfriXNLI'] + ["mya", "ind"]

#-----------POS-----------
POS_DICT = {
    "UDPOS": ['af', 'ar', 'bg', 'de', 'el', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'kk', 'ko', 'lt', 'mr', 'nl', 'pl', 'pt', 'ro', 'ru', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'wo', 'yo', 'zh'],
}
POS_LANGS = [
    lang for lang in POS_DICT['UDPOS'] 
    if Language.get(lang).to_alpha3() not in MASAKHA_DICT['Masakha']
    ] + MASAKHA_DICT['Masakha']

#-----------NER-----------
NER_DICT = {
    "Wikiann": ['af', 'ar', 'az', 'bg', 'bn', 'de', 'el', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'lt', 'ml', 'mr', 'ms', 'my', 'nl', 'pa', 'pl', 'pt', 'qu', 'ro', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yo', 'zh'],
}
NER_LANGS = [
    lang for lang in NER_DICT['Wikiann'] 
    if Language.get(lang).to_alpha3() not in MASAKHA_DICT['Masakha']
    ] + MASAKHA_DICT['Masakha']

#-----------QA-----------
QA_DICT = {
    "XQuAD": ['ar', 'de', 'el', 'es', 'hi', 'ru', 'th', 'tr', 'vi', 'zh']
}
QA_LANGS = [
    lang for lang in QA_DICT['XQuAD'] 
    if Language.get(lang).to_alpha3() != "ind"
    ] + ['ind']

SEEDS = [42, 66, 23, 17]

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

results_dir = "scripts/2025_aacl/results"
exps_directories = [d for d in os.listdir(results_dir)]
SORT_COLUMNS = sorted(exps_directories)

def check_exp_langs(df: pd.DataFrame):
    task_dict = {
        "xtreme_r.udpos": (POS_DICT, "final_eval_{}_accuracy"),
        "xnli": (NLI_DICT, "final_eval_{}_accuracy"),
        "wikiann": (NER_DICT, "final_eval_{}_f1"),
        "xquad": (QA_DICT, "eval_{}_f1"),
    }
    eval_dict, col_template = task_dict[true_task]
    df_cols = set(df.columns)

    for eval_set, langs in eval_dict.items():
        expected_cols = {col_template.format(lang) for lang in langs}
        missing_cols = expected_cols - df_cols
        if missing_cols:
            return False
    return True

def check_exp_seeds(df: pd.DataFrame):
    messages = []
    missing = set(SEEDS) - set(df['seed'].unique())
    if missing:
        return False
    return True

if with_en:
    for langs in POS_LANGS, NER_LANGS, NLI_LANGS, QA_LANGS:
        langs += ['same']

failed_exp = []
for model in tqdm(["xlm-roberta-base", "bert-base-multilingual-cased"], desc="Model"):
    for task in tqdm(['xtreme_r.udpos', 'xnli', 'wikiann', 'xquad', 'americasnli', 'masakha_unseen_ner', 'masakha_unseen_pos'], desc=f"Task for {model}", leave=False):
        tqdm.write(f"Running {model} on {task}")
        per_experiment_results = []
        
        if task == "xtreme_r.udpos":
            cols_filter = [f"final_eval_{lang}_accuracy" for lang in POS_LANGS]
        elif task == "wikiann":
            cols_filter = [f"final_eval_{lang}_f1" for lang in NER_LANGS]
        elif task == "xnli":
            cols_filter = [f"final_eval_{lang}_accuracy" for lang in NLI_LANGS]
        elif task == "americasnli":
            cols_filter = [f"final_eval_{lang}_accuracy" for lang in NLI_DICT['AmericasNLI']]
        elif task == "masakha_unseen_ner" or task == "masakha_unseen_pos":
            cols_filter = [f"final_eval_{lang}_accuracy" for lang in MASAKHA_UNSEEN]
        elif task == "xquad":
            cols_filter = [f"eval_{lang}_f1" for lang in QA_LANGS]
        cols_filter += ['seed']
        
        for exp in exps_directories:
            if exp in [
                'nllb_only', 'opus_only', 'most_uriel_5', 'most_uriel_10', 'most_uriel_20', 'most_uriel_40', 
                'least_uriel_5', 'least_uriel_10', 'least_uriel_20', 'least_uriel_40'
                ]:
                continue
            try:
                if task == "americasnli":
                    true_task = "xnli"
                elif task == "masakha_unseen_ner":
                    true_task = "wikiann"
                elif task == "masakha_unseen_pos":
                    true_task = "xtreme_r.udpos"
                else:
                    true_task = task
                aligner = "baseline" if exp == "baseline" else "before_noaligner"
                df = pd.read_csv(f"{results_dir}/{exp}/{model}__mix_opus100_nllb__{aligner}__{true_task}.csv")
                is_passed = check_exp_langs(df) and check_exp_seeds(df)
                if not is_passed:
                    raise Exception()
                df = df[cols_filter]
                df = df[df['seed'].isin(SEEDS)]
            except Exception as e:
                failed_exp.append(exp)
                continue
                        
            accuracy_cols = [
                col for col in df.columns
                if "eval_" in col
                and not col.endswith("avg_accuracy")
                # and not col.endswith("same_accuracy")
            ]
            # # Compute mean and std per column
            df[accuracy_cols] *= 100
            df['final_eval_avg_performance'] = df[accuracy_cols].mean(axis=1)
            
            seeds = df['seed'].tolist()
            list_acc_final = df['final_eval_avg_performance'].tolist()
            
            mean_per_lang = df[accuracy_cols + ['final_eval_avg_performance']].mean(axis=0)
            std_per_lang = df[accuracy_cols + ['final_eval_avg_performance']].std(axis=0)

            # Format as "mean ± std"
            formatted = {
                col: f"{mean_per_lang[col]:.2f} \scriptsize $\pm$ {std_per_lang[col]:.2f}"
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
            results_df["p_val_xt_afri"] = get_pvalues_against(results_df, "xt_afri")
            results_df["p_val_xt_only"] = get_pvalues_against(results_df, "xt_only")
            results_df["p_val_afri_only"] = get_pvalues_against(results_df, "afri_only")

            # Print LaTeX format
            results_df.to_csv(f"scripts/2025_aacl/plotting/summary_{model}_{task}.csv", index=False)
print(failed_exp)