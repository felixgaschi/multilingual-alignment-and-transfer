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

for exp in exps_directories:
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
            
            try:
                df1 = pd.read_csv(f"felix_results_lang_selection/{exp}/{model}__{true_task}.csv")
                df1_filtered = df1[df1['seed'].isin([42, 66])]
                df1_filtered = df1_filtered[cols_filter]
                
                assert(len(df1_filtered) == 2) 
            except Exception as e:
                print(f"Not finished: [FELIX] {exp} -- {model} -- {task}")
                continue

for exp in exps_directories:
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
            
            try:
                df2 = pd.read_csv(f"phuoc_seed_31/{exp}/{model}__opus100__before_dico__{true_task}.csv")
                df2_filtered = df2[df2['seed'] == 31]
                df2_filtered = df2_filtered[cols_filter]
                assert(len(df2_filtered) == 1) 
            except Exception as e:
                print(f"Not finished: [PHUOC] {exp} -- {model} -- {task}")
                continue
