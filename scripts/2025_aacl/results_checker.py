import pandas as pd
import numpy as np
import os 
from tqdm import tqdm

from langcodes import *

# Constants
MASAKHA_DICT = {
    "Masakha": ['bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos', 'nya', 'pcm', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul'],
}
#-----------NLI-----------
NLI_DICT = {
    "XNLI": ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'],
    "AfriXNLI": ['amh', 'eng', 'ewe', 'fra', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna', 'sot', 'swa', 'twi', 'wol', 'xho', 'yor', 'zul'],
    "AmericasNLI": ['aym', 'bzd', 'cni', 'gn', 'hch', 'nah', 'oto', 'quy', 'shp', 'tar']
}
NLI_LANGS = [
    lang for lang in NLI_DICT['XNLI'] 
    if Language.get(lang).to_alpha3() not in NLI_DICT['AfriXNLI'] + NLI_DICT['AmericasNLI'] + ["mya", "ind"]
    ] + NLI_DICT['AfriXNLI'] + NLI_DICT['AmericasNLI'] + ["mya", "ind"]

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

results_dir = "scripts/2025_aacl/results"
exps_directories = [d for d in os.listdir(results_dir)]

def check_exp_langs(df: pd.DataFrame):
    messages = []
    task_dict = {
        "xtreme_r.udpos": (POS_DICT, "final_eval_{}_accuracy"),
        "xnli": (NLI_DICT, "final_eval_{}_accuracy"),
        "wikiann": (NER_DICT, "final_eval_{}_f1"),
        "xquad": (QA_DICT, "eval_{}_f1"),
    }
    eval_dict, col_template = task_dict[task]
    df_cols = set(df.columns)

    for eval_set, langs in eval_dict.items():
        expected_cols = {col_template.format(lang) for lang in langs}
        missing_cols = expected_cols - df_cols
        if missing_cols:
            messages.append(f"Missing: {eval_set}")
    return messages

def check_exp_seeds(df: pd.DataFrame):
    messages = []
    missing = set(SEEDS) - set(df['seed'].unique())
    duplicates = df[df.duplicated('seed', keep=False)]
    if missing:
        messages.append(f"Misisng seeds: {sorted(missing)}")
    if not duplicates.empty:
        duplicate_seeds = sorted(duplicates['seed'].unique().tolist())
        messages.append(f"Duplicate seeds: {duplicate_seeds}")
    return messages

checker = dict()
for exp in tqdm(exps_directories):
    if exp in [
        'nllb_only', 'opus_only', 'most_uriel_5', 'most_uriel_10', 'most_uriel_20', 'most_uriel_40', 
        'least_uriel_5', 'least_uriel_10', 'least_uriel_20', 'least_uriel_40'
        ]:
        continue
    for model in ["xlm-roberta-base", "bert-base-multilingual-cased"]:
        for task in ['xtreme_r.udpos', 'xnli', 'wikiann', 'xquad']: 
            key = f"{exp} -- {model} -- {task}"
            messages = []
            aligner = "before_noaligner" if exp != "baseline" else "baseline"
            file_dir = f"{results_dir}/{exp}/{model}__mix_opus100_nllb__{aligner}__{task}.csv"
            
            try:
                df = pd.read_csv(file_dir)
            except Exception as e:
                messages.append(f"Missing file")
            else:            
                if len(df) == 0:
                    messages.append("Empty df")
                else:
                    messages.extend(check_exp_langs(df))
                    messages.extend(check_exp_seeds(df))
            checker[key] = messages
completed = [k for k, v in checker.items() if not v]
incompleted_main = {}
incompleted_abla = {}

for k, v in checker.items():
    if not v:
        continue
    if k.startswith("abla"):
        incompleted_abla[k] = v
    else:
        incompleted_main[k] = v

with open(f"./scripts/2025_aacl/results_checker.md", "w") as f:
    f.write("# Completed\n")
    for c in completed:
        f.write(f"- {c}\n")

    f.write("\n\n\n# Incompleted\n")
    f.write("\n\n## Incompleted main\n")
    for k, messages in incompleted_main.items():
        f.write(f"- {k}\n")
        for msg in messages:
            f.write(f"  - {msg}\n")
    f.write("\n\n## Incompleted ablation study\n")
    for k, messages in incompleted_abla.items():
        f.write(f"- {k}\n")
        for msg in messages:
            f.write(f"  - {msg}\n")
            