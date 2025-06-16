#====================In case results is overwritten, extract results from .out file
import os
import pandas as pd
import re

outfile_dir = "outfile"
info_root_str = "INFO:root:"
prefixes = (
    f"{info_root_str}{{'seed':",
    f"{info_root_str}{{'finetuning_steps': 24544,",
    f"{info_root_str}{{'eval_bg_accuracy':",
    f"{info_root_str}{{'eval_same_accuracy':",
    f"{info_root_str}{{'final_eval_bg_accuracy':",
    f"{info_root_str}{{'final_eval_same_accuracy':"
)

def process_data_line(line):
    return eval(line.removeprefix(info_root_str).strip())

for file_name in os.listdir(outfile_dir): 
    data_dict = {}
    file_id = file_name.split(".")[0].split("-")[1]

    # if int(file_id) < 56337807 or int(file_id) == 56337863:
    #     print(f"[SKIPPED] file_name: {file_name}")
    #     continue

    if not int(file_id) == 56372407:
        continue

    file_path = os.path.join(os.getcwd(), outfile_dir, file_name)
    with open(file_path, mode="r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Selected strategy:"):
                match = re.search(r'Selected strategy: (\S+)', line)
                strategy = match.group(1)
            elif line.startswith(prefixes):
                data_dict.update(process_data_line(line))
                
        save_dir = f"data/reg_lang_selection_results/{strategy}/xlm-roberta-base__opus100__before_dico__udpos.csv"
        print(pd.DataFrame([data_dict]))
        df = pd.read_csv(os.path.join(os.getcwd(), save_dir))
        df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
        df.to_csv(save_dir, index=False)
        print(f"[PROCESSED] file_name: {file_name}, strategy: {file_name}, save_dir: {save_dir}")
