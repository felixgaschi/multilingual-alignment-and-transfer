import pandas as pd
import os

import csv
from io import StringIO

results_dir = "./scripts/2025_aacl/results"

for method in os.listdir(results_dir):
    method_dir = os.path.join(results_dir, method)

    for file_name in os.listdir(method_dir):
        if not (file_name.endswith(".csv") and "xnli" in file_name):
            continue

        file_path = os.path.join(method_dir, file_name)

        with open(file_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            continue
        # Process header
        header = lines[0].strip().split(",")
        if len(header) == 83 and "eval_en_accuracy" in header:
            print(file_path)
            # Remove index 52 first, then 17 to avoid index shifting
            header.pop(52)
            header.pop(17)
        lines[0] = ",".join(header) + "\n"

        # Process rows
        for i in range(1, len(lines)):
            if lines[i].startswith("31"):
                row = next(csv.reader(StringIO(lines[i])))
                if len(row) == 83:
                    row.pop(52)
                    row.pop(17)
                    output = StringIO()
                    csv.writer(output).writerow(row)
                    lines[i] = output.getvalue()

        # Write corrected CSV back to file (or change to a new file if you prefer)
        with open(file_path, "w") as f:
            f.writelines(lines)
