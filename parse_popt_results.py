from glob import glob as glob
import os
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

results_root="/om2/user/zaho/btbench_popt/btbench/outputs/btbench_popt_lite/"
split_types = ["SS_SM", "SS_DM"]
split_type = split_types[1]

file_paths = glob(os.path.join(results_root, split_type, "popt_*_frozen_True", "results.json"))

all_records = []
for path in file_paths:
    result_name = path.split("/")[-2]
    sub_name = "_".join(result_name.split("_")[1:3])
    sub_id = int(sub_name[len("sub_"):])
    trial_name = result_name.split("_")[3]
    trial_id = int(trial_name[len("trial"):])
    task_name = "_".join(result_name.split("_")[4:])
   
    with open(path, "r") as f:
        results = json.load(f)
        
        for k in range(len(results)):
            test_roc_auc = results[k]["test_roc_auc"]
        
            all_records.append({
                "subject_name": sub_name,
                "trial_name": trial_name,
                "subject_id": sub_id,
                "trial_id": trial_id,
                "fold": k,
                "task_name": task_name,
                "test_roc_auc": test_roc_auc

            })
            print(all_records[-1])
results_df = pd.DataFrame.from_records(all_records)
# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Save the results DataFrame to a CSV file
output_filename = f"/om2/user/zaho/btbench/eval_results_popt/population_frozen_{split_type}_results.csv"
results_df.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")