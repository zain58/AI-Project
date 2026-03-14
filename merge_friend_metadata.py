import pandas as pd
import os

def merge_metadata():
    base_dir = 'Balanced_AI_Surrogate/compare'
    configs = ['IC_BA', 'IC_ER', 'Urn_BA', 'Urn_ER']
    all_meta = []

    for config in configs:
        meta_path = os.path.join(base_dir, config, 'run_metadata.csv')
        if os.path.exists(meta_path):
            df = pd.read_csv(meta_path)
            # Critical: Modify run_folder to match the actual filename in COMPR_output
            # Filename pattern: simulation_results_IC_BA_D1_N1000_run_001.csv
            # Current run_folder in df: D1_N1000_run_001
            df['run_folder'] = config + "_" + df['run_folder']
            all_meta.append(df)
            print(f"Loaded {len(df)} rows from {config}")

    if all_meta:
        master_meta = pd.concat(all_meta, ignore_index=True)
        # Save to the AI training folder
        output_path = 'Balanced_AI_Surrogate/gama_runs/run_metadata.csv'
        master_meta.to_csv(output_path, index=False)
        print(f"\nSUCCESS: Merged {len(master_meta)} rows into {output_path}")
    else:
        print("Error: No metadata files found to merge.")

if __name__ == "__main__":
    merge_metadata()
