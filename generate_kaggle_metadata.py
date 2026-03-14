import pandas as pd
import os
import glob

def generate_balanced_metadata(input_dir, output_meta_path):
    files = glob.glob(os.path.join(input_dir, "simulation_results_*.csv"))
    metadata = []

    for f in files:
        filename = os.path.basename(f)
        # We only want the 200 comparison files
        # simulation_results_IC_BA_D1_N1000_run_001.csv
        parts = filename.replace("simulation_results_", "").replace(".csv", "").split("_")
        
        # Skip baseline files if they accidentally got mixed in
        if len(parts) < 6: continue 

        try:
            df = pd.read_csv(f)
            if df.empty: continue
            first_row = df.iloc[0]
            
            unique_key = filename.replace("simulation_results_", "").replace(".csv", "")

            metadata.append({
                "scenario_id": first_row['scenario_id'],
                "repetition_id": int(first_row['repetition_id']),
                "run_folder": unique_key, 
                "num_agents": int(first_row['num_agents']),
                "network_type": first_row['network_type'],
                "preference_model": first_row['preference_model'],
                "prop_stubborn": float(first_row['prop_stubborn']),
                "prop_strategic": float(first_row['prop_strategic']),
                "prop_mixed": float(first_row['prop_mixed']),
                "avg_degree": 10,
                "seed": 42
            })
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv(output_meta_path, index=False)
    print(f"SUCCESS: Created balanced metadata for {len(df_meta)} comparison runs.")

if __name__ == "__main__":
    generate_balanced_metadata('COMPR_output', 'Balanced_AI_Surrogate/gama_runs/run_metadata.csv')
