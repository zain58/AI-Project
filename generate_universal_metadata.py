import pandas as pd
import os
import glob

def generate_metadata(data_dir):
    files = glob.glob(os.path.join(data_dir, "simulation_results_*.csv"))
    metadata = []

    for f in files:
        filename = os.path.basename(f)
        # Load the file to get the TRUTH from the first row
        # This is safer than parsing filenames which change styles
        try:
            df = pd.read_csv(f)
            if df.empty: continue
            first_row = df.iloc[0]
            
            # The LSTM script matches file -> metadata using RID
            # Format used by LSTM script: {scenario}_N{size}_run_{rep:03d}
            rid = f"{first_row['scenario_id']}_N{int(first_row['num_agents'])}_run_{int(first_row['repetition_id']):03d}"
            
            # However, because we have duplicate Scenario/N/Rep across IC/Urn/ER/BA,
            # we must make the run_folder unique in the metadata dictionary.
            # I will use the actual filename stem as the unique key.
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
    df_meta.to_csv(os.path.join(data_dir, "run_metadata.csv"), index=False)
    print(f"Created metadata for {len(df_meta)} runs.")

if __name__ == "__main__":
    generate_metadata('Balanced_AI_Surrogate/gama_runs')
