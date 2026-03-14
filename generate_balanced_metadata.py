import pandas as pd
import os
import glob

def generate_metadata(data_dir):
    files = glob.glob(os.path.join(data_dir, "simulation_results_*.csv"))
    metadata = []

    for f in files:
        filename = os.path.basename(f)
        # simulation_results_IC_ER_D1_N1000_run_001.csv
        parts = filename.replace("simulation_results_", "").replace(".csv", "").split("_")
        
        pref_model = parts[0]
        net_type = "erdos_renyi" if parts[1] == "ER" else "barabasi_albert"
        scenario = parts[2]
        num_agents = int(parts[3].replace("N", ""))
        rep = int(parts[5])
        
        # Folder name as ID (used by the LSTM script to match files)
        folder_name = f"{scenario}_N{num_agents}_run_{rep:03d}"
        
        # Load the file to get the actual proportions from the first row
        df = pd.read_csv(f)
        first_row = df.iloc[0]
        
        metadata.append({
            "scenario_id": scenario,
            "repetition_id": rep,
            "run_folder": folder_name,
            "num_agents": num_agents,
            "network_type": net_type,
            "preference_model": pref_model,
            "prop_stubborn": first_row['prop_stubborn'],
            "prop_strategic": first_row['prop_strategic'],
            "prop_mixed": first_row['prop_mixed'],
            "avg_degree": 10,
            "seed": 42 + rep # Dummy seed
        })

    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv(os.path.join(data_dir, "run_metadata.csv"), index=False)
    print(f"Created metadata for {len(df_meta)} runs.")

if __name__ == "__main__":
    generate_metadata('Balanced_AI_Surrogate/gama_runs')
