import pandas as pd
import os
import glob
import numpy as np

def final_master_analysis(directory):
    files = glob.glob(os.path.join(directory, "*.csv"))
    data_list = []

    for f in files:
        df = pd.read_csv(f)
        if df.empty: continue
        
        row = df.iloc[0]
        # Core Parameters
        net = row['network_type']
        pref = row['preference_model']
        n = int(row['num_agents'])
        scenario = row['scenario_id']
        
        # Results
        final_variance = df['variance_scores'].iloc[-1]
        final_welfare = df['social_welfare'].iloc[-1]
        total_changes = df['num_changes'].sum()
        normalized_changes = total_changes / n
        
        data_list.append({
            'Network': net,
            'Preference': pref,
            'N': n,
            'Scenario': scenario,
            'Variance': final_variance,
            'Welfare': final_welfare,
            'Changes_Per_Voter': normalized_changes,
            'Stability_Day': df[df['num_changes'] < (n * 0.01)]['day'].min() # Day consensus reached
        })

    full_df = pd.DataFrame(data_list)
    
    # 1. Main Scaling Effect
    scaling = full_df.groupby('N')[['Variance', 'Changes_Per_Voter']].mean()
    
    # 2. Main Model Effect (Aggregated at N=1000 for fairness)
    models_1k = full_df[full_df['N'] == 1000].groupby(['Network', 'Preference'])[['Variance', 'Changes_Per_Voter']].mean()
    
    # 3. Social Welfare Table
    welfare = full_df.groupby(['Network', 'Preference'])['Welfare'].mean()

    # 4. Scenario Impact
    scenarios = full_df.groupby('Scenario')[['Variance', 'Changes_Per_Voter']].mean()

    # Output to Console for the CLI to see
    print("--- SCALING ---")
    print(scaling)
    print("\n--- MODEL INTERACTIONS (N=1000) ---")
    print(models_1k)
    print("\n--- WELFARE ---")
    print(welfare)
    print("\n--- SCENARIO ---")
    print(scenarios)

if __name__ == "__main__":
    final_master_analysis('COMPR_output')
