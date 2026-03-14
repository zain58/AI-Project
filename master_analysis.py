import pandas as pd
import os
import glob

def analyze_scaling(directory):
    files = glob.glob(os.path.join(directory, "*.csv"))
    data_list = []

    for f in files:
        df = pd.read_csv(f)
        if df.empty: continue
        
        row = df.iloc[0]
        # Calculate scaling metrics
        # We normalize changes by population size to see if people switch MORE in large groups
        total_changes = df['num_changes'].sum()
        norm_changes = total_changes / row['num_agents']
        
        data_list.append({
            'Network': row['network_type'],
            'Preference': row['preference_model'],
            'N': row['num_agents'],
            'Variance': df['variance_scores'].iloc[-1],
            'Changes_Per_Voter': norm_changes
        })

    full_df = pd.DataFrame(data_list)
    
    # Analyze by Size
    scaling_effect = full_df.groupby('N')[['Variance', 'Changes_Per_Voter']].mean()
    
    # Interaction: How Size affects different Networks
    net_size_interaction = full_df.groupby(['Network', 'N'])[['Variance', 'Changes_Per_Voter']].mean()

    print("=== SCALING EFFECT: IMPACT OF POPULATION SIZE ===")
    print(scaling_effect)
    print("\n=== INTERACTION: NETWORK X SIZE ===")
    print(net_size_interaction)

if __name__ == "__main__":
    analyze_scaling('COMPR_output')
