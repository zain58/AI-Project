import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Config
DIST_LIST = ["D1", "D2", "D3", "D4", "D5"]
CONFIG_LIST = ["IC_ER", "IC_BA", "Urn_ER", "Urn_BA"]

LABELS = {
    "variance_scores": "Variance of Score Distribution",
    "social_welfare":  "Social Welfare (avg rank points)",
    "num_changes":     "Number of Opinion Changes"
}

COLORS = {
    "IC_ER": "#1f77b4", "IC_BA": "#ff7f0e", "Urn_ER": "#2ca02c", "Urn_BA": "#d62728"
}

D_COLORS = {
    "D1": "#9b59b6", "D2": "#3498db", "D3": "#2ecc71", "D4": "#e67e22", "D5": "#e74c3c"
}

def load_data(path):
    files = glob.glob(os.path.join(path, "simulation_results_*.csv"))
    if not files: return None
    all_dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(all_dfs, ignore_index=True)
    df["config"] = df["preference_model"] + "_" + df["network_type"].str.replace("erdos_renyi","ER").str.replace("barabasi_albert","BA")
    return df

def plot_single(df, metric, group_col, target_list, color_map, out_name, out_dir, title):
    plt.figure(figsize=(6, 4))
    for item in target_list:
        sub = df[df[group_col] == item]
        if sub.empty: continue
        avg = sub.groupby("day")[metric].mean()
        plt.plot(avg.index, avg.values, label=item, color=color_map.get(item), lw=2)
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel(LABELS[metric])
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_name))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="C:/Users/moeez/Downloads/ATAI/projWeek/AI-Project/COMPR_output")
    args = parser.parse_args()
    data = load_data(args.dir)
    if data is None: return
    p_dir = os.path.join(args.dir, "plots")
    os.makedirs(p_dir, exist_ok=True)
    
    # 1. Global Dynamics (Opinion Changes)
    plot_single(data, "num_changes", "config", CONFIG_LIST, COLORS, "plot1_configs.png", p_dir, "Opinion Changes by Configuration")
    
    # 2. Preference Comparison (Social Welfare)
    sub_pref = data[data["network_type"] == "erdos_renyi"]
    plot_single(sub_pref, "social_welfare", "preference_model", ["IC", "Urn"], {"IC": "#1f77b4", "Urn": "#2ca02c"}, "plot2_ic_vs_urn.png", p_dir, "Welfare: IC vs Urn")
    
    # 3. Network Comparison (Variance)
    sub_net = data[data["preference_model"] == "IC"]
    plot_single(sub_net, "variance_scores", "network_type", ["erdos_renyi", "barabasi_albert"], {"erdos_renyi": "#1f77b4", "barabasi_albert": "#ff7f0e"}, "plot3_er_vs_ba.png", p_dir, "Variance: ER vs BA")
    
    # 4. Agent Distributions (Changes)
    sub_dist = data[data["config"] == "IC_ER"]
    plot_single(sub_dist, "num_changes", "scenario_id", DIST_LIST, D_COLORS, "plot4_distributions_D1_D5.png", p_dir, "Changes: Agent Mix (D1-D5)")
    
    # 5. Interaction Effect
    plot_single(data, "num_changes", "config", ["IC_ER", "Urn_BA"], COLORS, "plot6_interaction_effect.png", p_dir, "Interaction: Baseline vs Urn+BA")
    
    print(f"Success. 5 Single Plots generated in {p_dir}")

if __name__ == "__main__":
    main()
