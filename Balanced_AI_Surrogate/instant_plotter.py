import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Paths
BASE_DIR = Path("C:/Users/moeez/Downloads/ATAI/projWeek/AI-Project/Balanced_AI_Surrogate")
CSV_FILE = BASE_DIR / "part3_outputs/csv/extrapolation_100000_agents.csv"
OUT_FILE = BASE_DIR / "part3_outputs/plots/extrapolation_100k_combined.png"

def create_combined_plot():
    if not CSV_FILE.exists():
        print(f"Error: Could not find {CSV_FILE}. Please make sure you have the extrapolation CSV.")
        return

    df = pd.read_csv(CSV_FILE)
    
    # Columns: day, predicted_num_changes_100000_agents, predicted_variance_scores_100000_agents, predicted_social_welfare_100000_agents
    metrics = df.iloc[:, 1:].values
    
    # Normalize
    mms = MinMaxScaler()
    norm_metrics = mms.fit_transform(metrics)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['day'], norm_metrics[:, 0], label="Opinion Changes", color="red", lw=2.5)
    plt.plot(df['day'], norm_metrics[:, 1], label="Variance (Consensus)", color="blue", lw=2.5)
    plt.plot(df['day'], norm_metrics[:, 2], label="Social Welfare", color="green", lw=2.5)
    
    plt.title("Relative Metric Evolution (100,000 Agents)")
    plt.xlabel("Day")
    plt.ylabel("Relative Scale (Normalized 0-1)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUT_FILE, dpi=200)
    print(f"Success! Plot saved to: {OUT_FILE}")

if __name__ == "__main__":
    create_combined_plot()
