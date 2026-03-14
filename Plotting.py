"""
Part 2 — Plotting Script
Reads GAMA simulation output CSVs and generates comparison plots.

Usage:
    python plot_results.py --output_dir "C:/Users/zainn/downloads/AI-Project/output"
"""

import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# -------------------------------------------------------
# Config
# -------------------------------------------------------
DISTRIBUTIONS  = ["D1", "D2", "D3", "D4", "D5"]
CONFIGS        = ["IC_ER", "IC_BA", "Urn_ER", "Urn_BA"]
METRICS        = ["variance_scores", "social_welfare", "num_changes"]
METRIC_LABELS  = {
    "variance_scores": "Variance of Score Distribution",
    "social_welfare":  "Social Welfare (avg rank points)",
    "num_changes":     "Number of Opinion Changes"
}
METRIC_COLORS  = {
    "variance_scores": "#2196F3",
    "social_welfare":  "#4CAF50",
    "num_changes":     "#F44336"
}
CONFIG_COLORS  = {
    "IC_ER":  "#1f77b4",
    "IC_BA":  "#ff7f0e",
    "Urn_ER": "#2ca02c",
    "Urn_BA": "#d62728"
}
DIST_COLORS = {
    "D1": "#9b59b6",
    "D2": "#3498db",
    "D3": "#2ecc71",
    "D4": "#e67e22",
    "D5": "#e74c3c"
}
DIST_LABELS = {
    "D1": "D1 (60% Stubborn)",
    "D2": "D2 (40% Stubborn)",
    "D3": "D3 (30% Stubborn / Balanced)",
    "D4": "D4 (60% Strategic)",
    "D5": "D5 (50% Mixed)"
}


# -------------------------------------------------------
def load_all_results(output_dir: str) -> pd.DataFrame:
    """Load all simulation result CSVs into one DataFrame."""
    pattern = os.path.join(output_dir, "simulation_results_*.csv")
    files   = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No simulation result files found in: {output_dir}\n"
            f"Expected files matching: simulation_results_*.csv"
        )

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[OK] Loaded {len(files)} files → {len(full_df)} rows total")
    print(f"     Configs found:       {sorted(full_df['preference_model'].unique())} x {sorted(full_df['network_type'].unique())}")
    print(f"     Scenarios found:     {sorted(full_df['scenario_id'].unique())}")
    print(f"     Days range:          {full_df['day'].min()} - {full_df['day'].max()}")
    print(f"     Repetitions found:   {sorted(full_df['repetition_id'].unique())}")

    # Add combined config column
    full_df["config"] = full_df["preference_model"] + "_" + full_df["network_type"].str.replace("erdos_renyi","ER").str.replace("barabasi_albert","BA")

    return full_df


# -------------------------------------------------------
def compute_mean_std(df: pd.DataFrame, group_cols: list, metric: str):
    """Compute mean and std of metric grouped by group_cols."""
    grouped = df.groupby(group_cols)[metric].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)
    return grouped


# -------------------------------------------------------
# PLOT 1: Evolution over 60 days — one plot per metric
# Shows all 4 configs, averaged over all scenarios and reps
# -------------------------------------------------------
def plot_evolution_by_config(df: pd.DataFrame, plots_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Evolution Over 60 Days — All Configurations\n(averaged over all scenarios and repetitions)",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, METRICS):
        for config in CONFIGS:
            if config not in df["config"].unique():
                continue
            sub = df[df["config"] == config]
            agg = compute_mean_std(sub, ["day"], metric)

            ax.plot(agg["day"], agg["mean"],
                    label=config,
                    color=CONFIG_COLORS.get(config, "gray"),
                    linewidth=2)
            ax.fill_between(agg["day"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            alpha=0.15,
                            color=CONFIG_COLORS.get(config, "gray"))

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("Day")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "plot1_evolution_by_config.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


# -------------------------------------------------------
# PLOT 2: IC vs Urn comparison (same network ER)
# Shows how preference model affects each metric over time
# -------------------------------------------------------
def plot_ic_vs_urn(df: pd.DataFrame, plots_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("IC vs Urn Model Comparison (Erdos-Renyi Network)\n(averaged over all scenarios and repetitions)",
                 fontsize=13, fontweight="bold")

    compare = {"IC_ER": "IC (Random)", "Urn_ER": "Urn (Polarized)"}
    colors  = {"IC_ER": "#1f77b4", "Urn_ER": "#2ca02c"}

    for ax, metric in zip(axes, METRICS):
        for config, label in compare.items():
            if config not in df["config"].unique():
                continue
            sub = df[df["config"] == config]
            agg = compute_mean_std(sub, ["day"], metric)

            ax.plot(agg["day"], agg["mean"],
                    label=label,
                    color=colors[config],
                    linewidth=2.5)
            ax.fill_between(agg["day"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            alpha=0.2,
                            color=colors[config])

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("Day")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "plot2_ic_vs_urn.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


# -------------------------------------------------------
# PLOT 3: ER vs BA comparison (same preference IC)
# Shows how network topology affects each metric over time
# -------------------------------------------------------
def plot_er_vs_ba(df: pd.DataFrame, plots_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Erdos-Renyi vs Barabasi-Albert Network Comparison (IC Model)\n(averaged over all scenarios and repetitions)",
                 fontsize=13, fontweight="bold")

    compare = {"IC_ER": "Erdos-Renyi (Random)", "IC_BA": "Barabasi-Albert (Hubs)"}
    colors  = {"IC_ER": "#1f77b4", "IC_BA": "#ff7f0e"}

    for ax, metric in zip(axes, METRICS):
        for config, label in compare.items():
            if config not in df["config"].unique():
                continue
            sub = df[df["config"] == config]
            agg = compute_mean_std(sub, ["day"], metric)

            ax.plot(agg["day"], agg["mean"],
                    label=label,
                    color=colors[config],
                    linewidth=2.5)
            ax.fill_between(agg["day"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            alpha=0.2,
                            color=colors[config])

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("Day")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "plot3_er_vs_ba.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


# -------------------------------------------------------
# PLOT 4: D1-D5 distribution comparison
# Shows how agent mix affects each metric (one config = IC_ER)
# -------------------------------------------------------
def plot_distributions(df: pd.DataFrame, plots_dir: str):
    sub = df[df["config"] == "IC_ER"] if "IC_ER" in df["config"].unique() else df

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Agent Distribution Comparison D1-D5 (IC + Erdos-Renyi)\n(averaged over all repetitions)",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, METRICS):
        for dist in DISTRIBUTIONS:
            dsub = sub[sub["scenario_id"] == dist]
            if len(dsub) == 0:
                continue
            agg = compute_mean_std(dsub, ["day"], metric)

            ax.plot(agg["day"], agg["mean"],
                    label=DIST_LABELS.get(dist, dist),
                    color=DIST_COLORS.get(dist, "gray"),
                    linewidth=2)

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("Day")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "plot4_distributions_D1_D5.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


# -------------------------------------------------------
# PLOT 5: Interaction effect — all 4 configs final day boxplot
# -------------------------------------------------------
def plot_final_day_boxplot(df: pd.DataFrame, plots_dir: str):
    final = df[df["day"] == df["day"].max()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Final Day (Day {df['day'].max()}) — Distribution Across All Runs",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, METRICS):
        data_to_plot = []
        labels       = []
        colors       = []

        for config in CONFIGS:
            sub = final[final["config"] == config][metric].dropna()
            if len(sub) == 0:
                continue
            data_to_plot.append(sub.values)
            labels.append(config)
            colors.append(CONFIG_COLORS.get(config, "gray"))

        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xlabel("Configuration")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(plots_dir, "plot5_final_day_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


# -------------------------------------------------------
# PLOT 6: Interaction effect — Urn+BA vs IC+ER (best vs baseline)
# -------------------------------------------------------
def plot_interaction_effect(df: pd.DataFrame, plots_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Interaction Effect: Hub + Polarization (Urn_BA) vs Baseline (IC_ER)",
                 fontsize=13, fontweight="bold")

    compare = {
        "IC_ER":  "Baseline (IC + Random)",
        "IC_BA":  "Hub Effect (IC + BA)",
        "Urn_ER": "Polarization Effect (Urn + ER)",
        "Urn_BA": "Hub + Polarization (Urn + BA)"
    }

    for ax, metric in zip(axes, METRICS):
        for config, label in compare.items():
            if config not in df["config"].unique():
                continue
            sub = df[df["config"] == config]
            agg = compute_mean_std(sub, ["day"], metric)

            ax.plot(agg["day"], agg["mean"],
                    label=label,
                    color=CONFIG_COLORS.get(config, "gray"),
                    linewidth=2,
                    linestyle="--" if "Urn" in config else "-")

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("Day")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "plot6_interaction_effect.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot GAMA simulation results")
    parser.add_argument("--output_dir", type=str,
                        default="C:/Users/zainn/downloads/AI-Project/output",
                        help="Directory containing simulation_results_*.csv files")
    args = parser.parse_args()

    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[OK] Plots will be saved to: {plots_dir}")

    # Load all CSVs
    df = load_all_results(args.output_dir)

    # Generate all plots
    print("\n[INFO] Generating plots...")
    plot_evolution_by_config(df, plots_dir)
    plot_ic_vs_urn(df, plots_dir)
    plot_er_vs_ba(df, plots_dir)
    plot_distributions(df, plots_dir)
    plot_final_day_boxplot(df, plots_dir)
    plot_interaction_effect(df, plots_dir)

    print(f"\n[DONE] All 6 plots saved to: {plots_dir}")
    print("\nPlots summary:")
    print("  plot1_evolution_by_config.png  — All 4 configs over 60 days")
    print("  plot2_ic_vs_urn.png            — IC vs Urn preference model")
    print("  plot3_er_vs_ba.png             — ER vs BA network topology")
    print("  plot4_distributions_D1_D5.png  — D1-D5 agent distributions")
    print("  plot5_final_day_boxplot.png    — Final day boxplot comparison")
    print("  plot6_interaction_effect.png   — Hub + Polarization interaction")


if __name__ == "__main__":
    main()