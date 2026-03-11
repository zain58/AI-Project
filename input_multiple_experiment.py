import argparse
import os
import random
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd


CANDIDATES = [
    "Macron", "Le Pen", "Melenchon", "Zemmour", "Pecresse",
    "Jadot", "Lassalle", "Roussel", "Dupont-Aignan",
    "Hidalgo", "Poutou", "Arthaud"
]

# Fixed distributions for Part 2
DISTRIBUTIONS: Dict[str, Tuple[float, float, float]] = {
    "D1": (0.6, 0.2, 0.2),  # stubborn-heavy
    "D2": (0.4, 0.3, 0.3),  # moderately stubborn
    "D3": (0.3, 0.4, 0.3),  # balanced / baseline
    "D4": (0.2, 0.6, 0.2),  # strategic-heavy
    "D5": (0.2, 0.3, 0.5),  # mixed-heavy
}


def sample_ic_preferences(num_agents: int, rng: random.Random) -> List[List[str]]:
    prefs = []
    for _ in range(num_agents):
        ranking = CANDIDATES[:]
        rng.shuffle(ranking)
        prefs.append(ranking)
    return prefs


def assign_agent_types(
    num_agents: int,
    prop_stubborn: float,
    prop_strategic: float,
    prop_mixed: float,
    rng: random.Random
) -> List[str]:
    total = prop_stubborn + prop_strategic + prop_mixed
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Agent proportions must sum to 1.0")

    if prop_stubborn < 0 or prop_strategic < 0 or prop_mixed < 0:
        raise ValueError("Agent proportions must be non-negative")

    n_stubborn = int(num_agents * prop_stubborn)
    n_strategic = int(num_agents * prop_strategic)
    n_mixed = num_agents - n_stubborn - n_strategic

    agent_types = (
        ["stubborn"] * n_stubborn +
        ["strategic"] * n_strategic +
        ["mixed"] * n_mixed
    )
    rng.shuffle(agent_types)
    return agent_types


def build_network(
    num_agents: int,
    network_type: str,
    avg_degree: int,
    seed: int,
    rng: random.Random
) -> nx.Graph:
    if network_type == "erdos_renyi":
        p = min(1.0, avg_degree / max(1, num_agents - 1))
        G = nx.erdos_renyi_graph(num_agents, p, seed=seed)
    elif network_type == "barabasi_albert":
        m = max(1, avg_degree // 2)
        G = nx.barabasi_albert_graph(num_agents, m, seed=seed)
    else:
        raise ValueError("network_type must be 'erdos_renyi' or 'barabasi_albert'")

    isolates = list(nx.isolates(G))
    for node in isolates:
        choices = [x for x in range(num_agents) if x != node]
        target = rng.choice(choices)
        G.add_edge(node, target)

    return G


def run_sanity_checks(voters_df: pd.DataFrame, edges_df: pd.DataFrame, num_agents: int) -> None:
    required_agent_types = {"stubborn", "strategic", "mixed"}
    found_agent_types = set(voters_df["agent_type"].unique())

    if not found_agent_types.issubset(required_agent_types):
        raise ValueError(f"Unexpected agent types found: {found_agent_types}")

    if len(voters_df) != num_agents:
        raise ValueError(f"Expected {num_agents} voters, got {len(voters_df)}")

    if not (voters_df["initial_vote"] == voters_df["pref_1"]).all():
        raise ValueError("Some rows have initial_vote different from pref_1")

    if (edges_df["source"] == edges_df["target"]).any():
        raise ValueError("Self-loops detected in edges file")

    if voters_df["voter_id"].duplicated().any():
        raise ValueError("Duplicate voter_id detected")

    pref_cols = [f"pref_{i}" for i in range(1, 13)]
    for idx, row in voters_df[pref_cols].iterrows():
        if len(set(row.tolist())) != 12:
            raise ValueError(f"Duplicate candidate in preferences for voter row {idx}")


def generate_one_run(
    scenario_id: str,
    repetition_id: int,
    num_agents: int,
    network_type: str,
    avg_degree: int,
    prop_stubborn: float,
    prop_strategic: float,
    prop_mixed: float,
    preference_model: str,
    seed: int,
    base_output_dir: str
) -> dict:
    rng = random.Random(seed)

    if preference_model != "IC":
        raise ValueError("This baseline currently implements IC only.")

    run_tag = f"{repetition_id:03d}"
    run_folder_name = f"{scenario_id}_N{num_agents}_run_{run_tag}"
    run_folder = os.path.join(base_output_dir, run_folder_name)
    os.makedirs(run_folder, exist_ok=True)

    preferences = sample_ic_preferences(num_agents, rng)

    agent_types = assign_agent_types(
        num_agents,
        prop_stubborn,
        prop_strategic,
        prop_mixed,
        rng
    )

    loyalties = []
    for t in agent_types:
        if t == "stubborn":
            loyalties.append(1.0)
        elif t == "strategic":
            loyalties.append(0.0)
        else:
            loyalties.append(round(rng.uniform(0.4, 0.9), 3))

    voter_rows = []
    for i in range(num_agents):
        row = {
            "voter_id": i,
            "agent_type": agent_types[i],
            "loyalty": loyalties[i]
        }
        for j, cand in enumerate(preferences[i], start=1):
            row[f"pref_{j}"] = cand
        row["initial_vote"] = preferences[i][0]
        voter_rows.append(row)

    voters_df = pd.DataFrame(voter_rows)

    G = build_network(
        num_agents=num_agents,
        network_type=network_type,
        avg_degree=avg_degree,
        seed=seed,
        rng=rng
    )

    edges_df = pd.DataFrame(list(G.edges()), columns=["source", "target"])

    run_sanity_checks(voters_df, edges_df, num_agents)

    voters_path = os.path.join(run_folder, "voters.csv")
    edges_path = os.path.join(run_folder, "edges.csv")

    voters_df.to_csv(voters_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    print(f"[OK] {run_folder_name}")
    print(f"     voters: {voters_path}")
    print(f"     edges : {edges_path}")
    print(f"     edges count: {len(edges_df)}")

    return {
        "scenario_id": scenario_id,
        "repetition_id": repetition_id,
        "run_folder": run_folder_name,
        "num_agents": num_agents,
        "network_type": network_type,
        "preference_model": preference_model,
        "prop_stubborn": prop_stubborn,
        "prop_strategic": prop_strategic,
        "prop_mixed": prop_mixed,
        "avg_degree": avg_degree,
        "seed": seed
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 3000, 5000])
    parser.add_argument("--runs_per_setting", type=int, default=10)
    parser.add_argument("--network_type", type=str, default="erdos_renyi")
    parser.add_argument("--avg_degree", type=int, default=10)
    parser.add_argument("--preference_model", type=str, default="IC")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="input")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata_rows = []
    seed_counter = 0

    for scenario_id, (prop_stubborn, prop_strategic, prop_mixed) in DISTRIBUTIONS.items():
        for num_agents in args.sizes:
            for repetition_id in range(1, args.runs_per_setting + 1):
                seed = args.base_seed + seed_counter
                seed_counter += 1

                meta = generate_one_run(
                    scenario_id=scenario_id,
                    repetition_id=repetition_id,
                    num_agents=num_agents,
                    network_type=args.network_type,
                    avg_degree=args.avg_degree,
                    prop_stubborn=prop_stubborn,
                    prop_strategic=prop_strategic,
                    prop_mixed=prop_mixed,
                    preference_model=args.preference_model,
                    seed=seed,
                    base_output_dir=args.output_dir
                )
                metadata_rows.append(meta)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = os.path.join(args.output_dir, "run_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\n[OK] Wrote metadata: {metadata_path}")
    print(f"[OK] Total runs generated: {len(metadata_df)}")


if __name__ == "__main__":
    main()