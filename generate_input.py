import argparse
import os
import random
from typing import List

import networkx as nx
import pandas as pd


CANDIDATES = [
    "Macron", "Le Pen", "Melenchon", "Zemmour", "Pecresse",
    "Jadot", "Lassalle", "Roussel", "Dupont-Aignan",
    "Hidalgo", "Poutou", "Arthaud"
]


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

    # Fix isolated nodes deterministically using the same RNG
    isolates = list(nx.isolates(G))
    for node in isolates:
        candidates = [x for x in range(num_agents) if x != node]
        target = rng.choice(candidates)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--num_agents", type=int, default=1000)
    parser.add_argument("--network_type", type=str, default="erdos_renyi")
    parser.add_argument("--avg_degree", type=int, default=10)
    parser.add_argument("--prop_stubborn", type=float, default=0.3)
    parser.add_argument("--prop_strategic", type=float, default=0.4)
    parser.add_argument("--prop_mixed", type=float, default=0.3)
    parser.add_argument("--preference_model", type=str, default="IC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="input")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)

    if args.preference_model != "IC":
        raise ValueError("This baseline currently implements IC only.")

    preferences = sample_ic_preferences(args.num_agents, rng)

    agent_types = assign_agent_types(
        args.num_agents,
        args.prop_stubborn,
        args.prop_strategic,
        args.prop_mixed,
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
    for i in range(args.num_agents):
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
        args.num_agents,
        args.network_type,
        args.avg_degree,
        args.seed,
        rng
    )

    edges_df = pd.DataFrame(list(G.edges()), columns=["source", "target"])

    run_sanity_checks(voters_df, edges_df, args.num_agents)

    voters_path = os.path.join(args.output_dir, f"voters_run_{args.run_id:03d}.csv")
    edges_path = os.path.join(args.output_dir, f"edges_run_{args.run_id:03d}.csv")
    meta_path = os.path.join(args.output_dir, "run_metadata.csv")

    voters_df.to_csv(voters_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    meta_row = {
        "run_id": args.run_id,
        "num_agents": args.num_agents,
        "network_type": args.network_type,
        "preference_model": args.preference_model,
        "prop_stubborn": args.prop_stubborn,
        "prop_strategic": args.prop_strategic,
        "prop_mixed": args.prop_mixed,
        "avg_degree": args.avg_degree,
        "seed": args.seed
    }

    if os.path.exists(meta_path):
        old_meta = pd.read_csv(meta_path)
        old_meta = old_meta[old_meta["run_id"] != args.run_id]
        new_meta = pd.concat([old_meta, pd.DataFrame([meta_row])], ignore_index=True)
    else:
        new_meta = pd.DataFrame([meta_row])

    new_meta.to_csv(meta_path, index=False)

    print(f"[OK] Wrote {voters_path}")
    print(f"[OK] Wrote {edges_path}")
    print(f"[OK] Wrote {meta_path}")
    print(f"[OK] Number of edges: {len(edges_df)}")


if __name__ == "__main__":
    main()