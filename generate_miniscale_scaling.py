import os
import random
from typing import List, Dict, Tuple
import networkx as nx
import pandas as pd
from preflibtools.instances.sampling import generate_IC, generate_urn

CANDIDATES = [
    "Macron", "Le Pen", "Melenchon", "Zemmour", "Pecresse",
    "Jadot", "Lassalle", "Roussel", "Dupont-Aignan",
    "Hidalgo", "Poutou", "Arthaud"
]

DISTRIBUTIONS: Dict[str, Tuple[float, float, float]] = {
    "D1": (0.6, 0.2, 0.2),
    "D2": (0.4, 0.3, 0.3),
    "D3": (0.3, 0.4, 0.3),
    "D4": (0.2, 0.6, 0.2),
    "D5": (0.2, 0.3, 0.5),
}

def sample_preferences(num_agents: int, pref_model: str, rng: random.Random) -> List[List[str]]:
    random.seed(rng.randint(0, 2**31 - 1))
    if pref_model == "IC":
        instance = generate_IC(num_agents, len(CANDIDATES))
    else: # Urn
        instance = generate_urn(num_agents, len(CANDIDATES), 0.1)

    prefs: List[List[str]] = []
    for order, multiplicity in instance.items():
        # Flatten nested tuples: ((8,), (4,), ...) -> [8, 4, ...]
        ranking = [CANDIDATES[int(item[0])] for item in order]
        for _ in range(multiplicity):
            prefs.append(ranking[:])
    rng.shuffle(prefs)
    return prefs[:num_agents]

def generate_scaling_run(scenario_id, num_agents, net_type, pref_model, output_dir):
    rng = random.Random(42)
    run_tag = "001"
    
    config_name = f"{pref_model}_{'ER' if net_type == 'erdos_renyi' else 'BA'}"
    folder_name = f"{scenario_id}_N{num_agents}_run_{run_tag}"
    full_path = os.path.join(output_dir, config_name, folder_name)
    os.makedirs(full_path, exist_ok=True)

    props = DISTRIBUTIONS[scenario_id]
    prefs = sample_preferences(num_agents, pref_model, rng)
    
    agent_types = (["stubborn"] * int(num_agents * props[0]) + 
                   ["strategic"] * int(num_agents * props[1]) + 
                   ["mixed"] * (num_agents - int(num_agents * props[0]) - int(num_agents * props[1])))
    rng.shuffle(agent_types)

    voter_rows = []
    for i in range(num_agents):
        row = {"voter_id": i, "agent_type": agent_types[i], "loyalty": 1.0 if agent_types[i] == "stubborn" else 0.0 if agent_types[i] == "strategic" else 0.6}
        for j, cand in enumerate(prefs[i], start=1): row[f"pref_{j}"] = cand
        row["initial_vote"] = prefs[i][0]
        voter_rows.append(row)

    if net_type == "erdos_renyi":
        G = nx.erdos_renyi_graph(num_agents, 10/num_agents)
    else:
        G = nx.barabasi_albert_graph(num_agents, 5)

    pd.DataFrame(voter_rows).to_csv(os.path.join(full_path, "voters.csv"), index=False)
    pd.DataFrame(list(G.edges()), columns=["source", "target"]).to_csv(os.path.join(full_path, "edges.csv"), index=False)
    print(f"[OK] {config_name} / {folder_name}")

def main():
    base_dir = "input_scaling"
    for config in [("erdos_renyi", "IC"), ("barabasi_albert", "IC"), ("erdos_renyi", "Urn"), ("barabasi_albert", "Urn")]:
        for scenario in DISTRIBUTIONS.keys():
            for size in [3000, 5000]:
                generate_scaling_run(scenario, size, config[0], config[1], base_dir)

if __name__ == "__main__":
    main()
