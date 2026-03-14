import argparse
import os
import random
import networkx as nx
import pandas as pd
from preflibtools.instances.sampling import generate_IC

CANDIDATES = [
    "Macron", "Le Pen", "Melenchon", "Zemmour", "Pecresse",
    "Jadot", "Lassalle", "Roussel", "Dupont-Aignan",
    "Hidalgo", "Poutou", "Arthaud"
]

SCENARIOS = {
    "D1": (0.6, 0.2, 0.2),
    "D2": (0.4, 0.3, 0.3),
    "D3": (0.3, 0.4, 0.3),
    "D4": (0.2, 0.6, 0.2),
    "D5": (0.2, 0.3, 0.5),
}

def get_prefs(n, rng):
    random.seed(rng.randint(0, 1000000))
    raw = generate_IC(n, len(CANDIDATES))
    final = []
    for order, count in raw.items():
        ranking = [CANDIDATES[int(item[0])] for item in order]
        for _ in range(count):
            final.append(ranking[:])
    rng.shuffle(final)
    return final[:n]

def setup_run(s_id, r_id, n, net_type, deg, p_stub, p_strat, p_mix, pref_mod, seed, out_dir):
    rng = random.Random(seed)
    tag = f"{r_id:03d}"
    folder_name = f"{s_id}_N{n}_run_{tag}"
    path = os.path.join(out_dir, folder_name)
    os.makedirs(path, exist_ok=True)

    prefs = get_prefs(n, rng)
    
    types = (["stubborn"] * int(n * p_stub) + 
             ["strategic"] * int(n * p_strat) + 
             ["mixed"] * (n - int(n * p_stub) - int(n * p_strat)))
    rng.shuffle(types)

    rows = []
    for i in range(n):
        l = 1.0 if types[i] == "stubborn" else 0.0 if types[i] == "strategic" else round(rng.uniform(0.4, 0.9), 3)
        row = {"voter_id": i, "agent_type": types[i], "loyalty": l}
        for j, c in enumerate(prefs[i], start=1): row[f"pref_{j}"] = c
        row["initial_vote"] = prefs[i][0]
        rows.append(row)

    if net_type == "erdos_renyi":
        G = nx.erdos_renyi_graph(n, deg/n, seed=seed)
    else:
        G = nx.barabasi_albert_graph(n, deg//2, seed=seed)

    pd.DataFrame(rows).to_csv(os.path.join(path, "voters.csv"), index=False)
    pd.DataFrame(list(G.edges()), columns=["source", "target"]).to_csv(os.path.join(path, "edges.csv"), index=False)
    
    return {"scenario_id": s_id, "repetition_id": r_id, "run_folder": folder_name, "num_agents": n, "network_type": net_type, "preference_model": pref_mod, "prop_stubborn": p_stub, "prop_strategic": p_strat, "prop_mixed": p_mix, "avg_degree": deg, "seed": seed}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", type=str, default="input")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = []
    s_val = 42

    for sid, (ps, pt, pm) in SCENARIOS.items():
        for r in range(1, args.reps + 1):
            res = setup_run(sid, r, args.n, "erdos_renyi", 10, ps, pt, pm, "IC", s_val, args.out)
            meta.append(res)
            s_val += 1

    pd.DataFrame(meta).to_csv(os.path.join(args.out, "run_metadata.csv"), index=False)
    print("Done. Generated", len(meta), "runs.")

if __name__ == "__main__":
    main()
