# Strategy for Bonus Points: Comparative Simulation Analysis

This document outlines the scientific workflow designed to earn extra points in Part 2 (Simulation) and Part 4 (Report) by comparing different mathematical models for voter preferences and social network topologies.

## 1. The Core Objective
The goal is to isolate the impact of **Opinion Distribution** and **Network Connectivity** on the dynamics of strategic voting. We achieve this through a **Nested Grid Search** while keeping the population size constant ($N=3000$) to ensure results are comparable.

---

## 2. Experimental Design (The Nested Matrix)
For each of the 5 agent configurations (D1 to D5), we generate four distinct simulation "worlds."

| Combination | Preference Model | Network Model | Research Goal |
| :--- | :--- | :--- | :--- |
| **A (Baseline)** | **IC** (Impartial Culture) | **Erdos-Renyi** (Random) | Establish a random baseline. |
| **B (Network Var)** | **IC** (Impartial Culture) | **Barabasi-Albert** (Scale-Free) | Isolate the effect of "Influencer" hubs. |
| **C (Pref Var)** | **Urn** (Polya-Eggenberger) | **Erdos-Renyi** (Random) | Isolate the effect of correlated/polarized opinions. |
| **D (Interaction)**| **Urn** (Polya-Eggenberger) | **Barabasi-Albert** (Scale-Free) | **Bonus Goal:** Observe how hubs amplify polarized opinions. |

---

## 3. Description of Comparison Models

### A. Preference Generation (Library: `preflibtools`)
*   **Impartial Culture (IC):** The default model. Every candidate has an equal probability of being ranked in any position. This represents a "purely random" society.
*   **Polya-Eggenberger Urn Model:** A sophisticated model where voters are more likely to pick rankings that have already been picked by others. This creates "social clumps" or polarization, making the simulation much more realistic.

### B. Network Generation (Library: `networkx`)
*   **Erdos-Renyi (Random Graph):** The default model. Every pair of voters has an equal probability of being connected. This represents a "flat" society without clear social structures.
*   **Barabasi-Albert (Scale-Free):** A "Rich-Get-Richer" model. It creates a few "hub" nodes with many connections and many nodes with few connections. This accurately models real-world social media where a few influencers can sway large numbers of followers.

---

## 4. Scientific Value (The "A+" Insights)
By nesting these models, we can answer Master's-level research questions in our Part 4 report:
1.  **The Hub Effect:** "Do Barabasi-Albert hubs make strategic surges happen faster than in random networks?"
2.  **The Polarization Effect:** "Does the Urn model lead to higher or lower Social Welfare compared to the IC model?"
3.  **The Interaction Effect:** "Does a strategic influencer (Hub) have significantly more power when the population already shares similar opinions (Urn)?"

---

## 5. Benefits for Part 3 (LSTM Surrogate Model)
This structured data is ideal for Deep Learning. Instead of giving the LSTM noisy, random data, we provide it with clear **categorical features**:
*   `Pref_Type`: 0 (IC) or 1 (Urn)
*   `Net_Type`: 0 (ER) or 1 (BA)

The Neural Network will learn the specific "weights" of these variables, allowing it to accurately predict future simulation outcomes even for scenarios it hasn't seen before.
