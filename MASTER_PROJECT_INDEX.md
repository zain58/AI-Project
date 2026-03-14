# MASTER PROJECT INDEX: The Road to A+

This document serves as the central repository for all technical justifications, scientific findings, and architectural decisions made during the ATAI Surrogate Model project.

## PART 1: GAME THEORETIC FOUNDATIONS
- **Finding:** Pure strategy is high-risk in small-world networks.
- **Maturity Point:** Implemented "Bounded Rationality" via a Two-Factor Tipping Point (Abandonment < 10%, Efficacy < 5% margin).
- **Reference:** `AGENT_MODELS_ANALYSIS.md`

## PART 2: SIMULATION & NETWORK DYNAMICS
- **Baseline:** Erdos-Renyi + IC (Established the "Simple" world).
- **Complexity:** Balanced 240-run dataset (IC/Urn x ER/BA x N1k/3k/5k).
- **Key Discovery 1:** The **Quadratic Scaling Law** (Variance increases as $N^2$).
- **Key Discovery 2:** The **Network Overrule Effect** (BA hubs stabilize the system and negate preference clumping).
- **Reference:** `FINAL_COMPARISON_RESULTS.md` & `COMPARISON_RESULTS_DEEP_DIVE.md`

## PART 3: SURROGATE ARCHITECTURE
- **Strategy:** 3-Model Many-to-One LSTM (Specialized for scale differences).
- **Techniques:** Recursive Rollout, Huber Loss, Log-Scaling, One-Hot Encoding.
- **Extrapolation:** Size-aware training allows calculated projection to 100,000 agents.
- **Reference:** `Balanced_AI_Surrogate/part3_surrogate_lstm.py`

## PART 4: FINAL PAPER REQUIREMENTS (TODO)
- [ ] Incorporate Kaggle `metrics.json` (MSE/R2).
- [ ] Analyze Extrapolation Curve (GAMA vs Surrogate Tradeoff).
- [ ] Compare Compute Time (Surrogate is 99% faster).
- [ ] Final 5-page synthesis in ACM format.
