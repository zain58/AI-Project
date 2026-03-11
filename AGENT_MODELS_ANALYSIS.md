# GAMA Model Refactoring Analysis - March 11, 2026

## 1. Social Welfare Calculation Refinement
The social welfare metric was redefined to strictly adhere to the assignment requirement: *"each voter gets X points if the candidate that she ranks in X-th position goes to the second round."*

### Key Changes:
- **Rank-based Point Mapping:** Updated the `welfare_score_of` function to return the actual rank (1st = 1 pt, 2nd = 2 pts, etc.).
- **Additive Scoring:** Changed the global calculation to sum the points for **both** candidates in the Top 2. 
- **Dynamic Response:** The calculation now triggers every step based on the *current* global poll, allowing the metric to fluctuate as voters shift their strategies.
- **Interpretation:** Lower values now indicate higher social satisfaction (better average rank for frontrunners).

## 2. Agent Decision Logic (Mixed Agents)
The "Mixed" agent type was overhauled to implement a more sophisticated **Bounded Rationality** model.

### Logic Improvements:
- **Loyalty Priority:** Added a check to ensure agents always stay loyal if their favorite candidate is already in the Top 2.
- **Tipping Point Rule:** 
    1. **Abandonment:** Switches only occur if the favorite's local support drops below **10%**.
    2. **Efficacy:** Switches only occur if the local race is tight (margin between Top 2 < **5%**).
- **Risk Penalty:** Integrated a `flip(1.0 - loyalty)` check, where high-loyalty agents are less likely to switch even when both conditions are met.
- **Abs() Margin:** Fixed the margin calculation to use absolute differences, preventing logical errors in local poll assessments.

## 3. Data Output Optimization for Surrogate Modeling
The CSV export logic was streamlined to prevent "data contamination" from hardcoded GAML variables.

### Structural Changes:
- **Input Removal:** Removed hardcoded columns like `network_type` and `prop_stubborn` from the GAMA output.
- **Output-Only Focus:** GAMA now only exports calculated results: `run_id`, `day`, `variance_scores`, `social_welfare`, and `num_changes`.
- **Surrogate Model Strategy:** This ensures the LSTM model in Part 3 learns the relationship between the **True Inputs** (from Python's `run_metadata.csv`) and the **True Outputs** (from GAMA) by joining them on `run_id`.

## 4. Current File Status
- **File:** `NewModel.gaml`
- **Status:** Verified and ready for Batch Runs (1-50).
- **Next Phase:** Data generation and LSTM Training.
