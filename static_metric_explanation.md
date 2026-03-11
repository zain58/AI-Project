# Analysis of Static Social Welfare Metric

## The "Randomly Static" Paradox
Although every voter is assigned a **random** list of preferences (e.g., Voter A ranks Macron #1 and Le Pen #10, while Voter B ranks Macron #5 and Le Pen #2), these lists are **Fixed** once the simulation starts.

### Why the line is flat:
1.  **Metric Definition:** The Social Welfare is calculated based on the **Global Top 2 Finalists** (the candidates currently leading the poll).
2.  **Leader Stability:** In most runs, the same two candidates (e.g., Macron and Le Pen) take the lead on Day 1 and stay there for all 60 days.
3.  **Constant Math:** Since the **Finalists' identities** don't change and the **Voters' hearts** (their internal ranking of those finalists) don't change, the mathematical result (`Rank_of_Finalist1 + Rank_of_Finalist2`) is the same every day.

## The "Undisputed Leader" Effect
Strategic voting creates a feedback loop that reinforces this stability:
- Strategic and Mixed agents see who is winning and switch their votes to them.
- This makes the leaders even stronger, making it nearly impossible for a lower-ranked candidate to "knock out" one of the Top 2.
- Without a change in the **Top 2 identities**, the Social Welfare value remains a flat line.

## Impact on Machine Learning (Part 3)
For your LSTM Neural Network, a flat line is **"Dead Data."**
- A machine learning model learns by observing how **Changes in Input** cause **Changes in Output**.
- If `Social Welfare` never moves, the LSTM cannot learn how network types or agent proportions affect voter satisfaction.

## Proposed Enhancement: "Current Vote Satisfaction"
To provide a dynamic signal for the surrogate model, we should track the rank of the candidate the agent is **actually voting for** each day:
- **Day 0:** Everyone votes for their #1 choice (Welfare = 1.0).
- **Day 30:** 300 people have switched to their #4 choice strategically (Welfare = ~1.9).
- **Result:** This creates a moving line that the LSTM can actually use to understand the "cost" of strategic voting.
