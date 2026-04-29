# Open Questions

## Resolved Assumption
- Initial implementation dynamics are fixed to single-integrator to keep the simulation and metrics layer minimal and reproducible.

## Q1: Collision Constraint Handling in MPC
- Question: Hard pairwise distance only, or slack-augmented hard constraints?
- Why it matters: Feasibility and interpretation of failure cases.
- Decision: Use hard linearized pairwise constraints for nominal baseline runs. Use `soft_penalty` for `crossing_paths` because its straight-line reference has zero midpoint separation and makes hard linearized constraints infeasible.
- Status: Resolved for baseline. Crossing remains a safety stress case because soft penalties restore solver feasibility but may still allow threshold violations.

## Q2: LMPC Safe-Set Representation
- Question: Store all successful trajectory points or prune/compress?
- Why it matters: runtime/memory tradeoff vs implementation simplicity.
- Status: Open

## Q3: Learned Cost-to-Go Form
- Question: trajectory-lookup cost-to-go or small fitted approximation?
- Why it matters: model complexity and robustness.
- Status: Open

## Q4: Failure Criteria for Reporting
- Question: Which thresholds define failure in evaluation tables?
- Why it matters: reproducible and fair baseline vs LMPC comparisons.
- Candidate answers:
  - optimization infeasible
  - collision/safety violation
  - goal not reached by horizon
- Status: Open
