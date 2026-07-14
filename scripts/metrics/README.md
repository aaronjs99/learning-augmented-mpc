# Metrics

Run-level metrics for manta trajectory rollouts.

Metrics use the first two state dimensions as position, so they work with the
active 7-state manta state and other position-leading state representations.

Current metrics:
- total position-error cost proxy
- swept minimum pairwise and obstacle distances between saved states
- collision interval count, including crossings missed by endpoint-only checks
- time-to-goal
- control effort when controls are provided
- rollout completion and safe-set admission validation
- first goal-tolerance hit step per agent and learning iteration
- shared padding of ragged per-agent histories
