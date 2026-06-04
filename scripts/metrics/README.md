# Metrics

Run-level metrics for manta trajectory rollouts.

Metrics use the first two state dimensions as position, so they work with the
active 7-state manta state and with legacy 2D states.

Current metrics:
- total position-error cost proxy
- minimum pairwise distance
- collision count
- time-to-goal
- control effort when controls are provided
