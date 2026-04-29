# Simulation

Minimal reusable 3-agent simulation layer.

## Files
- `environment.py`: single-integrator environment + rollout helper.
- `scenarios.py`: named scenario definitions used by all experiments.

## Rules
- Keep one rollout path.
- Keep scenario definitions centralized.
- Do not add MPC logic here.
