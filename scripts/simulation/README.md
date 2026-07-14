# Simulation

Reusable simulation layer for the manta workflow.

- `environment.py`: active variable-agent manta environment and fixed-three compatibility wrapper.
- `scenarios.py`: scenario and obstacle dataclasses plus YAML-backed lookup helpers.

Scenario starts, goals, obstacles, and safety distances live in `config/manta.yaml`.
