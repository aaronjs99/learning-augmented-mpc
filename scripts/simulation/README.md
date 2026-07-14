# Simulation

Reusable simulation layer for the manta workflow.

- `environment.py`: active variable-agent manta environment, a fixed-three compatibility wrapper, and a legacy 2D environment kept for reference.
- `scenarios.py`: scenario and obstacle dataclasses plus YAML-backed lookup helpers.

Scenario starts, goals, obstacles, and safety distances live in `config/manta.yaml`.
