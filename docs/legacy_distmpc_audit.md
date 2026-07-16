# Legacy `distmpc` Audit

## Repository Provenance

- Local path: `C:/Users/aaron/Desktop/aaronjs99/distmpc`
- GitHub remote: `git@github.com:aaronjs99/distmpc.git`
- Remote tip at audit: `adc6fdb` (`Initial commit`)
- Local-only tip at audit: `47a6d77` (`Refactor MPC pipeline: add modular
  main script and ADMM solver`)
- License: MIT

## What Was Present

`src/main.py` is a centralized GEKKO trajectory optimizer for multiple planar
double integrators. It repeatedly solves a finite-horizon terminal-tracking
problem and applies `numActSteps` controls before replanning. It uses a remote
GEKKO service, has no collision constraints, no decentralized information
boundary, no tests, and hard-codes the scenario and solver settings.

`src/admm.py` is not an ADMM implementation. It contains an incomplete objective
statement that makes the file syntactically invalid, invokes a centralized
comparison solve at import time, and has no consensus variables, augmented
Lagrangian update, dual update, residual test, or distributed MPC integration.

## Concept Retained Here

The useful idea is block receding-horizon execution: apply more than one control
step before recomputing guidance. It is now represented by
`simulation.guidance_update_interval_steps` in `config/harbor.yaml`. The harbor
simulator holds each platform's last control between updates and reports
`guidance_update_count`, allowing computation reduction to be tested against
swept safety and completion rather than assumed beneficial.

The unfinished ADMM file is not copied. A future ADMM contribution would need
real platform-local subproblems, shared collision/intent copies, primal and dual
residuals, adaptive penalty updates, communication accounting, and comparison
against the existing decentralized ETA-priority policy.

This audit preserves the repository's useful research intent and exact commit
identity without retaining broken duplicate source code.
