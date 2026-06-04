# Dynamics

Shared 7-state manta/CPG dynamics.

The same constants and RK4 integration logic are used by APF initialization,
zero-control simulation, and CasADi LMPC optimization. Numeric constants should
be edited in `config/manta.yaml`, then loaded into `MantaDynamicsConfig`.
