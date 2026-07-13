# Config

`manta.yaml` is the default runtime configuration for the project.

It owns the scenario states/goals, obstacle, dynamics constants, APF initializer
tuning, LMPC horizons/weights/slacks, and output defaults. CLI flags in
`run.py` are intended for quick overrides only.

The LMPC defaults are conservative: SVM hyperplanes use a margin larger than
half the configured pairwise safety distance, and terminal safe-set matching is
position-only by default so oscillator phase states do not over-constrain the
short-horizon NLP.

`warm_start_control_blend` controls how much stored safe-set control history is
used in IPOPT initialization. `0.0` means constant nominal controls only, while
`1.0` means raw stored controls only.
