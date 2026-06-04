# Project Notes

The current implementation is config-driven and keeps executable entry points at
the repository root or in `scripts/`.

Run the main manta LMPC workflow with:

`python run.py`

The default settings live in `config/manta.yaml`. Use command-line flags for
temporary overrides such as fewer LMPC iterations, a different output directory,
or enabling the final GIF.

Useful shortcuts:
- `python run.py baseline` runs APF iteration 0 only.
- `python run.py sanity` runs a zero-control manta rollout check.
- `python run.py --make-video` runs the full manta LMPC workflow and writes a GIF.
