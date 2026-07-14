# Reporting

`manta.py` owns the stable run-summary contract and generated LMPC artifacts.
It separates numerical report preparation from filesystem and plotting work:

- `prepare_manta_report`: compute summary fields and plot-ready arrays.
- `save_manta_run_report`: write JSON/CSV, diagnostic PNGs, and the optional GIF.

Command-line entry points should execute runs and delegate artifact generation
here instead of duplicating metrics or serialization logic.
