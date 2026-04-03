# Learning-Augmented Model Predictive Control

## Overview
This repository is a graduate research scaffold for studying learning-augmented model predictive control (MPC). It currently holds standardized project structure, documentation conventions, and collaboration workflow assets while implementation choices are being finalized by the team.

## Motivation
Classical MPC provides structure, constraint handling, and interpretability, but performance can degrade under model mismatch and changing dynamics. Learning-augmented MPC aims to retain the reliability of model-based control while improving adaptation and predictive quality through data-driven components.

## Research Direction
- Synthesize ideas from multiple MPC + learning papers into one coherent framework
- Compare integration strategies (e.g., residual dynamics, learned costs, learned terminal components)
- Evaluate tradeoffs between robustness, sample efficiency, and computational cost

## Project Objectives
- Implement a baseline MPC pipeline suitable for controlled comparisons
- Add learning-based augmentation modules with clear interfaces
- Build a fusion layer (`src/fusion/`) for multi-paper composition and ablation
- Run reproducible experiments and document findings rigorously

## Repository Structure
- `docs/`: Project documentation, planning, paper synthesis, and presentation/report scaffolds
- `src/`: Future implementation modules (`mpc`, `learning`, `fusion`, `simulation`)
- `experiments/`: Experiment plans and run metadata conventions (no implementation here yet)
- `results/`: Result logging conventions and future artifacts
- `scripts/`: Utility scripts (intentionally minimal at this stage)
- `references/`: Reference management (papers, links, and bibliography)

## Current Status
- Planned:
  - Baseline controller design and evaluation protocol
  - Learning augmentation strategy selection from literature
- Scaffolded:
  - Repository layout, documentation index, paper-note templates, planning docs
  - Contribution and collaboration workflow templates
- Implemented:
  - No MPC/learning implementation code yet

## Team
- Aaron John Sabu (`@aaronjs99`)
- Nicholas Councell (`@nick12512`)
- Ben (`@benw7454`)

## Collaboration Workflow
- Use short-lived feature branches for documentation and code changes
- Open PRs against `main` with clear scope and explicit assumptions
- Keep commits focused and reviewable
- Update docs whenever structure, experiments, or integration plans change
- See `CONTRIBUTING.md` for detailed norms

## Setup Status
Environment and tooling setup is not finalized yet. The repository is intentionally in a scaffolded holding state while the team finalizes scope and integration priorities.

## Documentation Map
- `docs/README.md`: Documentation index and navigation
- `docs/paper_notes/`: Per-paper extraction templates and notes
- `docs/literature_review/`: Cross-paper synthesis
- `docs/project_plan/`: Milestones, roadmap, open questions, and report/presentation planning
- `docs/presentation_notes/`: Slide planning and speaking-point scaffolds

## Next Steps
1. Align on baseline problem scope and first paper set at the next team meeting
2. Confirm integration direction and evaluation plan
3. Begin implementation only after team alignment on interfaces and milestones

## Contribution Expectations
- Keep changes scoped, documented, and reproducible
- Avoid undocumented assumptions in experiments and integration choices
- Prefer explicit design notes before major implementation changes

## Citation / Attribution (Placeholder)
If this project is used for reports, papers, or external collaboration, add a formal citation entry here once title/authorship details are finalized.

## License Status
No project license has been selected yet. See `LICENSE_STATUS.md` for current guidance.
