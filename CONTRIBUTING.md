# Contributing Guidelines

This project is a research collaboration for MAE 271D. Keep contributions practical, traceable, and easy to review.

## Branching Expectations
- Branch from `main` for each scoped task
- Use descriptive branch names, e.g.:
  - `docs/paper-notes-template-update`
  - `infra/repo-hygiene-standardization`
- Avoid long-lived branches unless explicitly coordinated

## Pull Request Expectations
- Keep PRs focused on one logical change
- Include:
  - what changed
  - why it changed
  - what remains out of scope
- Link related issue(s) when applicable
- For research decisions, include rationale and citations in docs

## Commit Message Expectations
- Use concise, professional commit subjects
- Prefer imperative style, e.g.:
  - `Add experiment tracking template`
  - `Refine paper integration issue template`
- Separate unrelated work into separate commits

## Proposing Papers to Integrate
- Open a `paper-integration` issue (template provided)
- Add a note under `docs/paper_notes/` using `TEMPLATE.md`
- Capture assumptions, expected fit, and fusion opportunities before implementation

## Experiment Documentation Expectations
When adding or running experiments, include documentation updates in the same PR where appropriate:
- planned experiment design in `experiments/`
- run metadata and observations in `results/`
- links to relevant papers/assumptions in `docs/`

## Updating Docs with Structure Changes
If folders, conventions, or workflows change:
- update root `README.md`
- update `docs/README.md`
- update any affected templates

## Code + Documentation Coupling Rule
If a code change materially affects experiment design, assumptions, interfaces, or evaluation, include a corresponding documentation change in the same PR.

## Team Collaboration Norms
- Default reviewers: Aaron, Nicholas, Ben (as available)
- Raise blockers early in meeting notes or PR discussion
- Resolve open questions in writing (issue/PR/docs) to keep research decisions auditable
