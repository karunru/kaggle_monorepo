# Repository Guidelines

## Project Structure & Module Organization
- `codes/src/`: Core library modules (`features/`, `models/`, `evaluation/`, `utils/`, etc.).
- `codes/exp/expNNN/`: Experiment sandboxes (self-contained configs/scripts). Example: `codes/exp/exp021/`.
- `tests/`: Pytest suite, organized by experiment and module (e.g., `tests/test_exp021_*.py`).
- `data/`: Local datasets (do not commit). `outputs/`: Experiment artifacts and reports. `docs/`: Plans and notes.
- `sub/`, `codes/`, `deps/`: Kaggle dataset/kernel packaging.

## Build, Test, and Development Commands
- Install tools: `mise install` (sets Python/uv) then `uv sync` (installs deps).
- Run tests: `uv run pytest -q` or target an experiment: `uv run pytest tests/test_exp021_*.py`.
- Lint/format: `uv run ruff check .` and `uv run ruff format`.
- Type check: `uv run mypy codes/src`.
- Kaggle packaging (via mise tasks):
  - `mise run update-codes`
  - `mise run create-output-dataset --experiment_name exp021`
  - `mise run update-subs --experiment_name exp021`
  - `mise run track-submission`

## Coding Style & Naming Conventions
- Python 3.12, 4-space indent, type hints required in `codes/src` and new experiments.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Keep experiments isolated: no cross-imports between `expNNN`; import shared code from `codes/src`.
- Prefer pure functions; limit side effects to `expNNN` entry scripts.

## Testing Guidelines
- Framework: pytest. Place unit tests in `tests/` mirroring modules/experiments.
- Names: files `test_*.py`, tests `test_*` functions.
- Scope: add tests for new data transforms, loss/metric logic, and dataset contracts.
- Run focused tests during development (example above); ensure suite passes before PR.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped. Example: `exp021: add dataset + model; update tests`.
- PRs: include summary, motivation, key commands, experiment name (e.g., `exp021`), and sample outputs path (e.g., `outputs/exp021/`).
- Link related docs under `docs/` (e.g., `docs/exp021_plan.md`) and reference affected tests.
- CI expectations: lint, type check, and tests should pass locally before requesting review.

## Security & Configuration Tips
- Do not commit raw data, secrets, or `.venv/`. Use `mise` tasks to publish Kaggle datasets/kernels.
- GPU wheels come via uv index overrides; avoid pin changes unless necessary.

## Claude Code
- Read CLAUDE.md