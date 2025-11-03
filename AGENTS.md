# Repository Guidelines

## Project Structure & Module Organization
- `main.py` drives training runs; CLI flags mirror defaults in `genrec/default.yaml`.
- `genrec/` contains the modular stack (`dataset.py`, `tokenizer.py`, `model.py`, `trainer.py`, `utils.py`); keep new components self-contained.
- `scripts/` holds diagnostics—rerun them after touching data, tokenizers, or multimodal inputs.
- Keep visuals in `asset/`; runtime outputs stay in `cache/`, `ckpt/`, `logs/`, and `tensorboard/` to honor `.gitignore`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` (or `conda env create -f env.yaml`) prepares an isolated environment.
- `pip install -r requirements.txt` installs project dependencies.
- `python main.py --category=Sports_and_Outdoors` launches the baseline run and downloads data if missing.
- `bash ap_sport.sh` and related scripts reproduce the paper sweeps with fixed seeds.
- `python scripts/validate_multimodal_alignment.py --run_dir logs/latest` and `python scripts/diagnose_image_data.py --config genrec/default.yaml` handle post-train smoke checks.

## Coding Style & Naming Conventions
- Match the repository's two-space indentation inside Python blocks; avoid auto-formatters that reflow comments or docstrings.
- Use snake_case for modules and functions, PascalCase for classes (`AbstractModel`), and expressive YAML keys for configurations.
- Add type hints for tensors and configs to mirror patterns in `genrec/`.
- Keep docstrings concise and descriptive, noting tensor shapes or expected side effects only when they aid reviewers.

## Testing Guidelines
- Co-locate unit and regression tests in a `tests/` package that mirrors `genrec` modules; run them with `pytest`.
- Pair tests with lightweight fixtures or synthetic sequences so they execute quickly on CPU.
- Validate preprocessing changes with `python scripts/diagnose_image_data.py` and inspect new cache artifacts before committing.
- Surface metrics via TensorBoard and share paths when reporting results.

## Commit & Pull Request Guidelines
- Use short, imperative commit titles in the style of recent history (e.g., `update merge image data tokenizer`).
- Expand in the body or PR description: motivation, key code or config moves, and validation commands.
- Keep PRs scoped to a subsystem, call out downstream retraining requirements, and highlight config edits.
- Attach evidence (logs, TensorBoard links, sample outputs) and tag maintainers when touching shared abstractions like `genrec/dataset.py` or `genrec/tokenizer.py`.

## Data & Configuration Notes
- Keep credentials out of version control; load them via environment variables referenced in `env.yaml`.
- Store large checkpoints solely in `ckpt/` with descriptive names (e.g., `actionpiece_step123.pt`) and exclude them from commits.
- Update YAML defaults alongside code to ensure the `Sports_and_Outdoors` baseline stays reproducible.
