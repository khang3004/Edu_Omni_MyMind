# Contributing Guidelines - NLP Research & Development

Welcome to the NLP Research Team! To maintain high code quality, absolute scientific reproducibility, and seamless collaboration, all members are expected to adhere to the standards outlined in this guide.

---

## 1. Git Workflow & Branch Strategy

We follow a structured branch strategy. Never commit directly to `main`. Always create a branch from `main` and submit a Pull Request (PR) for review.

### Branch Naming Conventions
* **Features:** `feat/feature-name` (e.g., `feat/bert-classifier`)
* **Bug Fixes:** `fix/bug-description` (e.g., `fix/dataloader-padding-leak`)
* **Experiments:** `research/experiment-name` (e.g., `research/eval-lora-tuning`)
* **Refactoring:** `refactor/area-changed` (e.g., `refactor/pydantic-config`)
* **Documentation:** `docs/page-name` (e.g., `docs/add-installation-guide`)

---

## 2. Commit Message Standards

We enforce **Conventional Commits** to auto-generate changelogs and keep commit histories clean. Commit messages must be structured as:

```
<type>(<scope>): <short description>
```

### Approved Commit Types:
* **`feat`**: A new model architecture, data preprocess step, or utility.
* **`fix`**: A bug fix (e.g., resolving a dimension mismatch in attention).
* **`docs`**: Documentation updates (e.g., editing `README.md`).
* **`style`**: Changes that do not affect the meaning of the code (formatting, white-space).
* **`refactor`**: Code changes that neither fix a bug nor add a feature.
* **`perf`**: A code change that improves compute/inference speed or memory consumption.
* **`test`**: Adding missing tests or correcting existing tests.
* **`chore`**: Maintenance tasks, library upgrades, or modifying project build configurations.

### Examples:
* `feat(model): add registry decorator and custom transformer architecture`
* `fix(data): fix padding collation truncation inside dataloader`
* `docs(readme): add training execution guide for multi-GPU hardware`

---

## 3. Code Style & Formatting

We use **Ruff** for strict linting, import sorting, and code formatting. Before committing any code, you must format it using our configurations.

### Commands:
To format and fix lint errors automatically:
```bash
make format
```
To run checkers only:
```bash
make lint
```

### General Style Expectations:
1. **Type Hints:** All function signatures must be fully type-hinted (parameters and return types).
2. **Docstrings:** All classes and public functions must have descriptive docstrings following the Google Docstring Format.
3. **No Magic Numbers:** Define hyperparameter values or threshold settings in `configs/default_config.yaml` rather than hardcoding them in Python files.

---

## 4. Jupyter Notebook Best Practices

Jupyter Notebooks are fantastic for exploratory data analysis (EDA) and rapid prototyping, but they can quickly lead to out-of-order execution bugs and massive Git diffs if not managed correctly.

1. **Keep Notebooks in `/notebooks`:** Never run training or production runs from notebooks. Use them only for EDA, visualization, or quick proof-of-concept testing.
2. **Strip Outputs Before Committing:** Run `Kernel -> Restart & Clear All Outputs` before committing notebooks. This keeps Git diffs minimal and clean.
3. **Migrate to Scripts:** Once a prototype model or preprocessing method is stable, refactor and migrate it immediately into the core package (`nlp_model_training/`).

---

## 5. Pull Request & Review Checklist

Before marking your PR as ready for review, check off these items:
- [ ] My code successfully builds and passes all tests (`make test`).
- [ ] My code adheres to the Ruff formatting requirements (`make lint`).
- [ ] I have fully documented new classes/functions with docstrings.
- [ ] If I changed configurations, I updated `configs/default_config.yaml`.
- [ ] My commit messages follow the Conventional Commits specification.
