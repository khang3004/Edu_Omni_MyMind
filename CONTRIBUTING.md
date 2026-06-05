# Contributing to EduMIND

First, thank you for contributing to **EduMIND (Bilingual Lecture Assistant)**! To maintain high code quality, architectural consistency, and reproducible NLP/ML research, we require all contributors to adhere to the guidelines outlined in this document.

---

## 1. Git Workflow & Branch Strategy

We follow a structured branch naming and merge workflow. Never push directly to `main`. Always create a local branch and submit a Pull Request (PR) for review.

### Branch Naming Conventions
* **Features:** `feat/feature-name` (e.g., `feat/add-whisper-m1-support`)
* **Bug Fixes:** `fix/bug-description` (e.g., `fix/qdrant-memory-leak`)
* **Refactoring:** `refactor/component-name` (e.g., `refactor/di-container`)
* **Documentation:** `docs/page-name` (e.g., `docs/update-installation-guide`)
* **Testing:** `test/test-case` (e.g., `test/add-translator-tests`)

---

## 2. Commit Message Standards

Commit messages must follow the **Conventional Commits** specification. This helps in auto-generating changelogs and keeping history readable.

Format:
```text
<type>(<scope>): <short description>
```

### Approved Commit Types:
* **`feat`**: A new feature, module, or architecture component.
* **`fix`**: A bug fix (e.g., correcting ASR post-processing regex).
* **`docs`**: Documentation updates (e.g., README or docstrings).
* **`style`**: Formatting, missing semi-colons, white-space changes (no logic changes).
* **`refactor`**: Code restructuring that neither fixes a bug nor adds a feature.
* **`perf`**: Changes that improve execution latency, memory utilization, or hardware acceleration.
* **`test`**: Adding missing tests or refactoring test suites.
* **`chore`**: Maintenance tasks, library upgrades, or modifying build configurations.

### Examples:
* `feat(asr): add teencode mapping post-processor for Vietnamese lecture notes`
* `fix(rag): resolve off-by-one error during layout-aware PDF text chunking`
* `refactor(core): decouple translation providers using the Strategy pattern`

---

## 3. Clean Architecture Guidelines

EduMIND is designed to be highly modular, testable, and production-ready. We enforce the following structural constraints:

### No Global Singletons
Avoid importing global shared instances of heavy services (e.g., a shared Qdrant client, embedding models, or Whisper pipelines). Instead, use **Dependency Injection (DI)**. Services must accept their dependencies via constructor parameters.

* **Incorrect:**
  ```python
  from edumind.core.clients import qdrant_client
  class MultimodalRAG:
      def search(self, query):
          qdrant_client.query(...)
  ```
* **Correct:**
  ```python
  class MultimodalRAG:
      def __init__(self, vector_store: VectorStore):
          self._vector_store = vector_store
  ```

### Strategy Pattern for Providers
Services that connect to external APIs or local models must define a clear interface (base class) and use specific strategy implementations (e.g., `TranslationProvider` has `RuleBasedTranslationProvider` and `HuggingFaceTranslationProvider`).

### Configuration & Validation
Hyperparameters and server options must not be hardcoded. Define settings in `configs/default_config.yaml` or `.env` and load them using schema-validated Pydantic structures (`edumind/config/`).

---

## 4. Code Style & Formatting

We use **Ruff** for strict linting, import sorting, and code formatting. Before committing any changes, you must format the codebase:

```bash
# Format codebase (checks & fixes imports and style)
make format

# Verify strict compliance
make lint
```

### Style Expectations
1. **Type Hints:** All function parameters and return types must be fully type-hinted.
2. **Docstrings:** All classes, modules, and public functions must have docstrings in **Google Docstring Format**.
3. **Python Version:** The codebase is locked to Python `3.10` via `pyproject.toml` and `uv`.

---

## 5. Jupyter Notebooks

Jupyter Notebooks in `notebooks/` should only be used for **Exploratory Data Analysis (EDA)** or **prototyping**.
* **Do not** write core product features or pipeline steps as notebooks.
* Always **Restart & Clear All Outputs** before committing notebooks to minimize Git diff size.
* Migrate finalized prototypes to standard modules under `edumind/` as soon as they are stable.

---

## 6. Pull Request Checklist

Before submitting your PR, ensure:
- [ ] Code formats cleanly with `make format` and passes `make lint`.
- [ ] All unit and integration tests pass successfully (`make test`).
- [ ] All new functions/classes are documented with Google-style docstrings.
- [ ] No global singletons are introduced, and dependencies are cleanly injected.
