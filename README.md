# NLP Research & Model Training Hub

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repository houses a highly optimized, enterprise-grade, and reproducible **NLP Research & Deep Learning Pipeline**. It is designed with modularity, schema-validated configuration loading, strict coding standards, and reproducibility at its core, enabling research teams to prototype, train, and evaluate state-of-the-art NLP models.

---

## 🚀 Key Features

* **⚡ Ultra-fast Environment Management:** Handled natively via `uv` (replaces standard pip/poetry).
* **⚙️ Config-Driven Architecture:** Decoupled hyperparameters managed in `configs/default_config.yaml` and loaded via schema-validated Pydantic structures.
* **🏷️ Model Factory Registry:** Simple string-based registration decorators to easily switch between Transformers (BERT, DistilBERT, RoBERTa) and custom architectures (BiLSTM baselines).
* **📉 Production-grade PyTorch Trainer:** Custom loop supporting auto-device acceleration (CUDA, MPS Apple Silicon, CPU), early stopping, and metric evaluation.
* **📏 Standardized Evaluator:** Automated computing of macro/weighted F1, precision, recall, and accuracy benchmarks.
* **🧹 Strict Quality Control:** Ruff configured for instant PEP-8 styling, import sorting, and code smell auditing.

---

## 📂 Project Organization

```
├── LICENSE                      <- Open-source license (MIT)
├── README.md                    <- The top-level documentation for researchers
├── CONTRIBUTING.md              <- Git branch, commit, formatting, and notebook guidelines
├── Makefile                     <- Simple automation tasks (sync, format, train, evaluate)
├── pyproject.toml               <- Structured dependencies & strict tool configurations (Ruff)
├── configs/
│   └── default_config.yaml      <- Centralized hyperparameters for model & training runs
│
├── data
│   ├── external                 <- Data from third party sources
│   ├── interim                  <- Intermediate data that has been transformed
│   ├── processed                <- Split train/val partitions ready for modeling
│   └── raw                      <- Original, immutable data dump
│
├── docs                         <- default documentation project (mkdocs)
├── models                       <- Serialized model weights (.pt checkpoints)
├── notebooks                    <- Jupyter notebooks for EDA and prototypes
├── references                   <- Explanatory reference materials
├── reports                      <- Generated analysis (figures, metrics summaries)
│   └── figures
│
├── nlp_model_training           <- Core source package
│   ├── __init__.py              <- Exports primary classes and helpers
│   ├── config.py                <- Pydantic validation schema and config merger
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py        <- Text normalization, cleaning, HF tokenizer loading
│   │   └── dataset.py           <- PyTorch Datasets and pad-collation builders
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── architectures.py     <- Registered PyTorch/Transformer model structures
│   │   └── registry.py          <- Decoupled model factory registration
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           <- Custom training, early-stopping, and validation loops
│   │   └── metrics.py           <- Classification metrics (Accuracy, F1, Precision, Recall)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py           <- Multi-hardware accelerator selector and random seeds
│   │
│   ├── dataset.py               <- CLI entry point for downloading & processing data
│   └── train.py                 <- CLI entry point to trigger model training runs
│
└── tests                        <- Unit tests directory
    ├── test_data.py             <- Verifies preprocessing, cleaning, and datasets
    └── test_models.py           <- Verifies registry lookups and model instantiations
```

---

## 🛠️ Environment Setup & Installation

We use `uv` for python environment compilation. If you don't have `uv` installed, get it via [astral.sh/uv](https://astral.sh/uv).

1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd NLP_Playground_code_ala
   ```

2. **Sync the Environment:**
   Run `uv sync` (or through the Makefile) to compile the virtual environment and install all packages in editable mode:
   ```bash
   make requirements
   ```
   *This automatically builds the virtual environment in `.venv/` and registers the local `nlp_model_training` package.*

---

## 🏃 Pipeline Execution

### 1. Data Preprocessing
Generate mock raw text inputs (if raw data is missing), normalize textual components, partition data into train/val datasets, and serialize:
```bash
make data
```

### 2. Model Training
Load hyperparameters from `configs/default_config.yaml`, fetch datasets, instantiate the model architecture, and launch the PyTorch Trainer:
```bash
make train
```

### 3. Run Testing Suites
Execute unit tests for tokenizers, datasets, and registered architectures:
```bash
make test
```

### 4. Code Quality & Styling
Format all source files and run strict checkers:
```bash
make format
```

---

## ⚙️ Hyperparameter Configuration

To tweak parameters for your experiments, directly edit `configs/default_config.yaml`:

```yaml
# configs/default_config.yaml
seed: 42

data:
  max_length: 128
  batch_size: 32

model:
  model_type: "transformer" # Use "transformer" or "custom_lstm"
  pretrained_model_name: "distilbert-base-uncased"
  dropout: 0.1

training:
  epochs: 5
  learning_rate: 2.0e-5
  weight_decay: 0.01
```

You can override any variable programmatically or via environment variables, as `nlp_model_training/config.py` merges environment inputs automatically.
