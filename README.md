# EduMIND — Multimodal Bilingual Lecture Assistant & Active Learning Pipeline

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/Package%20Manager-uv-de5d68?logo=astral" alt="uv" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Vector%20DB-Qdrant-red?logo=qdrant" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Annotation-Label%20Studio-orange?logo=labelstudio" alt="Label Studio" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" />
</div>

---

**EduMIND** is an enterprise-grade, highly modular **Bilingual Lecture Assistant & Active Learning Pipeline**. Designed specifically for academic environments where lectures mix languages (e.g., Code-Mixed Vietnamese-English, such as *"hôm nay chúng ta học attention mechanism"*), EduMIND transcribes bilingual speech, measures code-switching metrics, translates text preserving technical terms, indexes slides, and executes retrieval-augmented generation (RAG).

The system integrates a **Human-in-the-Loop Active Learning** framework powered by **Label Studio** and an ML backend to continually harvest human-corrected data, immediately updating the local knowledge base and building a gold-standard corpus.

---

## 🎙️ Core Components & Architecture

```text
                  +----------------------------------+
                  |    Bilingual Audio Lecture       |
                  +-----------------+----------------+
                                    |
                                    v
                       [ 🎙️ Bilingual Note-Taker ]
                         Whisper ASR + Post-RegEx
                                    |
                                    v
                     [ 🔄 VietMix Translation & CMI ]
                        Dict / Seq2Seq Translation
                                    |
                                    v
                      [ 📚 Anti-Forget RAG Engine ]
                         PDF Chunking -> Qdrant
                                    |
        +---------------------------+---------------------------+
        |                                                       |
        v (Retrieval QA)                                        v (Active Learning)
 [ Streamlit Assistant ]                              [ Label Studio UI (Port 8080) ]
   RAG Chat + Analytics                                  TA/Human Review & Correction
                                                                |
                                                                v
                                                       [ edumind_ml_backend ]
                                                         - Writes to corpus.jsonl
                                                         - Re-indexes to Qdrant Vector DB
```

### 1. Bilingual Note-Taker (Speech ASR)
* Utilizes OpenAI's **Whisper** model (with dynamic CPU/MPS/CUDA hardware detection).
* Integrates a post-processing **Teencode Resolver** to map colloquial abbreviations and slang to formal academic terms.
* Computes segment-level confidence scores mapped from Whisper's average log-probabilities.

### 2. VietMix Machine Translation
* Implements token-level language identification (`vi`, `en`, or `other`) to compute the **Code-Mixing Index (CMI)**:
  $$\text{CMI} = \frac{N - \max(w_{\text{vi}}, w_{\text{en}})}{N}$$
  *(where $N$ is the total count of linguistic tokens).*
* Decouples translation providers via the **Strategy Pattern**:
  * **`RuleBasedTranslationProvider`**: High-performance, zero-latency dictionary lookup mapping.
  * **`HuggingFaceTranslationProvider`**: Neural Seq2Seq model (e.g., `Helsinki-NLP/opus-mt-vi-en`) with automatic rule-based fallback.

### 3. Anti-Forget RAG Engine
* Handles **Layout-Aware PDF Chunking** (splitting slides, capturing section headers, and avoiding sentence fragmentation).
* Integrates **Qdrant Vector Database** (supporting in-memory modes for local prototyping or dedicated server connections).
* Applies keyword-boosting, hybrid searches, and **Cross-Encoder Re-Ranking** (`ms-marco-MiniLM-L-6-v2`) before synthesis.
* Supports pluggable generative models (e.g. Gemini, Groq) via LangChain integrations.

### 4. Active Learning Loop (Label Studio Hook)
* An administrative dashboard built using **Label Studio** interfaces with a custom **EduMIND ML Backend** (running on Flask).
* When a human annotator reviews and submits a correction:
  1. The gold-standard text is appended to an audit file (`data/processed/corpus.jsonl`).
  2. The text is dynamically vectorized and indexed into the active Qdrant database to immediately update the RAG knowledge pool.

---

## 📂 Project Organization

```text
├── LICENSE                           <- MIT License
├── README.md                         <- This main system guide
├── CONTRIBUTING.md                   <- Development, CI/CD, and style guidelines
├── Makefile                          <- Task automation commands
├── pyproject.toml                    <- Project specs & package dependencies
├── uv.lock                           <- Lockfile for exact package reproducibility
├── docker-compose.yml                <- Docker compose configuration for the LS stack
├── Dockerfile.label-studio           <- Multi-stage Docker build for the ML backend
│
├── configs/
│   └── default_config.yaml           <- Hyperparameter configurations
│
├── data/
│   ├── raw/
│   │   ├── audio_chunks/             <- Raw lecture wav chunks
│   │   └── pdf_slides/               <- PDF lecture materials
│   └── processed/
│       └── corpus.jsonl              <- Target gold-standard active learning corpus
│
├── edumind/                          <- Core Python source package
│   ├── app.py                        <- Streamlit frontend implementation
│   ├── config/                       <- Pydantic validation definitions
│   ├── core/                         <- Logger, Dependency Injection container, Exceptions
│   ├── models/                       <- Data models & schemas (ASR, Translation, RAG)
│   ├── modules/                      <- Core engines (RAG, Speech ASR, VietMix Translator)
│   ├── services/                     <- Strategy implementations (Embedding, LLM, Translation)
│   └── utils/                        <- String utilities, file helpers, model registries
│
├── label_studio_backend/             <- Flask active learning ML Backend
│   ├── _wsgi.py                      <- WSGI entry point for container execution
│   ├── model.py                      <- Label Studio ML backend subclass code
│   └── setup_env.sh                  <- Shell bootstrapper for local host testing
│
└── tests/                            <- Complete unit & integration test suite
```

---

## 🛠️ Installation & Environment Setup

This project uses `uv` for python virtual environment compilation. Ensure it is installed on your machine.

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd edumind
   ```

2. **Synchronize environment and install dependencies:**
   ```bash
   make requirements
   ```
   *This automatically builds a virtual environment under `.venv/` and installs the package in editable mode.*

3. **Configure Environment Variables:**
   Copy the template file to `.env` and fill in your values (like LLM API keys):
   ```bash
   cp .env.example .env
   ```

---

## 🏃 Execution Guide

The system can be run in two main ways: **Local Host Development** or **Containerized Docker Compose Stack**.

### 1. Running the Streamlit Lecture Assistant
To launch the interactive frontend dashboard:
```bash
make app
```
*Access the interface at `http://localhost:8501`.*

### 2. Running the Label Studio Active Learning Stack

#### Option A: Running Containerized (Recommended)
This launches both Label Studio UI and the EduMIND ML Backend in a shared Docker network:
```bash
# Start the stack in background
make docker-up

# Check container status
docker compose ps

# View logs
make docker-logs

# Stop the stack
make docker-down
```
* Access **Label Studio UI**: `http://localhost:8080` (Credentials: `admin@edumind.local` / `edumind_admin_2024`)
* Access **ML Backend**: `http://localhost:9090` (connected at `http://ml-backend:9090` inside Docker)

#### Option B: Running Local on Host (Directly)
If you want to run Label Studio and the ML Backend natively on your host system:
```bash
# Installs Label Studio binaries and starts both servers in one terminal session
make run-ls
```

---

## 🧪 Testing & Code Quality

### Running Tests
To run the complete suite of 50+ unit and integration tests:
```bash
make test
```

### Checking Style & Formatting
Code formatting is strictly checked using **Ruff**. Always format your code before pushing changes:
```bash
# Auto-format and resolve lint errors
make format

# Dry-run check
make lint
```
