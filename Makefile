#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = edumind
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run all unit and integration tests via pytest
.PHONY: test
test: requirements
	$(PYTHON_INTERPRETER) -m pytest tests/ -v


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Launch EduMIND Streamlit dashboard
.PHONY: app
app: requirements
	$(PYTHON_INTERPRETER) -m streamlit run edumind/app.py --server.headless true

## Run the performance benchmark suite
.PHONY: benchmark
benchmark: requirements
	$(PYTHON_INTERPRETER) -m edumind.utils.benchmark

## Run EduMIND tests via pytest
.PHONY: test-edumind
test-edumind: requirements
	$(PYTHON_INTERPRETER) -m pytest tests/ -v

## Install Label Studio host requirements
.PHONY: install-ls
install-ls:
	uv sync --extra label-studio
	uv pip install label-studio

## Launch Label Studio and ML backend locally on host
.PHONY: run-ls
run-ls: install-ls
	bash label_studio_backend/setup_env.sh

## Start the containerized Label Studio stack (UI + ML Backend + Qdrant + Neo4j)
.PHONY: docker-up
docker-up:
	docker compose up --build -d

## Alias to start the full containerized stack
.PHONY: docker-up-all
docker-up-all: docker-up

## Stop the containerized Label Studio stack
.PHONY: docker-down
docker-down:
	docker compose down

## View logs for the containerized Label Studio stack
.PHONY: docker-logs
docker-logs:
	docker compose logs -f

## Open Neo4j Web UI in browser
.PHONY: neo4j-ui
neo4j-ui:
	open http://localhost:7474

## Open Qdrant Web Dashboard in browser
.PHONY: qdrant-ui
qdrant-ui:
	open http://localhost:6334

## Wipe Qdrant vector index and Neo4j knowledge graph databases
.PHONY: clean-db
clean-db:
	$(PYTHON_INTERPRETER) -c "from edumind.core.container import get_vectorstore, get_graph_store; get_vectorstore().clear_index(); get_graph_store().clear_graph()"

## Install ColPali optional multimodal dependencies
.PHONY: colpali-setup
colpali-setup:
	uv sync --extra colpali

## Display data directory stats and database info
.PHONY: data-stats
data-stats:
	$(PYTHON_INTERPRETER) -c "from edumind.utils.data_manager import get_storage_stats; import pprint; pprint.pprint(get_storage_stats())"



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
