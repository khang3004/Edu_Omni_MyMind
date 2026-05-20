from pathlib import Path
import sys

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml

# Absolute Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJ_ROOT / "configs" / "default_config.yaml"


class DataConfig(BaseModel):
    raw_data_path: str = "data/raw/dataset.csv"
    processed_data_path: str = "data/processed/dataset.csv"
    test_size: float = 0.2
    max_length: int = 128
    batch_size: int = 32
    num_workers: int = 0


class ModelConfig(BaseModel):
    model_type: str = "transformer"
    pretrained_model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    hidden_size: int = 256
    dropout: float = 0.1


class TrainingConfig(BaseModel):
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    save_steps: int = 100
    logging_steps: int = 10
    output_dir: str = "models/"
    use_early_stopping: bool = True
    patience: int = 2


class LoggingConfig(BaseModel):
    log_level: str = "INFO"
    use_wandb: bool = False


class Settings(BaseSettings):
    """
    Schema-validated project settings loaded from YAML and overridden by environment variables.
    """

    project_name: str = "nlp_model_training"
    experiment_name: str = "distilbert_classification"
    seed: int = 42

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load_config(cls, yaml_path: Path | None = None) -> "Settings":
        """
        Loads configuration from a YAML file, merging it with schema defaults.
        """
        path = yaml_path or DEFAULT_CONFIG_PATH
        if path.exists():
            try:
                with open(path) as f:
                    yaml_dict = yaml.safe_load(f) or {}
                # Pydantic parses and validates the dict
                return cls(**yaml_dict)
            except Exception as e:
                logger.warning(f"Failed to load config from {path}. Error: {e}. Using defaults.")
        else:
            logger.warning(f"Config file not found at {path}. Using defaults.")
        return cls()


# Load globally initialized settings
settings = Settings.load_config()

# Standardized folder structure helpers
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / settings.training.output_dir
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure crucial directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Configure logger
logger.remove()

logger.add(
    sys.stderr,
    level=settings.logging.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
logger.info(f"Loaded config from: {DEFAULT_CONFIG_PATH}")
logger.debug(f"Current Settings: {settings.model_dump()}")
