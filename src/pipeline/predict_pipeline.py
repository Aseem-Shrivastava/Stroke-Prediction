from pathlib import Path

from logging_config import logger

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

model_path: Path = (MODELS_DIR / "best_tuned_model.pkl",)
