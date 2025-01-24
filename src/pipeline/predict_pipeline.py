from pathlib import Path

from logging_config import logger

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):


if __name__ == "__main__":
    .........
