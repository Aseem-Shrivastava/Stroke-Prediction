from pathlib import Path

from logging_config import logger


from src.config import MODELS_DIR, PROCESSED_DATA_DIR


input_path: Path = (RAW_DATA_DIR / "dataset.csv",)
features_path: Path = (PROCESSED_DATA_DIR / "features.csv",)
labels_path: Path = (PROCESSED_DATA_DIR / "labels.csv",)
model_path: Path = (MODELS_DIR / "model.pkl",)


def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step

    raw_data = data_ingestion_step(
        file_path="data/raw/healthcare-dataset-stroke-data.csv"
    )

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(
        clean_data, target_column="SalePrice"
    )

    # Data Transformation Step
    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    # Model Training Step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
