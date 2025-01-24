from abc import ABC, abstractmethod
from logging_config import logger
import pandas as pd
from pathlib import Path
from src.config import EXTRACTED_DATA_DIR
import zipfile


# Abstract class for data ingestor
class DataIngestor(ABC):
    output_dir: Path = EXTRACTED_DATA_DIR

    @abstractmethod
    def ingest(self, filepath: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Concrete class for zip ingestor
class ZipDataIngestor(DataIngestor):
    def ingest(self, filepath: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""
        if not filepath.endswith(".zip"):
            raise ValueError("The provide file is not a .zip file.")

        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        extracted_files = os.listdir(output_dir)
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError(
                "Multiple CSV files found. Please specify which one to use."
            )

        csv_file_path = os.path.join(output_dir, csv_files[0])
        df = pd.read_csv(csv_file_path)
        logger.success("Zipped data ingestion complete.")
        return df


# Implement a Factory to create DataIngestors or return a DataFrame if the input file has ".csv" extension
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(filepath: str):
        logger.info("Data ingestion started...")
        file_extension = os.path.splittext(filepath)[1]
        if file_extension == ".csv":
            df = pd.read_csv(filepath)
            logger.success("Data ingestion complete.")
            return df
        elif file_extension == ".zip":
            return ZipDataIngestor()
        else:
            logger.error(f"No ingestor available for file extension: {file_extension}")
            raise ValueError(
                f"No ingestor available for file extension: {file_extension}"
            )
