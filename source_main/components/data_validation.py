import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from source_main.entity.config import DataValidationConfig, DataIngestionConfig
from source_main.entity.artifact import DataValidationArtifact, DataIngestionArtifact
from source_main.constants.pipelineconstants import SCHEMA_FILE_PATH
from source_main.logging.logging import logging
from source_main.exception.exception import BankException
from source_main.utlis.main_utlis.utlis import read_yaml_file, write_yaml_file


class DataValidation:
    """
    Data Validation Component

    Responsibilities:
    - Schema validation
    - Dataset drift detection using JS divergence
    - Drift report generation
    - Saving validated / invalid datasets
    """

    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact

            self.schema_config: Dict = read_yaml_file(SCHEMA_FILE_PATH)
            if not self.schema_config or "columns" not in self.schema_config:
                raise ValueError("Invalid or empty schema file")

            logging.info("Schema file loaded successfully")

        except Exception as e:
            raise BankException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise BankException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate whether dataframe columns match schema columns
        """
        try:
            expected_columns = set(self.schema_config["columns"].keys())
            actual_columns = set(dataframe.columns)

            logging.info(f"Expected columns: {expected_columns}")
            logging.info(f"Actual columns: {actual_columns}")

            return expected_columns == actual_columns

        except Exception as e:
            raise BankException(e, sys)

    def calculate_js_divergence(
        self,
        expected: pd.Series,
        actual: pd.Series,
        bins: int = 10,
    ) -> float:
        """
        Calculate Jensen–Shannon divergence for numeric columns
        JS divergence ∈ [0, 1]
        """
        try:
            expected = expected.dropna()
            actual = actual.dropna()

            if expected.empty or actual.empty:
                return 0.0

            _, bin_edges = np.histogram(expected, bins=bins)

            expected_hist, _ = np.histogram(
                expected, bins=bin_edges, density=True
            )
            actual_hist, _ = np.histogram(
                actual, bins=bin_edges, density=True
            )

            expected_hist = np.where(expected_hist == 0, 1e-10, expected_hist)
            actual_hist = np.where(actual_hist == 0, 1e-10, actual_hist)

            return float(jensenshannon(expected_hist, actual_hist))

        except Exception as e:
            raise BankException(e, sys)

    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        js_threshold: float = 0.10,
    ) -> bool:
        """
        Detect dataset drift using JS divergence
        """
        try:
            validation_status = True
            drift_report = {}

            for column in self.schema_config["columns"].keys():

                if column not in current_df.columns:
                    logging.warning(f"Column '{column}' missing in current dataset")
                    continue

                if pd.api.types.is_numeric_dtype(base_df[column]):
                    js_value = self.calculate_js_divergence(
                        expected=base_df[column],
                        actual=current_df[column],
                    )

                    drift_detected = js_value >= js_threshold
                    if drift_detected:
                        validation_status = False

                    drift_report[column] = {
                        "js_divergence": js_value,
                        "drift_detected": drift_detected,
                    }
                else:
                    drift_report[column] = {
                        "js_divergence": None,
                        "drift_detected": "Not Applicable (Non-numeric)",
                    }

            drift_report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)

            write_yaml_file(drift_report_path, drift_report)
            logging.info(f"Drift report saved at: {drift_report_path}")

            return validation_status

        except Exception as e:
            raise BankException(e, sys)

    def save_validated_dataset(
        self,
        dataframe: pd.DataFrame,
        validation_status: bool,
    ) -> str:
        try:
            if validation_status:
                save_dir = self.data_validation_config.valid_dir
                file_name = "validated.csv"
            else:
                save_dir = self.data_validation_config.invalid_dir
                file_name = "invalid.csv"

            os.makedirs(save_dir, exist_ok=True)

            file_path = os.path.join(save_dir, file_name)
            dataframe.to_csv(file_path, index=False)

            logging.info(f"Dataset saved at: {file_path}")
            return file_path

        except Exception as e:
            raise BankException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # Current data
            current_df = self.read_data(
                self.data_ingestion_artifact.feature_store_file_path
            )

            # Base data (reference)
            base_df = self.read_data(
                self.data_ingestion_artifact.feature_store_file_path
            )

            # 1. Schema validation
            schema_status = self.validate_number_of_columns(current_df)

            # 2. Drift detection (only if schema is valid)
            drift_status = (
                self.detect_dataset_drift(base_df, current_df)
                if schema_status
                else False
            )

            validation_status = schema_status and drift_status

            validated_file_path = self.save_validated_dataset(
                dataframe=current_df,
                validation_status=validation_status,
            )

            return DataValidationArtifact(
                validation_status=validation_status,
                validated_file_path=validated_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

        except Exception as e:
            raise BankException(e, sys)
