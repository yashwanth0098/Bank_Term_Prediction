import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

from source_main.entity.artifact import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from source_main.entity.config import DataTransformationConfig
from source_main.exception.exception import BankException
from source_main.logging.logging import logging
from source_main.utlis.main_utlis.utlis import load_schema


class DataTransformation:
    """
    Data Transformation Component

    Responsibilities:
    - Read validated data
    - Train-test split
    - Encoding & scaling
    - Handle class imbalance
    - Save transformed data & preprocessor
    """

    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            logging.info("DataTransformation initialized successfully")
        except Exception as e:
            raise BankException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise BankException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline using schema
        """
        try:
            numeric_columns, categorical_columns, _ = load_schema()

            logging.info(f"Numeric columns: {numeric_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_columns),
                    ("cat", categorical_pipeline, categorical_columns),
                ],
                remainder="drop",
            )

            return preprocessor

        except Exception as e:
            raise BankException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            # Read validated data
            df = self.read_data(
                self.data_validation_artifact.validated_file_path
            )

            _, _, target_column = load_schema()

            X = df.drop(columns=[target_column])
            y = df[target_column].map({"no": 0, "yes": 1})

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.data_transformation_config.test_size,
                random_state=self.data_transformation_config.random_state,
                stratify=y,
            )

            logging.info("Train-test split completed")

            # Preprocessing
            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Encoding and scaling completed")

            # Handle imbalance (ONLY on train data)
            smote = SMOTETomek(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                X_train_transformed, y_train
            )

            logging.info("Class imbalance handled using SMOTETomek")

            # Create directories
            os.makedirs(self.data_transformation_config.train_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.test_dir, exist_ok=True)
            os.makedirs(
                self.data_transformation_config.transformed_object_dir,
                exist_ok=True,
            )

            # Save transformed data
            np.save(self.data_transformation_config.X_train_file_path, X_train_balanced)
            np.save(self.data_transformation_config.y_train_file_path, y_train_balanced)
            np.save(self.data_transformation_config.X_test_file_path, X_test_transformed)
            np.save(self.data_transformation_config.y_test_file_path, y_test)

            # Save preprocessor
            joblib.dump(
                preprocessor,
                self.data_transformation_config.preprocessor_object_file_path,
            )

            logging.info("Transformed data and preprocessor saved successfully")
            

            return DataTransformationArtifact(
                X_train_file_path=self.data_transformation_config.X_train_file_path,
                y_train_file_path=self.data_transformation_config.y_train_file_path,
                X_test_file_path=self.data_transformation_config.X_test_file_path,
                y_test_file_path=self.data_transformation_config.y_test_file_path,
                preprocessor_object_file_path=self.data_transformation_config.preprocessor_object_file_path,
            )

        except Exception as e:
            raise BankException(e, sys)
